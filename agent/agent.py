"""LangGraph ReAct ajanı: LLM ↔ araçlar döngüsü.

LLM .env'deki anahtara göre seçilir: önce ANTHROPIC_API_KEY, yoksa OPENAI_API_KEY.
"Nasıl hesapladım" bölümü LLM'e bırakılmaz; ajanın gerçekten çağırdığı araçlardan üretilir.
"""
import json
import os

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.tools import TOOLS
from config import ORTALAMA_YOLCULUK_GELIRI_TL

# Sunucu tarafı yedek modeli (fallbacks) destekleyen modeller; Haiku'da yok
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}

SYSTEM_PROMPT = f"""Sen Martı'nın promosyon bütçesini analiz eden bir iş analistisin ve üst yöneticiyle konuşuyorsun.
Türkçe, kısa ve iş diliyle cevap ver: önce net sonuç (1-2 cümle), sonra en fazla 3-4 madde rakam.
Teknik jargon yerine "ek yolculuk", "boşa giden indirim", "TL başına getiri" gibi ifadeler kullan.

Önemli bağlam:
- Veriler SENTETİKTİR; Martı'nın gerçek verisi değildir. Rakamları Martı hakkında kesin bilgi gibi sunma.
- Her kampanyada hedef kitlenin yarısına rastgele indirim verildi, yarısı kontrol grubu (holdout) olarak kaldı.
  Bu yüzden "indirimli − kontrol" yolculuk oranı farkı, indirimin gerçek (nedensel) etkisidir (uplift).
- Ek yolculuk = indirim olmasaydı gerçekleşmeyecek yolculuk. Ek yolculuk başına maliyet
  {ORTALAMA_YOLCULUK_GELIRI_TL} TL'yi (ortalama yolculuk geliri varsayımı) aşıyorsa kampanya verimsiz sayılır.
- "Boşa giden pay": indirimi kullananların, indirim olmasa da yolculuk yapacak olan kısmı.
- uplift_alt/uplift_ust %95 güven aralığıdır. Aralık sıfırı içeriyorsa (anlamli_etki=false)
  "ek yolculuk yarattığına dair kanıt yok" de; kesin "sıfır" deme.
- Bilmediğin veya veride olmayan bir şeyi uydurma; bilmiyorsan söyle.

Araç seçimi:
- Kampanya etkisi / verimlilik / boşa harcama sorularında campaign_uplift kullan (hepsi için campaign_name="tümü").
- Bütçe dağıtımı sorularında budget_simulator kullan. Sonucu, "yontem" alanındaki mantıkla açıkla:
  simülatör kullanıcı bazında seçer, bu yüzden kampanya ortalamalarıyla farklı görünmesi bir çelişki değildir.
  Dağıtıma girmeyen kampanyalar zaten bütçe almıyor; yeniden simülasyon önerme.
- Diğer sorularda sql_query ile veritabanını sorgula (yalnızca SELECT, SQLite söz dizimi).

Veritabanı şeması:
- users(user_id, sehir, yas_grubu, son_30_gun_yolculuk, uygulama_yasi_gun, kullanilan_hizmet)
  sehir ∈ İstanbul, Ankara, İzmir, Bursa, Antalya; kullanilan_hizmet ∈ scooter, moped, TAG
- campaigns(campaign_id, kampanya_adi, birim_maliyet_tl, hedef_kitle)
- assignments(user_id, campaign_id, treated [1=indirim aldı, 0=kontrol], cost_tl [kullanılan indirimin TL maliyeti])
- outcomes(user_id, campaign_id, rode [14 gün içinde yolculuk yaptı mı 0/1], rides [yolculuk sayısı])
- uplift_scores(user_id, campaign_id, p_treated, p_control, uplift) — modelin kullanıcı bazında tahmini

Cevabın sonuna "Nasıl hesapladım" bölümü EKLEME; sistem bunu otomatik ekliyor."""


def get_llm():
    """Anahtara göre Claude veya OpenAI modeli döner."""
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        model = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")
        fallback = {}
        if model in FALLBACK_MODELS:
            # Güvenlik filtresi bir isteği reddederse sunucu tarafında otomatik yedek modele geçer
            fallback = {"betas": ["server-side-fallback-2026-07-01"], "model_kwargs": {"fallbacks": "default"}}
        return ChatAnthropic(
            model=model,
            max_tokens=16000,
            output_config={"effort": os.getenv("LLM_EFFORT", "low")},  # sohbet için düşük gecikme
            **fallback,
        )
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1"), temperature=0)
    raise RuntimeError("LLM anahtarı bulunamadı. .env dosyasına ANTHROPIC_API_KEY veya OPENAI_API_KEY ekleyin.")


def llm_provider() -> str | None:
    if os.getenv("ANTHROPIC_API_KEY"):
        return f"Claude ({os.getenv('ANTHROPIC_MODEL', 'claude-opus-5-5')})"
    if os.getenv("OPENAI_API_KEY"):
        return f"OpenAI ({os.getenv('OPENAI_MODEL', 'gpt-4.1')})"
    return None


def build_agent(llm=None):
    """agent → (araç çağrısı varsa) tools → agent ... → son cevap."""
    model = (llm or get_llm()).bind_tools(TOOLS)

    def call_model(state: MessagesState):
        return {"messages": [model.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def explain_steps(messages: list[BaseMessage]) -> str:
    """Ajanın bu turda çağırdığı araçlardan "Nasıl hesapladım" bölümünü üretir."""
    lines = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            args = call["args"]
            if call["name"] == "sql_query":
                lines.append(f"- SQL sorgusu:\n\n```sql\n{args.get('query', '').strip()}\n```")
            else:
                arg_text = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
                lines.append(f"- `{call['name']}({arg_text})`")
    if not lines:
        lines.append("- Araç kullanılmadı; cevap genel bağlamdan verildi.")
    return "**Nasıl hesapladım:**\n\n" + "\n".join(lines)


def ask(agent, question: str, history: list[BaseMessage] | None = None) -> tuple[str, list[BaseMessage]]:
    """Soruyu sorar; (cevap metni + nasıl hesapladım, güncel mesaj geçmişi) döner.

    Geçmiş yalnızca sona ekleme ile büyür (önceki mesajlar değiştirilmez).
    """
    history = list(history or [])
    start = len(history)
    result = agent.invoke({"messages": [*history, HumanMessage(question)]}, {"recursion_limit": 12})
    messages = result["messages"]
    new_messages = messages[start:]
    final = next((m for m in reversed(new_messages) if isinstance(m, AIMessage)), None)
    answer = (final.text if final else "").strip() or "Bu soruya bir cevap üretemedim."
    return f"{answer}\n\n{explain_steps(new_messages)}", messages
