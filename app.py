"""Martı Promo-ROI Copilot — tek sayfalık Streamlit arayüzü.

Çalıştırma: python data/generate.py && streamlit run app.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import inspect

from agent.agent import ask, build_agent, llm_provider
from config import (GERCEK_PROMOSYON_GELIR_PAYI, GERCEK_PROMOSYON_MUSD,
                    ORTALAMA_YOLCULUK_GELIRI_TL, get_engine)
from model.budget import campaign_stats, compare, overall_waste_share

ORNEK_SORULAR = [
    "Hangi kampanya en çok boşa para harcıyor?",
    "Bursa'da hangi kampanya ek yolculuk yaratmadı?",
    "100.000 TL'lik bütçeyi nasıl dağıtmalıyım?",
    "Hoş geldin kampanyası gerçekten işe yarıyor mu?",
]
RENK_NORMAL, RENK_VERIMSIZ, RENK_ESIK = "#5b7a99", "#d03b3b", "#898781"
EKSEN_KESME_TL = 400

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.kpi {border: 1px solid #e6e4df; border-radius: 10px; padding: 18px 20px; background: #fcfcfb; min-height: 185px;}
.kpi.hero {border-color: #0b0b0b; border-width: 2px;}
.kpi .label {font-size: 0.9rem; color: #52514e; margin-bottom: 4px;}
.kpi .value {font-size: 3rem; font-weight: 700; color: #0b0b0b; line-height: 1.1; white-space: nowrap;}
.kpi .value small {font-size: 1.4rem; font-weight: 600;}
.kpi .sub {font-size: 0.85rem; color: #52514e; margin-top: 6px;}
</style>
"""


def tl(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".") + " TL"


def musd(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",") + " M$"


def kpi(label: str, value: str, sub: str, hero: bool = False) -> str:
    cls = "kpi hero" if hero else "kpi"
    return f'<div class="{cls}"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>'


@st.cache_data
def load_numbers():
    stats = campaign_stats()
    return stats, overall_waste_share(stats), compare()


@st.cache_resource
def load_agent():
    return build_agent()


def cost_chart(stats: pd.DataFrame) -> go.Figure:
    df = stats.sort_values("ek_yolculuk_basi_maliyet_tl", ascending=True)
    cost = df["ek_yolculuk_basi_maliyet_tl"]
    colors = [RENK_VERIMSIZ if v else RENK_NORMAL for v in df["verimsiz"]]
    labels = [
        (f"{tl(c)}  ›" if c > EKSEN_KESME_TL else tl(c)) + ("  · verimsiz" if v else "")
        for c, v in zip(cost, df["verimsiz"])
    ]
    fig = go.Figure(go.Bar(
        y=df["kampanya_adi"], x=cost.clip(upper=EKSEN_KESME_TL), orientation="h",
        marker=dict(color=colors, cornerradius=4), text=labels, textposition="outside",
        cliponaxis=False, textfont=dict(color="#0b0b0b"),
        customdata=df[["harcama_tl", "ek_yolculuk", "bosa_giden_pay", "ek_yolculuk_basi_maliyet_tl"]],
        hovertemplate="<b>%{y}</b><br>Ek yolculuk başına: %{customdata[3]:,.0f} TL"
                      "<br>Harcama: %{customdata[0]:,.0f} TL<br>Ek yolculuk: %{customdata[1]:,.0f}"
                      "<br>Boşa giden pay: %{customdata[2]:.0%}<extra></extra>",
    ))
    fig.add_vline(x=ORTALAMA_YOLCULUK_GELIRI_TL, layer="below",
                  line=dict(color=RENK_ESIK, dash="dash", width=1.5))
    fig.add_annotation(x=ORTALAMA_YOLCULUK_GELIRI_TL, y=1.0, yref="paper", yanchor="bottom", yshift=6,
                       showarrow=False, text=f"Ortalama yolculuk geliri ({ORTALAMA_YOLCULUK_GELIRI_TL} TL)",
                       font=dict(color="#52514e", size=12), xanchor="center")
    fig.update_layout(
        height=350, margin=dict(l=10, r=10, t=40, b=40), bargap=0.35,
        plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        xaxis=dict(range=[0, EKSEN_KESME_TL * 1.55], showgrid=True, gridcolor="#eeede9",
                   tickvals=[0, 100, 200, 300, 400], ticksuffix=" TL", zeroline=False,
                   title=dict(text=f"Eksen {EKSEN_KESME_TL} TL'de kesildi; › işaretli çubuklar daha uzun",
                              font=dict(size=11, color="#898781"))),
        yaxis=dict(title=None, tickfont=dict(size=13, color="#0b0b0b")),
        font=dict(family="sans-serif"),
    )
    return fig


def sidebar(provider: str | None) -> None:
    with st.sidebar:
        st.markdown("**Not:** Tüm veriler sentetiktir; Martı'nın gerçek verisi kullanılmamıştır.")
        st.caption("Rakamlar yöntemi göstermek içindir, Martı hakkında bir iddia değildir.")
        st.divider()
        st.caption(f"Yapay zekâ modeli: {provider or 'tanımlı değil (.env)'}")
        st.caption("Ölçüm: her kampanyada %50 indirim / %50 kontrol grubu (rastgele atama).")


def chat_section() -> None:
    st.subheader("Copilot'a sorun")
    if not llm_provider():
        st.info("Sohbet için .env dosyasına ANTHROPIC_API_KEY veya OPENAI_API_KEY ekleyin.")
        return
    st.session_state.setdefault("display", [])   # ekranda gösterilen (soru, cevap) çiftleri
    st.session_state.setdefault("history", [])   # ajanın mesaj geçmişi

    cols = st.columns(len(ORNEK_SORULAR))
    clicked = next((q for c, q in zip(cols, ORNEK_SORULAR) if c.button(q, width="stretch")), None)
    question = st.chat_input("Promosyon bütçesi hakkında bir soru yazın…") or clicked

    for role, content in st.session_state["display"]:
        st.chat_message(role).markdown(content)

    if question:
        st.chat_message("user").markdown(question)
        with st.chat_message("assistant"), st.spinner("Veriye bakıyorum…"):
            try:
                answer, history = ask(load_agent(), question, st.session_state["history"])
                st.session_state["history"] = history
            except Exception as e:  # demo sırasında ağ/anahtar hatası sayfayı çökertmesin
                answer = f"Şu an cevap veremiyorum: {e}"
            st.markdown(answer)
        st.session_state["display"] += [("user", question), ("assistant", answer)]


def main() -> None:
    st.set_page_config(page_title="Martı Promo-ROI Copilot", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    if not inspect(get_engine()).has_table("uplift_scores"):
        st.error("Veri bulunamadı. Önce şunu çalıştırın: `python data/generate.py`")
        st.stop()

    stats, waste, comp = load_numbers()
    sidebar(llm_provider())

    st.title("Promo-ROI Copilot")
    st.markdown("Promosyon bütçesinin hangi kısmı **gerçekten ek yolculuk getiriyor**, hangisi zaten "
                "yolculuk yapacak kullanıcıya gidiyor?")

    savings = comp["tasarruf_orani"]
    saved_musd = musd(GERCEK_PROMOSYON_MUSD * savings, 1)
    savings_sub = (f"{tl(comp['mevcut_rastgele']['harcanan_tl'])} yerine "
                   f"{tl(comp['ayni_sonuc_icin_gereken_butce_tl'])}. "
                   f"{musd(GERCEK_PROMOSYON_MUSD)} ölçeğinde ≈ {saved_musd} (örnek hesap)")
    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Toplam promosyon (2026 H1)", musd(GERCEK_PROMOSYON_MUSD),
                    f"Gelirin ~%{GERCEK_PROMOSYON_GELIR_PAYI * 100:.0f}'i"), unsafe_allow_html=True)
    c2.markdown(kpi("Boşa giden tahmini pay", f"%{waste * 100:.0f}",
                    "İndirimi kullananların bu kadarı indirimsiz de yolculuk yapacaktı"),
                unsafe_allow_html=True)
    c3.markdown(kpi("Aynı sonuç, daha az bütçe", f"%{savings * 100:.0f}<small> daha az</small>",
                    savings_sub, hero=True), unsafe_allow_html=True)

    st.write("")
    st.subheader("Kampanya bazında ek yolculuk başına maliyet")
    st.caption("Kırmızı: ek yolculuk başına maliyet yolculuk gelirini aşıyor veya etki istatistiksel "
               "olarak sıfırdan ayırt edilemiyor.")
    st.plotly_chart(cost_chart(stats), width="stretch", config={"displayModeBar": False})

    chat_section()


main()
