"""Ajan grafını API anahtarı olmadan, senaryolu sahte LLM ile test eder."""
from langchain_core.messages import AIMessage

from agent.agent import ask, build_agent
from fake_llm import ScriptedLLM


def test_agent_runs_tool_and_explains(small_db):
    llm = ScriptedLLM(script=[
        AIMessage("", tool_calls=[{"name": "campaign_uplift", "args": {"campaign_name": "tümü"}, "id": "1"}]),
        AIMessage("Sadakat bonusu en çok boşa harcayan kampanya."),
    ])
    answer, history = ask(build_agent(llm), "Hangi kampanya en çok boşa para harcıyor?")
    assert answer.startswith("Sadakat bonusu")
    assert "Nasıl hesapladım" in answer and "campaign_uplift" in answer
    assert "Sadakat bonusu" in history[2].content  # araç sonucu geçmişte
