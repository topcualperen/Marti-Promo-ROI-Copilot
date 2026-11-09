"""API anahtarı olmadan ajan grafını test etmek için senaryolu sahte LLM."""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedLLM(BaseChatModel):
    """Sırayla önceden yazılmış AIMessage'ları döner."""
    script: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg: AIMessage = self.script.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])
