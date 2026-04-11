from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from kume.domain.tools import ask_recommendation as domain_ask_recommendation


class AskRecommendationInput(BaseModel):
    query: str = Field(description="The user's nutrition question")


class AskRecommendationTool(BaseTool):
    name: str = "ask_recommendation"
    description: str = (
        "Get personalized nutrition recommendations based on the user's question about diet, meals, or health goals"
    )
    args_schema: type[BaseModel] = AskRecommendationInput
    llm: BaseChatModel = Field(exclude=True)

    def _run(self, query: str) -> str:
        return domain_ask_recommendation(query, llm_call=self._call_llm)

    def _call_llm(self, prompt: str) -> str:
        response = self.llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        if isinstance(content, list):
            return "".join(
                str(b.get("text", "")) if isinstance(b, dict) and b.get("type") == "text" else str(b) for b in content
            )
        return str(content or "")
