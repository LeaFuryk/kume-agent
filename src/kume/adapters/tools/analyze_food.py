from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from kume.domain.tools import analyze_food as domain_analyze_food


class AnalyzeFoodInput(BaseModel):
    description: str = Field(description="Description of the food to analyze")


class AnalyzeFoodTool(BaseTool):
    name: str = "analyze_food"
    description: str = "Analyze a food item or meal for nutritional content and alignment with health goals"
    args_schema: type[BaseModel] = AnalyzeFoodInput
    llm: BaseChatModel = Field(exclude=True)

    def _run(self, description: str) -> str:
        return domain_analyze_food(description, llm_call=self._call_llm)

    def _call_llm(self, prompt: str) -> str:
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return str(response.content)
