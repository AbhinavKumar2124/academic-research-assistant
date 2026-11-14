"""Generate focused subqueries from a research question."""
from pydantic import BaseModel, Field

from app.agents import ask_json
from app.config import Settings
from app.llm import make_agent
from app.models import QueryPlan
from app.tools.query_tools import QueryTools, extract_keywords, validate_and_normalize_queries


class _QueryResponse(BaseModel):
    queries: list[str] = Field(min_length=1)


class QueryAgent:
    name = "Query Agent"

    def __init__(self, settings: Settings):
        self.tools = QueryTools()
        self.agent = make_agent(settings, "QueryAgent",
                                "You decompose academic questions into 2-5 concise search dimensions. "
                                "Keep the original subject and avoid inventing findings.", [self.tools])

    async def run(self, question: str, tool_event=lambda name: None) -> QueryPlan:
        if len(question.strip()) < 3:
            raise ValueError("Research question is too short")
        keywords = extract_keywords(question)
        tool_event("extract_keywords")
        result = await ask_json(self.agent, f"Question: {question}\nKeywords: {keywords}\n"
                                "Return {\"queries\": [focused search strings]}.", _QueryResponse)
        queries = validate_and_normalize_queries(result.queries, question)
        tool_event("validate_and_normalize_queries")
        return QueryPlan(question=question.strip(), keywords=keywords, queries=queries)
