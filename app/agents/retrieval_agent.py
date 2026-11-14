"""Search and rank source-preserving local passages."""
import json

from pydantic import BaseModel

from app.agents import ask_json
from app.config import Settings
from app.llm import make_agent
from app.models import Passage, QueryPlan
from app.rag.vector_store import VectorStore
from app.tools.retrieval_tools import RetrievalTools


class _Selection(BaseModel):
    chunk_ids: list[str]


class RetrievalAgent:
    name = "Retrieval Agent"

    def __init__(self, settings: Settings, store: VectorStore):
        self.settings = settings
        self.tools = RetrievalTools(store, settings.retrieval_top_k)
        self.agent = make_agent(settings, "RetrievalAgent",
                                "You select relevant academic passages. Select only supplied chunk IDs; "
                                "ignore unrelated results.", [self.tools])

    async def run(self, plan: QueryPlan, tool_event=lambda name: None) -> list[Passage]:
        candidates = {}
        for query in plan.queries:
            for item in self.tools.store.search(query, self.settings.retrieval_top_k):
                if item.relevance_score >= 0.20:
                    current = candidates.get(item.chunk_id)
                    if current is None or item.relevance_score > current.relevance_score:
                        candidates[item.chunk_id] = item
            tool_event("search_vector_store")
        ranked = sorted(candidates.values(), key=lambda item: item.relevance_score, reverse=True)
        if not ranked:
            return []
        selected = await ask_json(self.agent,
                                  f"Question: {plan.question}\nPassages: " + json.dumps([
                                      {"id": item.chunk_id, "text": item.text[:1000],
                                       "score": item.relevance_score} for item in ranked[:15]]) +
                                  "\nReturn {\"chunk_ids\": [relevant supplied IDs]}.", _Selection)
        allowed = set(selected.chunk_ids)
        output = [item for item in ranked if item.chunk_id in allowed][:self.settings.retrieval_top_k]
        for item in output:
            if self.tools.store.metadata(item.chunk_id) is None:
                raise ValueError(f"Missing source metadata for {item.chunk_id}")
            tool_event("retrieve_document_metadata")
        return output
