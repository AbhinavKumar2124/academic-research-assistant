import pytest

from app.agents.retrieval_agent import RetrievalAgent
from app.models import QueryPlan
from conftest import FakeAgent


class StubStore:
    def __init__(self, passage):
        self.passage = passage

    def search(self, query, limit):
        return [self.passage]

    def metadata(self, chunk_id):
        return {"paper_title": self.passage.paper_title} if chunk_id == self.passage.chunk_id else None


@pytest.mark.asyncio
async def test_retrieval_agent(monkeypatch, settings, passage):
    monkeypatch.setattr("app.agents.retrieval_agent.make_agent", lambda *args: FakeAgent(
        {"chunk_ids": [passage.chunk_id, "fabricated"]}))
    tools = []
    output = await RetrievalAgent(settings, StubStore(passage)).run(
        QueryPlan(question="Hospital privacy?", keywords=["hospital"], queries=["privacy"]), tools.append)
    assert output == [passage]
    assert tools == ["search_vector_store", "retrieve_document_metadata"]
