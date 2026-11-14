import pytest

from app.agents.query_agent import QueryAgent
from app.tools.query_tools import extract_keywords, validate_and_normalize_queries
from conftest import FakeAgent


@pytest.mark.asyncio
async def test_query_agent(monkeypatch, settings):
    monkeypatch.setattr("app.agents.query_agent.make_agent", lambda *args: FakeAgent(
        {"queries": ["Federated learning healthcare", "Federated learning healthcare",
                     "Patient privacy hospitals"]}))
    tools = []
    plan = await QueryAgent(settings).run("How does federated learning protect hospital data?", tools.append)
    assert len(plan.queries) == 2
    assert "federated" in plan.keywords
    assert tools == ["extract_keywords", "validate_and_normalize_queries"]


def test_query_tools():
    assert "learning" in extract_keywords("How does learning work?")
    assert validate_and_normalize_queries(["  one query  ", "one query"], "fallback") == ["one query"]
