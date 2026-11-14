import pytest

from app.agents.summary_agent import SummaryAgent
from app.models import DraftClaim, PreliminarySummary
from app.tools.validation_tools import group_passages_by_theme, validate_summary_structure
from conftest import FakeAgent


@pytest.mark.asyncio
async def test_summary_agent(monkeypatch, settings, passage):
    draft = {"text": "Training occurs across hospitals.", "theme": "distributed training",
             "evidence_ids": [passage.chunk_id]}
    monkeypatch.setattr("app.agents.summary_agent.make_agent", lambda *args: FakeAgent(
        {"overview": "One paper", "claims": [draft], "gaps": [], "limitations": []}))
    tools = []
    summary = await SummaryAgent(settings).run("Hospital training?", [passage], tools.append)
    assert summary.claims[0].evidence_ids == [passage.chunk_id]
    assert tools == ["group_passages_by_theme", "validate_summary_structure"]


def test_summary_tools(passage):
    assert group_passages_by_theme([passage]) == {passage.paper_title: [passage.chunk_id]}
    bad = PreliminarySummary(overview="", claims=[DraftClaim(text="x", theme="x",
                                                           evidence_ids=["missing"])])
    with pytest.raises(ValueError):
        validate_summary_structure(bad, [passage])
