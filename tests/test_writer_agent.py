import pytest

from app.agents.writer_agent import WriterAgent
from app.models import CheckedClaim, ClaimStatus, FactCheckResult
from app.tools.citation_tools import generate_research_report, validate_final_citations
from conftest import FakeAgent


@pytest.mark.asyncio
async def test_writer_excludes_unsupported(monkeypatch, settings, passage):
    monkeypatch.setattr("app.agents.writer_agent.make_agent", lambda *args: FakeAgent(
        {"claim_indexes": [0, 1]}))
    claims = FactCheckResult(claims=[
        CheckedClaim(text="Training occurs across hospitals.", theme="method",
                     status=ClaimStatus.SUPPORTED, evidence_ids=[passage.chunk_id]),
        CheckedClaim(text="Quantum encryption is used.", theme="method",
                     status=ClaimStatus.UNSUPPORTED, evidence_ids=[]),
    ])
    tools = []
    report = await WriterAgent(settings).run("Hospital training?", claims, [passage], tools.append)
    main_body = report.markdown.split("# Unsupported Claims Removed")[0]
    assert "Quantum encryption" not in main_body
    assert f"[{passage.chunk_id}]" in main_body
    assert "Quantum encryption" in report.markdown.split("# Unsupported Claims Removed")[1]
    assert tools == ["validate_final_citations", "generate_research_report"]


def test_citation_tools(passage):
    claim = CheckedClaim(text="x", theme="x", status=ClaimStatus.SUPPORTED,
                         evidence_ids=["missing"])
    assert not validate_final_citations([claim], [passage])
    with pytest.raises(ValueError):
        generate_research_report("Question?", [claim], [passage])


@pytest.mark.asyncio
async def test_writer_handles_no_evidence_without_model(monkeypatch, settings):
    monkeypatch.setattr("app.agents.writer_agent.make_agent", lambda *args: FakeAgent())
    report = await WriterAgent(settings).run("Unknown topic?", FactCheckResult(claims=[]), [])
    assert "Insufficient indexed evidence" in report.markdown
    assert "No references available" in report.markdown
