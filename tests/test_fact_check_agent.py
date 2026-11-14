import pytest

from app.agents.fact_check_agent import FactCheckAgent
from app.models import ClaimStatus, DraftClaim, PreliminarySummary
from app.tools.validation_tools import extract_claims, verify_claim_against_evidence
from conftest import FakeAgent


@pytest.mark.asyncio
async def test_fact_check_rejects_unproven_claim(monkeypatch, settings, passage):
    monkeypatch.setattr("app.agents.fact_check_agent.make_agent", lambda *args: FakeAgent(
        {"status": "SUPPORTED", "evidence_ids": [passage.chunk_id],
         "evidence_quotes": ["No such phrase"], "explanation": "seems likely"}))
    summary = PreliminarySummary(overview="", claims=[DraftClaim(
        text="Hospitals use quantum encryption", theme="privacy", evidence_ids=[passage.chunk_id])])
    tools = []
    checked = await FactCheckAgent(settings).run(summary, [passage], tools.append)
    assert checked.claims[0].status == ClaimStatus.UNSUPPORTED
    assert tools == ["extract_claims", "verify_claim_against_evidence"]


def test_fact_check_tools(passage):
    summary = PreliminarySummary(overview="", claims=[DraftClaim(
        text="x", theme="x", evidence_ids=[passage.chunk_id])])
    assert len(extract_claims(summary)) == 1
    assert verify_claim_against_evidence([passage.chunk_id], ["across hospitals"], [passage])
    assert not verify_claim_against_evidence([passage.chunk_id], ["not present"], [passage])
