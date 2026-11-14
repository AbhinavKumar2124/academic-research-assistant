import pytest

from app.models import (CheckedClaim, ClaimStatus, FactCheckResult, PipelineStatus,
                        PreliminarySummary, QueryPlan, ResearchReport)
from app.orchestration.engine import ResearchEngine
from app.store import list_sessions


class StubStore:
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["success", "no_evidence", "unsupported"])
async def test_end_to_end_scenarios(monkeypatch, settings, passage, scenario):
    async def connection(_):
        return "ready"
    monkeypatch.setattr("app.orchestration.engine.check_connection", connection)
    engine = ResearchEngine.__new__(ResearchEngine)
    engine.settings = settings
    engine.store = StubStore()

    class Query:
        async def run(self, question, event):
            event("extract_keywords")
            event("validate_and_normalize_queries")
            return QueryPlan(question=question, keywords=["hospital"], queries=["hospital"])

    class Retrieval:
        async def run(self, plan, event):
            event("search_vector_store")
            return [] if scenario == "no_evidence" else [passage]

    class Summary:
        async def run(self, question, passages, event):
            event("group_passages_by_theme")
            event("validate_summary_structure")
            from app.models import DraftClaim
            return PreliminarySummary(overview="One study", claims=[DraftClaim(
                text="Hospital training uses quantum encryption" if scenario == "unsupported"
                else "Training occurs across hospitals", theme="training",
                evidence_ids=[passage.chunk_id])])

    class Critic:
        async def run(self, summary, passages, event):
            event("extract_claims")
            if not summary.claims:
                return FactCheckResult(claims=[])
            event("verify_claim_against_evidence")
            supported = scenario == "success"
            return FactCheckResult(claims=[CheckedClaim(
                text=summary.claims[0].text, theme="training",
                status=ClaimStatus.SUPPORTED if supported else ClaimStatus.UNSUPPORTED,
                evidence_ids=[passage.chunk_id] if supported else [])])

    class Writer:
        async def run(self, question, checked, passages, event):
            from app.tools.citation_tools import generate_research_report
            event("validate_final_citations")
            event("generate_research_report")
            return ResearchReport(question=question,
                                  markdown=generate_research_report(question, checked.claims, passages))

    engine.query, engine.retrieval, engine.summary = Query(), Retrieval(), Summary()
    engine.fact_check, engine.writer = Critic(), Writer()
    state = await engine.run("How do hospitals train models?")
    assert state.status == (PipelineStatus.COMPLETED if scenario == "success"
                            else PipelineStatus.INSUFFICIENT_EVIDENCE)
    assert state.final_report and state.final_report.path
    assert list_sessions(settings.sessions_directory)[0].session_id == state.session_id
    assert (settings.log_directory / "execution.jsonl").exists()
    if scenario == "unsupported":
        body = state.final_report.markdown.split("# Unsupported Claims Removed")[0]
        assert "quantum" not in body
