"""The five-stage research workflow and its shared state."""
from collections.abc import Callable

from app.agents.fact_check_agent import FactCheckAgent
from app.agents.query_agent import QueryAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.summary_agent import SummaryAgent
from app.agents.writer_agent import WriterAgent
from app.config import Settings
from app.llm import check_connection
from app.models import PipelineStatus, PreliminarySummary, ResearchState
from app.orchestration.events import record_event
from app.rag.vector_store import VectorStore
from app.store import save_session


class ResearchEngine:
    def __init__(self, settings: Settings, store: VectorStore | None = None):
        self.settings = settings
        self.store = store or VectorStore(settings)
        self.query = QueryAgent(settings)
        self.retrieval = RetrievalAgent(settings, self.store)
        self.summary = SummaryAgent(settings)
        self.fact_check = FactCheckAgent(settings)
        self.writer = WriterAgent(settings)

    async def run(self, question: str,
                  on_event: Callable[[str], None] | None = None) -> ResearchState:
        state = ResearchState(question=question.strip())

        def emit(agent: str, kind: str, status: str = "OK", tool: str | None = None,
                 error: str | None = None):
            safe_error = error.replace(self.settings.openrouter_api_key, "[redacted]") if (
                error and self.settings.openrouter_api_key) else error
            record_event(state, self.settings.log_directory, agent, kind, status, tool, safe_error)
            if on_event:
                on_event(f"{agent}: {tool or kind}" + (f" ({status})" if status != "OK" else ""))

        current = "Provider"
        try:
            emit(current, "connectivity_check")
            await check_connection(self.settings)
            current = "Query Agent"
            emit(current, "start")
            state.generated_queries = await self.query.run(
                state.question, lambda name: emit(current, "tool", tool=name))
            emit(current, "complete")
            save_session(state, self.settings.sessions_directory)

            current = "Retrieval Agent"
            emit(current, "start")
            state.retrieved_passages = await self.retrieval.run(
                state.generated_queries, lambda name: emit(current, "tool", tool=name))
            emit(current, f"complete: {len(state.retrieved_passages)} passages")
            save_session(state, self.settings.sessions_directory)

            current = "Summary Agent"
            emit(current, "start")
            if state.retrieved_passages:
                state.preliminary_summary = await self.summary.run(
                    state.question, state.retrieved_passages,
                    lambda name: emit(current, "tool", tool=name))
            else:
                state.preliminary_summary = PreliminarySummary(
                    overview="No relevant indexed passages were found.", claims=[])
                emit(current, "insufficient_evidence")
            state.extracted_claims = (state.preliminary_summary.claims +
                                      state.preliminary_summary.gaps +
                                      state.preliminary_summary.limitations)
            emit(current, "complete")
            save_session(state, self.settings.sessions_directory)

            current = "Fact-Check Agent"
            emit(current, "start")
            state.fact_check_results = await self.fact_check.run(
                state.preliminary_summary, state.retrieved_passages,
                lambda name: emit(current, "tool", tool=name))
            counts = {}
            for claim in state.fact_check_results.claims:
                counts[claim.status.value] = counts.get(claim.status.value, 0) + 1
            emit(current, f"complete: {counts}")
            save_session(state, self.settings.sessions_directory)

            current = "Writer Agent"
            emit(current, "start")
            state.final_report = await self.writer.run(
                state.question, state.fact_check_results, state.retrieved_passages,
                lambda name: emit(current, "tool", tool=name))
            report_path = self.settings.sessions_directory / f"{state.session_id}.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(state.final_report.markdown, encoding="utf-8")
            state.final_report.path = str(report_path)
            state.status = (PipelineStatus.COMPLETED if any(
                claim.status.value in {"SUPPORTED", "PARTIALLY_SUPPORTED"} and
                claim.category == "finding" for claim in state.fact_check_results.claims)
                else PipelineStatus.INSUFFICIENT_EVIDENCE)
            emit(current, "complete")
        except Exception as exc:
            state.status = PipelineStatus.FAILED
            emit(current, "failure", "ERROR", error=str(exc))
        save_session(state, self.settings.sessions_directory)
        return state
