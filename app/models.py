"""Validated data passed between the five agents."""
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class PipelineStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    FAILED = "FAILED"


class ClaimStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class QueryPlan(BaseModel):
    question: str = Field(min_length=3)
    keywords: list[str]
    queries: list[str] = Field(min_length=1, max_length=6)


class Passage(BaseModel):
    document_id: str
    paper_title: str
    filename: str
    page_number: int = Field(ge=1)
    chunk_id: str
    text: str
    relevance_score: float = Field(ge=0, le=1)


class DraftClaim(BaseModel):
    text: str
    theme: str
    evidence_ids: list[str]


class PreliminarySummary(BaseModel):
    overview: str
    claims: list[DraftClaim]
    gaps: list[DraftClaim] = Field(default_factory=list)
    limitations: list[DraftClaim] = Field(default_factory=list)


class CheckedClaim(BaseModel):
    text: str
    theme: str
    status: ClaimStatus
    evidence_ids: list[str]
    evidence_quotes: list[str] = Field(default_factory=list)
    explanation: str = ""
    corrected_text: str | None = None
    category: str = "finding"


class FactCheckResult(BaseModel):
    claims: list[CheckedClaim]


class ResearchReport(BaseModel):
    question: str
    markdown: str
    path: str | None = None


class ExecutionEvent(BaseModel):
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: str
    event_type: str
    tool_invocation: str | None = None
    execution_status: str
    error: str | None = None


class ResearchState(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid4().hex)
    question: str
    generated_queries: QueryPlan | None = None
    retrieved_passages: list[Passage] = Field(default_factory=list)
    preliminary_summary: PreliminarySummary | None = None
    extracted_claims: list[DraftClaim] = Field(default_factory=list)
    fact_check_results: FactCheckResult | None = None
    final_report: ResearchReport | None = None
    events: list[ExecutionEvent] = Field(default_factory=list)
    status: PipelineStatus = PipelineStatus.RUNNING
