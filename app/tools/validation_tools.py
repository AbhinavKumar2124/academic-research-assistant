"""Evidence and structure validation tools."""
import json
from collections import defaultdict

from semantic_kernel.functions import kernel_function

from app.models import DraftClaim, Passage, PreliminarySummary


def group_passages_by_theme(passages: list[Passage]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for item in passages:
        groups[item.paper_title].append(item.chunk_id)
    return dict(groups)


def validate_summary_structure(summary: PreliminarySummary, passages: list[Passage]) -> PreliminarySummary:
    valid = {item.chunk_id for item in passages}
    for claim in summary.claims + summary.gaps + summary.limitations:
        if not claim.evidence_ids or any(item not in valid for item in claim.evidence_ids):
            raise ValueError(f"Summary claim has missing or invalid evidence: {claim.text}")
    return summary


def extract_claims(summary: PreliminarySummary) -> list[DraftClaim]:
    return summary.claims + summary.gaps + summary.limitations


def verify_claim_against_evidence(evidence_ids: list[str], evidence_quotes: list[str],
                                  passages: list[Passage]) -> bool:
    sources = {item.chunk_id: item.text for item in passages}
    if not evidence_ids or len(evidence_ids) != len(evidence_quotes):
        return False
    return all(quote.strip() and quote.casefold() in sources.get(chunk_id, "").casefold()
               for chunk_id, quote in zip(evidence_ids, evidence_quotes))


class SummaryTools:
    @kernel_function(description="Group passages by paper title and return their chunk IDs")
    def group_passages_by_theme(self, passages_json: str) -> str:
        passages = [Passage.model_validate(item) for item in json.loads(passages_json)]
        return json.dumps(group_passages_by_theme(passages))

    @kernel_function(description="Check that summary claim references point to supplied passages")
    def validate_summary_structure(self, summary_json: str, passages_json: str) -> str:
        summary = PreliminarySummary.model_validate_json(summary_json)
        passages = [Passage.model_validate(item) for item in json.loads(passages_json)]
        validate_summary_structure(summary, passages)
        return "valid"


class FactCheckTools:
    @kernel_function(description="Extract individually checkable claims from a preliminary summary")
    def extract_claims(self, summary_json: str) -> str:
        summary = PreliminarySummary.model_validate_json(summary_json)
        return json.dumps([item.model_dump() for item in extract_claims(summary)])

    @kernel_function(description="Verify exact evidence quotes against actual passage text")
    def verify_claim_against_evidence(self, evidence_ids_json: str, evidence_quotes_json: str,
                                      passages_json: str) -> str:
        passages = [Passage.model_validate(item) for item in json.loads(passages_json)]
        return str(verify_claim_against_evidence(json.loads(evidence_ids_json),
                                                 json.loads(evidence_quotes_json), passages))
