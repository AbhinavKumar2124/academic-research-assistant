"""Check each proposed claim against exact retrieved passages."""
import json

from pydantic import BaseModel

from app.agents import ask_json
from app.config import Settings
from app.llm import make_agent
from app.models import CheckedClaim, ClaimStatus, FactCheckResult, Passage, PreliminarySummary
from app.tools.validation_tools import FactCheckTools, extract_claims, verify_claim_against_evidence


class _Verdict(BaseModel):
    status: ClaimStatus
    evidence_ids: list[str]
    evidence_quotes: list[str]
    explanation: str = ""
    corrected_text: str | None = None


class FactCheckAgent:
    name = "Fact-Check Agent"

    def __init__(self, settings: Settings):
        self.tools = FactCheckTools()
        self.agent = make_agent(settings, "FactCheckAgent",
                                "You are a skeptical evidence critic. Supported means the exact claim "
                                "is established by the supplied passages. Supply exact short verbatim "
                                "quotes with matching chunk IDs. Mark partial, unsupported or "
                                "contradicted honestly. Never use outside knowledge.", [self.tools])

    async def run(self, summary: PreliminarySummary, passages: list[Passage],
                  tool_event=lambda name: None) -> FactCheckResult:
        claims = extract_claims(summary)
        tool_event("extract_claims")
        by_id = {item.chunk_id: item for item in passages}
        output = []
        for index, claim in enumerate(claims):
            evidence = [by_id[item] for item in claim.evidence_ids if item in by_id]
            verdict = await ask_json(self.agent,
                                     f"Claim: {claim.text}\nEvidence: " + json.dumps([
                                         {"id": item.chunk_id, "text": item.text}
                                         for item in evidence]) +
                                     "\nReturn JSON: status, evidence_ids, evidence_quotes, "
                                     "explanation, corrected_text. Quotes must be exact substrings.",
                                     _Verdict)
            proof_valid = verify_claim_against_evidence(verdict.evidence_ids,
                                                        verdict.evidence_quotes, evidence)
            tool_event("verify_claim_against_evidence")
            status = verdict.status
            if status in {ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED} and not proof_valid:
                status = ClaimStatus.UNSUPPORTED
            if status == ClaimStatus.PARTIALLY_SUPPORTED and not verdict.corrected_text:
                status = ClaimStatus.UNSUPPORTED
            category = "finding" if index < len(summary.claims) else (
                "gap" if index < len(summary.claims) + len(summary.gaps) else "limitation")
            output.append(CheckedClaim(text=claim.text, theme=claim.theme, status=status,
                                       evidence_ids=verdict.evidence_ids if proof_valid else [],
                                       evidence_quotes=verdict.evidence_quotes if proof_valid else [],
                                       explanation=verdict.explanation,
                                       corrected_text=verdict.corrected_text if proof_valid else None,
                                       category=category))
        return FactCheckResult(claims=output)
