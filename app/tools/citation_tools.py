"""Citation checking and final report rendering."""
import json

from semantic_kernel.functions import kernel_function

from app.models import CheckedClaim, ClaimStatus, Passage


def validate_final_citations(claims: list[CheckedClaim], passages: list[Passage]) -> bool:
    valid = {item.chunk_id for item in passages}
    return all(claim.evidence_ids and all(item in valid for item in claim.evidence_ids)
               for claim in claims if claim.status in {ClaimStatus.SUPPORTED,
                                                       ClaimStatus.PARTIALLY_SUPPORTED})


def generate_research_report(question: str, claims: list[CheckedClaim],
                             passages: list[Passage]) -> str:
    if not validate_final_citations(claims, passages):
        raise ValueError("Final report has invalid citations")
    source = {item.chunk_id: item for item in passages}
    approved = [item for item in claims if item.status in
                {ClaimStatus.SUPPORTED, ClaimStatus.PARTIALLY_SUPPORTED} and item.evidence_ids]
    rejected = [item for item in claims if item.status in
                {ClaimStatus.UNSUPPORTED, ClaimStatus.CONTRADICTED}]
    def render(items):
        return "\n".join("- " + (item.corrected_text or item.text) + " " +
                         " ".join(f"[{chunk_id}]" for chunk_id in item.evidence_ids)
                         for item in items) or "No supported evidence identified."
    findings = [item for item in approved if item.category == "finding"]
    gaps = [item for item in approved if item.category == "gap"]
    limits = [item for item in approved if item.category == "limitation"]
    themes = list(dict.fromkeys(item.theme for item in findings))
    thematic_review = "\n".join(
        f"### {theme}\n{render([item for item in findings if item.theme == theme])}"
        for theme in themes) or "No supported evidence identified."
    major_findings = render([next(item for item in findings if item.theme == theme)
                             for theme in themes])
    used_ids = list(dict.fromkeys(chunk_id for item in approved for chunk_id in item.evidence_ids))
    references = "\n".join(f"- [{chunk_id}] {source[chunk_id].paper_title}, "
                            f"{source[chunk_id].filename}, p. {source[chunk_id].page_number}."
                            for chunk_id in used_ids) or "No references available."
    return (f"# Research Question\n{question}\n\n# Introduction\n"
            f"This review draws on {len({source[item].document_id for item in used_ids})} indexed paper(s)."
            f"\n\n# Thematic Literature Review\n{thematic_review}\n\n# Major Findings\n"
            f"{major_findings}\n\n# Research Gaps\n"
            f"{render(gaps) if gaps else 'No research gap was stated in the retrieved evidence.'}"
            f"\n\n# Limitations\n"
            f"{render(limits) if limits else 'No limitation was stated in the retrieved evidence.'}"
            f"\n\n# Conclusion\n"
            f"{'The indexed evidence supports the cited findings above; interpret them within the stated limitations.' if findings else 'Insufficient indexed evidence to answer the question.'}"
            f"\n\n# References\n{references}\n\n# Unsupported Claims Removed\n"
            + ("\n".join(f"- {item.text} ({item.status})" for item in rejected)
               or "None.") + "\n")


class WriterTools:
    @kernel_function(description="Validate all approved claim citations against retrieved passage IDs")
    def validate_final_citations(self, claims_json: str, passages_json: str) -> str:
        claims = [CheckedClaim.model_validate(item) for item in json.loads(claims_json)]
        passages = [Passage.model_validate(item) for item in json.loads(passages_json)]
        return str(validate_final_citations(claims, passages))

    @kernel_function(description="Generate a research report using only checked claims and real citations")
    def generate_research_report(self, question: str, claims_json: str, passages_json: str) -> str:
        claims = [CheckedClaim.model_validate(item) for item in json.loads(claims_json)]
        passages = [Passage.model_validate(item) for item in json.loads(passages_json)]
        return generate_research_report(question, claims, passages)
