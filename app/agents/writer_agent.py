"""Assemble the final report from verified claims only."""
import json

from pydantic import BaseModel

from app.agents import ask_json
from app.config import Settings
from app.llm import make_agent
from app.models import FactCheckResult, Passage, ResearchReport
from app.tools.citation_tools import WriterTools, generate_research_report, validate_final_citations


class _Order(BaseModel):
    claim_indexes: list[int]


class WriterAgent:
    name = "Writer Agent"

    def __init__(self, settings: Settings):
        self.tools = WriterTools()
        self.agent = make_agent(settings, "WriterAgent",
                                "Organize verified academic findings for a readable literature review. "
                                "Never add new factual content or unsupported claims.", [self.tools])

    async def run(self, question: str, checked: FactCheckResult, passages: list[Passage],
                  tool_event=lambda name: None) -> ResearchReport:
        if not validate_final_citations(checked.claims, passages):
            raise ValueError("Citations do not match retrieved passages")
        tool_event("validate_final_citations")
        if not checked.claims:
            markdown = generate_research_report(question, [], passages)
            tool_event("generate_research_report")
            return ResearchReport(question=question, markdown=markdown)
        order = await ask_json(self.agent,
                               f"Question: {question}\nChecked claims: " + json.dumps([
                                   {"index": index, "text": claim.corrected_text or claim.text,
                                    "status": claim.status.value, "theme": claim.theme}
                                   for index, claim in enumerate(checked.claims)]) +
                               "\nReturn {\"claim_indexes\": [all indexes in a useful order]}. "
                               "Include every index exactly once.", _Order)
        indexes = order.claim_indexes
        if sorted(indexes) != list(range(len(checked.claims))):
            raise ValueError("Writer returned an invalid claim ordering")
        markdown = generate_research_report(question, [checked.claims[i] for i in indexes], passages)
        tool_event("generate_research_report")
        return ResearchReport(question=question, markdown=markdown)
