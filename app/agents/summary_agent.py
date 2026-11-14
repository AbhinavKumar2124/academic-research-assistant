"""Create a citation-linked preliminary literature summary."""
import json

from app.agents import ask_json
from app.config import Settings
from app.llm import make_agent
from app.models import Passage, PreliminarySummary
from app.tools.validation_tools import SummaryTools, group_passages_by_theme, validate_summary_structure


class SummaryAgent:
    name = "Summary Agent"

    def __init__(self, settings: Settings):
        self.tools = SummaryTools()
        self.agent = make_agent(settings, "SummaryAgent",
                                "Summarize only supplied academic evidence. Every individual claim, "
                                "gap and limitation needs one or more exact chunk IDs. Do not infer "
                                "findings absent from the passages.", [self.tools])

    async def run(self, question: str, passages: list[Passage],
                  tool_event=lambda name: None) -> PreliminarySummary:
        groups = group_passages_by_theme(passages)
        tool_event("group_passages_by_theme")
        prompt = (f"Question: {question}\nPaper groups: {json.dumps(groups)}\n"
                  f"Passages: {json.dumps([{'id': p.chunk_id, 'title': p.paper_title, 'text': p.text} for p in passages])}\n"
                  "Return JSON with overview, claims, gaps, limitations. Each item in the last "
                  "three lists has text, theme, evidence_ids. Empty lists are fine.")
        summary = await ask_json(self.agent, prompt, PreliminarySummary)
        validate_summary_structure(summary, passages)
        tool_event("validate_summary_structure")
        return summary
