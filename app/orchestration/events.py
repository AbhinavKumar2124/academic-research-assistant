"""Append-only JSONL audit events."""
from pathlib import Path

from app.models import ExecutionEvent, ResearchState


def record_event(state: ResearchState, directory: Path, agent: str, event_type: str,
                 status: str = "OK", tool: str | None = None, error: str | None = None) -> ExecutionEvent:
    event = ExecutionEvent(session_id=state.session_id, agent_name=agent,
                           event_type=event_type, tool_invocation=tool,
                           execution_status=status, error=error)
    state.events.append(event)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "execution.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(event.model_dump_json() + "\n")
    return event
