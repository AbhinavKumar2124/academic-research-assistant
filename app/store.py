"""JSON session persistence; no database server required."""
from pathlib import Path

from app.models import ResearchState


def save_session(state: ResearchState, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{state.session_id}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def list_sessions(directory: Path) -> list[ResearchState]:
    return [ResearchState.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json"), reverse=True)] if directory.exists() else []
