"""Read a saved final Markdown report."""
from pathlib import Path


def read_report(path: Path) -> str:
    return path.read_text(encoding="utf-8")
