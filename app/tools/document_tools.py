"""Small document entry points for the CLI and notebook."""
from pathlib import Path

from app.rag.loader import load_pdf


def inspect_pdf(path: Path) -> dict:
    metadata, pages = load_pdf(path)
    return {**metadata, "readable_pages": len(pages)}
