"""Extract page text and metadata from PDFs."""
import hashlib
import re
from pathlib import Path

import pymupdf


class UnreadablePDF(ValueError):
    pass


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_pdf(path: Path) -> tuple[dict, list[tuple[int, str]]]:
    try:
        raw = path.read_bytes()
        with pymupdf.open(stream=raw, filetype="pdf") as document:
            title = clean_text(document.metadata.get("title") or "") or path.stem
            pages = [(index + 1, clean_text(page.get_text())) for index, page in enumerate(document)]
    except (OSError, RuntimeError, ValueError) as exc:
        raise UnreadablePDF(f"Cannot read {path.name}: {exc}") from exc
    pages = [(number, text) for number, text in pages if text]
    if not pages:
        raise UnreadablePDF(f"No extractable text in {path.name}; scanned PDFs need OCR")
    return {"document_id": hashlib.sha256(raw).hexdigest()[:20], "paper_title": title,
            "filename": path.name}, pages
