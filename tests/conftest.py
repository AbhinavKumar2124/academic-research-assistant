import json
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.models import Passage


class FakeAgent:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.name = "FakeAgent"

    async def get_response(self, messages):
        value = self.responses.pop(0)
        return SimpleNamespace(content=json.dumps(value))


@pytest.fixture
def settings(tmp_path):
    return Settings(papers_directory=tmp_path / "papers", vector_store_path=tmp_path / "chroma",
                    sessions_directory=tmp_path / "sessions", log_directory=tmp_path / "logs")


@pytest.fixture
def passage():
    return Passage(document_id="paper1", paper_title="Federated Health Study",
                   filename="health.pdf", page_number=2, chunk_id="paper1:p2:c0",
                   text="Federated learning trains across hospitals without moving patient records.",
                   relevance_score=0.9)
