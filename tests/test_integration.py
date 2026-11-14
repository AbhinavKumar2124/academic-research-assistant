"""A live Ollama run on a temporary synthetic paper, never committed as research."""
import pymupdf
import pytest

from app.config import Settings, get_settings
from app.llm import check_connection
from app.orchestration.engine import ResearchEngine
from app.rag.vector_store import VectorStore


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_provider_and_research(tmp_path):
    default = get_settings()
    settings = Settings(llm_provider=default.llm_provider,
                        ollama_base_url=default.ollama_base_url, ollama_model=default.ollama_model,
                        openrouter_api_key=default.openrouter_api_key,
                        openrouter_model=default.openrouter_model,
                        openrouter_base_url=default.openrouter_base_url,
                        embedding_model=default.embedding_model,
                        papers_directory=tmp_path / "papers",
                        vector_store_path=tmp_path / "chroma",
                        sessions_directory=tmp_path / "sessions",
                        log_directory=tmp_path / "logs", retrieval_top_k=3)
    await check_connection(settings)
    settings.papers_directory.mkdir()
    path = settings.papers_directory / "synthetic_study.pdf"
    with pymupdf.open() as pdf:
        pdf.set_metadata({"title": "Synthetic Hospital Federated Learning Study"})
        page = pdf.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 700),
                            "This synthetic study evaluates federated learning across three hospitals. "
                            "Each hospital trains a local model while patient records remain on site. "
                            "The study reports comparable accuracy to central training on its test set. "
                            "A limitation is that only three hospitals were studied.", fontsize=11)
        pdf.save(path)
    store = VectorStore(settings)
    assert "indexed" in store.ingest()[path.name]
    state = await ResearchEngine(settings, store).run(
        "What does the synthetic study report about federated learning and patient records?")
    assert state.final_report is not None, state.events[-1].error if state.events else "no events"
    assert "synthetic_study.pdf" in state.final_report.markdown
