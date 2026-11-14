import pymupdf

from app.rag.chunker import chunk_text
from app.rag.embeddings import embed
from app.rag.loader import UnreadablePDF, load_pdf
from app.rag.vector_store import VectorStore


def test_chunking():
    chunks = chunk_text("one two three four five six", 12, 3)
    assert len(chunks) >= 2
    assert all(chunks)


def test_embedding_adapter(monkeypatch):
    class StubModel:
        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings
            return type("Array", (), {"tolist": lambda self: [[1.0, 0.0] for _ in texts]})()
    monkeypatch.setattr("app.rag.embeddings.get_model", lambda name: StubModel())
    assert embed(["a"], "local") == [[1.0, 0.0]]


def test_pdf_ingestion_and_retrieval(settings):
    settings.papers_directory.mkdir()
    pdf_path = settings.papers_directory / "sample.pdf"
    with pymupdf.open() as pdf:
        for text in ["Federated learning trains across hospitals. " * 8,
                     "Patient records remain in each hospital. " * 8]:
            page = pdf.new_page()
            page.insert_text((50, 50), text[:100])
        pdf.save(pdf_path)
    metadata, pages = load_pdf(pdf_path)
    assert len(pages) == 2 and metadata["filename"] == "sample.pdf"
    def fake_embed(texts, model):
        return [[1.0, 0.0] for _ in texts]
    store = VectorStore(settings, embedder=fake_embed)
    assert "indexed" in store.ingest()["sample.pdf"]
    assert store.ingest()["sample.pdf"] == "already indexed"
    found = store.search("hospital", 3)
    assert found and all(item.page_number in {1, 2} for item in found)
    assert store.metadata(found[0].chunk_id)["filename"] == "sample.pdf"
    assert "indexed" in store.ingest(reindex=True)["sample.pdf"]


def test_unreadable_pdf(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"not a pdf")
    try:
        load_pdf(path)
        assert False
    except UnreadablePDF:
        pass
