"""Persistent local Chroma collection; embeddings are generated locally."""
from pathlib import Path

import chromadb

from app.config import Settings
from app.models import Passage
from app.rag.chunker import chunk_text
from app.rag.embeddings import embed
from app.rag.loader import UnreadablePDF, load_pdf

COLLECTION = "academic_papers"


class VectorStore:
    def __init__(self, settings: Settings, embedder=embed):
        self.settings = settings
        self.embedder = embedder
        settings.vector_store_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(settings.vector_store_path))
        self.collection = self.client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def ingest(self, reindex: bool = False) -> dict[str, str]:
        self.settings.papers_directory.mkdir(parents=True, exist_ok=True)
        if reindex:
            self.client.delete_collection(COLLECTION)
            self.collection = self.client.get_or_create_collection(
                COLLECTION, metadata={"hnsw:space": "cosine"}
            )
        results = {}
        for path in sorted(self.settings.papers_directory.glob("*.pdf")):
            try:
                metadata, pages = load_pdf(path)
                if self.collection.get(where={"document_id": metadata["document_id"]}, limit=1)["ids"]:
                    results[path.name] = "already indexed"
                    continue
                # Replace an earlier version of this filename when its bytes changed.
                old = self.collection.get(where={"filename": path.name})["ids"]
                if old:
                    self.collection.delete(ids=old)
                ids, texts, metadatas = [], [], []
                for page_number, page_text in pages:
                    for index, text in enumerate(chunk_text(
                        page_text, self.settings.chunk_size, self.settings.chunk_overlap
                    )):
                        chunk_id = f"{metadata['document_id']}:p{page_number}:c{index}"
                        ids.append(chunk_id)
                        texts.append(text)
                        metadatas.append({**metadata, "page_number": page_number,
                                          "chunk_id": chunk_id})
                vectors = self.embedder(texts, self.settings.embedding_model)
                self.collection.add(ids=ids, documents=texts, metadatas=metadatas,
                                    embeddings=vectors)
                results[path.name] = f"indexed {len(ids)} passages"
            except (UnreadablePDF, OSError, ValueError) as exc:
                results[path.name] = f"error: {exc}"
        return results

    def papers(self) -> list[dict]:
        rows = self.collection.get(include=["metadatas"])["metadatas"]
        unique = {item["document_id"]: item for item in rows}
        return sorted(unique.values(), key=lambda item: item["filename"])

    def search(self, query: str, limit: int) -> list[Passage]:
        if not self.collection.count():
            return []
        vector = self.embedder([query], self.settings.embedding_model)[0]
        found = self.collection.query(query_embeddings=[vector], n_results=min(limit, self.collection.count()),
                                      include=["documents", "metadatas", "distances"])
        passages = []
        for text, meta, distance in zip(found["documents"][0], found["metadatas"][0],
                                        found["distances"][0]):
            passages.append(Passage(**meta, text=text,
                                    relevance_score=max(0.0, min(1.0, 1.0 - distance))))
        return passages

    def metadata(self, chunk_id: str) -> dict | None:
        result = self.collection.get(ids=[chunk_id], include=["metadatas"])
        return result["metadatas"][0] if result["metadatas"] else None
