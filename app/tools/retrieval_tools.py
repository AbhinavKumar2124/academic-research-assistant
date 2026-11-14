"""Local vector search and source metadata tools."""
import json

from semantic_kernel.functions import kernel_function

from app.rag.vector_store import VectorStore


class RetrievalTools:
    def __init__(self, store: VectorStore, top_k: int):
        self.store = store
        self.top_k = top_k

    @kernel_function(description="Search indexed academic PDF passages for a query")
    def search_vector_store(self, query: str) -> str:
        return json.dumps([item.model_dump() for item in self.store.search(query, self.top_k)])

    @kernel_function(description="Retrieve the paper metadata for a passage chunk ID")
    def retrieve_document_metadata(self, chunk_id: str) -> str:
        return json.dumps(self.store.metadata(chunk_id))
