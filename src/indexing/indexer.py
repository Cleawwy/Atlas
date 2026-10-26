import os
import sys
import json
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import MilvusClient

from src.ingestion.overpass_client import fetch_karlsruhe_places
from src.ingestion.document_builder import build_place_documents, PlaceDocument

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "atlas_llama_milvus.db",
)
COLLECTION_NAME = "karlsruhe_places"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class AtlasIndexer:
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.db_path = db_path
        self.model_name = model_name
        self.collection_name = COLLECTION_NAME

        print(f"[Indexer] Loading embedding model '{self.model_name}' via LlamaIndex...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.model_name,
            embed_batch_size=64,
        )

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.client = MilvusClient(uri=self.db_path)

        self.vector_store: Optional[MilvusVectorStore] = None
        self.index: Optional[VectorStoreIndex] = None

        if self.client.has_collection(self.collection_name):
            self._attach_existing_index()

    def _ensure_loaded(self):
        """Ensures the Milvus Lite collection is in loaded memory state for search/query."""
        try:
            self.client.load_collection(self.collection_name)
        except Exception:
            pass

    def _attach_existing_index(self):
        """Attaches LlamaIndex VectorStoreIndex to an already created Milvus collection."""
        self.vector_store = MilvusVectorStore(
            uri=self.db_path,
            collection_name=self.collection_name,
            dim=EMBEDDING_DIM,
            similarity_metric="COSINE",
            overwrite=False,
        )
        self._ensure_loaded()
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=self.embed_model,
        )

    def init_collection(self, drop_existing: bool = False):
        """Creates or resets the Milvus Lite collection via LlamaIndex MilvusVectorStore."""
        if drop_existing and self.client.has_collection(self.collection_name):
            print(f"[Indexer] Dropping existing collection '{self.collection_name}'...")
            self.client.drop_collection(self.collection_name)
            self.vector_store = None
            self.index = None

        if not self.client.has_collection(self.collection_name):
            print(f"[Indexer] Initializing MilvusVectorStore '{self.collection_name}' (dim={EMBEDDING_DIM}, metric=COSINE)...")
            self.vector_store = MilvusVectorStore(
                uri=self.db_path,
                collection_name=self.collection_name,
                dim=EMBEDDING_DIM,
                similarity_metric="COSINE",
                overwrite=drop_existing,
            )
            self._ensure_loaded()
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model,
            )
            print("[Indexer] Vector store initialized successfully.")
        else:
            if self.index is None:
                self._attach_existing_index()
            print(f"[Indexer] Collection '{self.collection_name}' already exists.")

    def index_documents(self, documents: List[PlaceDocument], batch_size: int = 64) -> int:
        """
        Embeds place documents and indexes them into Milvus Lite via LlamaIndex.
        Preserves all structured metadata fields for downstream spatial reranking.
        """
        total_docs = len(documents)
        print(f"[Indexer] Preparing {total_docs} documents for LlamaIndex indexing...")

        llama_docs = []
        for doc in documents:
            address_str = doc.metadata.get("address", "") or f"{doc.metadata.get('street', '')} {doc.metadata.get('housenumber', '')}".strip()
            extra_info = {
                "id": str(doc.id),
                "name": doc.name,
                "category": doc.category,
                "subcategory": doc.subcategory,
                "latitude": float(doc.latitude),
                "longitude": float(doc.longitude),
                "address": address_str,
                "opening_hours": str(doc.metadata.get("opening_hours", "")),
                "wheelchair": str(doc.metadata.get("wheelchair", "")),
                "website": str(doc.metadata.get("website", "")),
                "phone": str(doc.metadata.get("phone", "")),
            }
            llama_doc = Document(
                text=doc.text,
                doc_id=str(doc.id),
                extra_info=extra_info,
            )
            llama_docs.append(llama_doc)

        print(f"[Indexer] Initializing fresh MilvusVectorStore at {self.db_path}...")
        self.vector_store = MilvusVectorStore(
            uri=self.db_path,
            collection_name=self.collection_name,
            dim=EMBEDDING_DIM,
            similarity_metric="COSINE",
            overwrite=True,
        )
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        print(f"[Indexer] Generating embeddings & building VectorStoreIndex via LlamaIndex...")
        self.index = VectorStoreIndex.from_documents(
            llama_docs,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )
        self._ensure_loaded()
        print(f"[Indexer] Successfully indexed {len(llama_docs)} places into {self.db_path}.")
        return len(llama_docs)

    def add_single_document(
        self,
        name: str,
        text: str,
        latitude: float,
        longitude: float,
        category: str = "custom",
        subcategory: str = "place",
        address: str = "",
        opening_hours: str = "",
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Embeds a single custom place document and appends it to Milvus Lite via LlamaIndex.
        Does not drop or reset existing records.
        """
        if self.index is None:
            self.init_collection(drop_existing=False)

        if not doc_id:
            import uuid
            doc_id = f"custom_{uuid.uuid4().hex[:8]}"

        extra_info = {
            "id": doc_id,
            "name": name,
            "category": category,
            "subcategory": subcategory,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "address": address,
            "opening_hours": opening_hours,
        }
        llama_doc = Document(
            text=text,
            doc_id=doc_id,
            extra_info=extra_info,
        )

        self.index.insert(llama_doc)
        self._ensure_loaded()
        print(f"[Indexer] Document '{name}' (ID: {doc_id}) added to Milvus Lite via LlamaIndex.")
        return {
            "id": doc_id,
            "name": name,
            "category": category,
            "subcategory": subcategory,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "address": address,
            "opening_hours": opening_hours,
            "text": text,
        }

    def search(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search using LlamaIndex VectorStoreIndex retriever,
        returning candidates formatted for downstream GeoRetriever spatial reranking.
        """
        if self.index is None:
            if self.client.has_collection(self.collection_name):
                self._attach_existing_index()
            else:
                return []

        self._ensure_loaded()

        fetch_k = limit * 2 if category else limit
        retriever = self.index.as_retriever(similarity_top_k=fetch_k)
        nodes = retriever.retrieve(query)

        hits = []
        for n in nodes:
            meta = n.node.metadata
            cat = meta.get("category", "")
            if category and cat != category:
                continue

            doc_id = meta.get("id") or getattr(n.node, "ref_doc_id", None) or n.node.id_
            hits.append({
                "id": doc_id,
                "score": float(n.score) if n.score is not None else 0.0,
                "name": meta.get("name", ""),
                "category": cat,
                "subcategory": meta.get("subcategory", ""),
                "latitude": float(meta.get("latitude", 0.0)),
                "longitude": float(meta.get("longitude", 0.0)),
                "address": meta.get("address", ""),
                "opening_hours": meta.get("opening_hours", ""),
                "text": n.node.get_content() or meta.get("text", ""),
            })
            if len(hits) >= limit:
                break

        return hits

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics of the current Milvus collection."""
        if not self.client.has_collection(self.collection_name):
            return {
                "collection_name": self.collection_name,
                "total_entities": 0,
                "dimension": EMBEDDING_DIM,
            }

        self._ensure_loaded()
        res = self.client.query(
            collection_name=self.collection_name,
            filter='id != ""',
            limit=16384,
            output_fields=["id"],
        )
        return {
            "collection_name": self.collection_name,
            "total_entities": len(res),
            "dimension": EMBEDDING_DIM,
            "metric_type": "COSINE",
        }


def build_and_populate_index() -> AtlasIndexer:
    """Helper to run the full ingestion -> LlamaIndex indexing pipeline."""
    raw_data = fetch_karlsruhe_places()
    documents = build_place_documents(raw_data)
    indexer = AtlasIndexer()
    indexer.index_documents(documents)
    return indexer


if __name__ == "__main__":
    indexer = build_and_populate_index()

    print("\n--- Testing Search 1: 'best sushi restaurant' ---")
    results = indexer.search("best sushi restaurant", limit=3)
    for r in results:
        print(f"[{r['score']:.4f}] {r['name']} ({r['category']}) - {r['address']}: {r['text'][:80]}...")

    print("\n--- Testing Search 2: 'washing machine appliance store' ---")
    results = indexer.search("washing machine appliance store", limit=3)
    for r in results:
        print(f"[{r['score']:.4f}] {r['name']} ({r['category']}) - {r['address']}: {r['text'][:80]}...")
