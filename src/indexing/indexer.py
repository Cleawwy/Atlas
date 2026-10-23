import os
import sys
import json
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.ingestion.overpass_client import fetch_karlsruhe_places
from src.ingestion.document_builder import build_place_documents, PlaceDocument

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "atlas_milvus.db",
)
COLLECTION_NAME = "karlsruhe_places"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class AtlasIndexer:
    def __init__(self, db_path: str = DEFAULT_DB_PATH, model_name: str = EMBEDDING_MODEL_NAME):
        self.db_path = db_path
        self.model_name = model_name
        print(f"[Indexer] Loading embedding model '{self.model_name}'...")
        self.encoder = SentenceTransformer(self.model_name)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.client = MilvusClient(uri=self.db_path)

    def init_collection(self, drop_existing: bool = False):
        """Creates or resets the Milvus Lite collection."""
        if drop_existing and self.client.has_collection(COLLECTION_NAME):
            print(f"[Indexer] Dropping existing collection '{COLLECTION_NAME}'...")
            self.client.drop_collection(COLLECTION_NAME)

        if not self.client.has_collection(COLLECTION_NAME):
            print(f"[Indexer] Creating collection '{COLLECTION_NAME}' (dim={EMBEDDING_DIM}, metric=COSINE)...")
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                dimension=EMBEDDING_DIM,
                metric_type="COSINE",
                primary_field_name="id",
                id_type="string",
                max_length=64,
            )
            print("[Indexer] Collection created successfully.")
        else:
            print(f"[Indexer] Collection '{COLLECTION_NAME}' already exists.")

    def index_documents(self, documents: List[PlaceDocument], batch_size: int = 64) -> int:
        """
        Embeds place documents in batches and inserts them into Milvus Lite
        along with their structured metadata.
        """
        self.init_collection(drop_existing=True)
        total_docs = len(documents)
        print(f"[Indexer] Generating embeddings for {total_docs} documents in batches of {batch_size}...")

        texts = [doc.text for doc in documents]
        embeddings = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        entries = []
        for doc, emb in zip(documents, embeddings):
            address_str = f"{doc.metadata.get('street', '')} {doc.metadata.get('housenumber', '')}".strip()
            entry = {
                "id": doc.id,
                "vector": emb.tolist(),
                "name": doc.name,
                "category": doc.category,
                "subcategory": doc.subcategory,
                "latitude": float(doc.latitude),
                "longitude": float(doc.longitude),
                "address": address_str,
                "opening_hours": str(doc.metadata.get("opening_hours", "")),
                "text": doc.text,
            }
            entries.append(entry)

        print(f"[Indexer] Inserting {len(entries)} records into Milvus Lite...")
        # Milvus Lite supports bulk insertion
        self.client.insert(collection_name=COLLECTION_NAME, data=entries)
        print(f"[Indexer] Successfully indexed {len(entries)} places into {self.db_path}.")
        return len(entries)

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
        Embeds a single custom place document and appends it to Milvus Lite.
        Does not drop or reset existing records.
        """
        self.init_collection(drop_existing=False)
        if not doc_id:
            import uuid
            doc_id = f"custom_{uuid.uuid4().hex[:8]}"

        emb = self.encoder.encode([text], normalize_embeddings=True)[0]
        entry = {
            "id": doc_id,
            "vector": emb.tolist(),
            "name": name,
            "category": category,
            "subcategory": subcategory,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "address": address,
            "opening_hours": opening_hours,
            "text": text,
        }
        self.client.insert(collection_name=COLLECTION_NAME, data=[entry])
        print(f"[Indexer] Document '{name}' (ID: {doc_id}) added to Milvus Lite.")
        return entry

    def search(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search, with optional scalar category filter.
        """
        try:
            self.client.load_collection(COLLECTION_NAME)
        except Exception:
            pass

        query_vector = self.encoder.encode(query, normalize_embeddings=True).tolist()
        
        filter_expr = f'category == "{category}"' if category else ""
        
        search_params = {
            "collection_name": COLLECTION_NAME,
            "data": [query_vector],
            "limit": limit,
            "output_fields": [
                "name",
                "category",
                "subcategory",
                "latitude",
                "longitude",
                "address",
                "opening_hours",
                "text",
            ],
        }
        if filter_expr:
            search_params["filter"] = filter_expr

        results = self.client.search(**search_params)
        
        hits = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get("entity", {})
                hits.append({
                    "id": hit.get("id"),
                    "score": float(hit.get("distance", 0.0)),
                    "name": entity.get("name"),
                    "category": entity.get("category"),
                    "subcategory": entity.get("subcategory"),
                    "latitude": entity.get("latitude"),
                    "longitude": entity.get("longitude"),
                    "address": entity.get("address"),
                    "opening_hours": entity.get("opening_hours"),
                    "text": entity.get("text"),
                })
        return hits


def build_and_populate_index() -> AtlasIndexer:
    """Helper to run the full ingestion -> indexing pipeline."""
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
