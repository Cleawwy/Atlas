import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def test_ollama():
    print("\n--- [1/3] Testing Ollama (Mistral) ---")
    try:
        from llama_index.llms.ollama import Ollama
        llm = Ollama(model="mistral", request_timeout=60.0)
        response = llm.complete("In one short sentence, what is Karlsruhe famous for?")
        print(f"Ollama Response: {response.text.strip()}")
        print("[PASS] Ollama (Mistral) smoke test passed.")
        return True
    except Exception as e:
        print(f"[FAIL] Ollama smoke test failed: {e}")
        return False

def test_embedding():
    print("\n--- [2/3] Testing Sentence-Transformers Embedding ---")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        text = "Karlsruhe Schloss und Schlosspark"
        vector = model.encode(text)
        print(f"Embedded sample text '{text}' into vector shape: {vector.shape}")
        assert len(vector) == 384, f"Expected 384 dimensions, got {len(vector)}"
        print("[PASS] Sentence-Transformers smoke test passed.")
        return True
    except Exception as e:
        print(f"[FAIL] Embedding smoke test failed: {e}")
        return False

def test_milvus_lite():
    print("\n--- [3/3] Testing Milvus Lite ---")
    test_db = "test_smoke_milvus.db"
    if os.path.exists(test_db):
        try:
            os.remove(test_db)
        except Exception:
            pass

    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri=test_db)
        collection_name = "smoke_test"

        # Drop collection if already exists
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)

        # Create collection
        client.create_collection(
            collection_name=collection_name,
            dimension=384,
            metric_type="COSINE"
        )

        # Insert dummy vectors
        import numpy as np
        vec1 = np.random.rand(384).astype(np.float32).tolist()
        vec2 = np.random.rand(384).astype(np.float32).tolist()

        data = [
            {"id": 1, "vector": vec1, "name": "Europaplatz Sushi", "amenity": "restaurant"},
            {"id": 2, "vector": vec2, "name": "Schloss Cafe", "amenity": "cafe"},
        ]
        client.insert(collection_name=collection_name, data=data)

        # Query
        res = client.search(
            collection_name=collection_name,
            data=[vec1],
            limit=1,
            output_fields=["name", "amenity"]
        )
        print(f"Milvus search result: {res}")
        assert len(res) > 0 and len(res[0]) > 0
        top_match = res[0][0]
        print(f"Top match entity: {top_match['entity']}")
        assert top_match["entity"]["name"] == "Europaplatz Sushi"
        
        # Cleanup
        client.close()
        try:
            if os.path.exists(test_db):
                os.remove(test_db)
        except Exception:
            pass  # Windows file lock release can be deferred until process exit
        print("[PASS] Milvus Lite smoke test passed.")
        return True
    except Exception as e:
        print(f"[FAIL] Milvus Lite smoke test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running Atlas Pre-flight Smoke Tests...")
    ok_ollama = test_ollama()
    ok_embed = test_embedding()
    ok_milvus = test_milvus_lite()

    print("\n================ SUMMARY ================")
    print(f"Ollama Mistral:       {'PASS' if ok_ollama else 'FAIL'}")
    print(f"SentenceTransformer:  {'PASS' if ok_embed else 'FAIL'}")
    print(f"Milvus Lite:          {'PASS' if ok_milvus else 'FAIL'}")
    print("=========================================")

    if ok_ollama and ok_embed and ok_milvus:
        print("\nAll 3 components verified healthy! Ready to proceed.")
        sys.exit(0)
    else:
        print("\nOne or more checks failed.")
        sys.exit(1)
