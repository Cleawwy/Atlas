import os
import sys
sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.indexing.indexer import AtlasIndexer

idx = AtlasIndexer()

queries = [
    ("Sushi near Europaplatz", "I am looking for the best sushi restaurant in Europaplatz."),
    ("Kids / Family", "Where can I bring my 2 kids to visit in Karlsruhe?"),
    ("Washing machine / Shopping", "I want to buy a new washing machine, where can I go in the city?"),
    ("Sparse cuisine", "Where can I find an authentic Peruvian restaurant in Karlsruhe?"),
    ("Station cafe", "I have an hour free near the central station, what's a nice café to relax in?"),
]

for title, q in queries:
    print(f"\n==========================================")
    print(f"QUERY: '{q}' [{title}]")
    print(f"==========================================")
    hits = idx.search(q, limit=3)
    for i, h in enumerate(hits, 1):
        print(f"{i}. [{h['score']:.4f}] {h['name']} ({h['category']}/{h['subcategory']})")
        print(f"   Coords: ({h['latitude']:.4f}, {h['longitude']:.4f}) | Addr: {h['address']}")
        print(f"   Text: {h['text']}")
