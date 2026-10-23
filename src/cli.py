import argparse
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure package root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.overpass_client import fetch_karlsruhe_places
from src.ingestion.document_builder import build_place_documents
from src.indexing.indexer import AtlasIndexer
from src.rag.generator import AtlasRAG
from src.visualization.map_renderer import render_route_map


def handle_ingest(args):
    print("=" * 60)
    print("Atlas: Document Ingestion & Milvus Lite Indexing")
    print("=" * 60)
    
    start_t = time.time()
    raw = fetch_karlsruhe_places(force_refresh=args.refresh)
    docs = build_place_documents(raw)
    print(f"[Ingest] Generated {len(docs)} structured place documents.")
    
    indexer = AtlasIndexer()
    count = indexer.index_documents(docs)
    elapsed = time.time() - start_t
    print(f"\n[Ingest] Completed! {count} places indexed in {elapsed:.2f} seconds.")


def handle_add(args):
    print("=" * 60)
    print("Atlas: Add Custom Document to Milvus Lite")
    print("=" * 60)

    try:
        parts = [float(x.strip()) for x in args.coords.split(",")]
        lat, lon = parts[0], parts[1]
    except Exception:
        print(f"[Error] Invalid coordinates format '{args.coords}'. Expected 'lat,lon' (e.g. 49.0094,8.3942)")
        sys.exit(1)

    indexer = AtlasIndexer()
    record = indexer.add_single_document(
        name=args.name,
        text=args.text,
        latitude=lat,
        longitude=lon,
        category=args.category,
        subcategory=args.subcategory,
        address=args.address or "",
        opening_hours=args.hours or "",
    )
    print("\n[Success] Document added successfully to Milvus Lite!")
    print(f"  ID: {record['id']}")
    print(f"  Name: {record['name']}")
    print(f"  Category: {record['category']} / {record['subcategory']}")
    print(f"  Coordinates: ({record['latitude']:.4f}, {record['longitude']:.4f})")
    print(f"  Address: {record['address'] or 'N/A'}")
    print(f"  Hours: {record['opening_hours'] or 'N/A'}")
    print(f"  Description: \"{record['text']}\"")
    print("\nYou can now query this newly added document immediately:")
    print(f"  python atlas.py ask \"{args.name}\"")
    print("=" * 60 + "\n")


def handle_ask(args):
    query = args.query
    user_coords = None
    if args.coords:
        try:
            parts = [float(x.strip()) for x in args.coords.split(",")]
            user_coords = (parts[0], parts[1])
        except Exception:
            print(f"[Error] Invalid coordinates format '{args.coords}'. Expected 'lat,lon' (e.g. 49.0094,8.3942)")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(f"QUERY: \"{query}\"")
    print("=" * 70)

    rag = AtlasRAG(model_name=args.model)
    result = rag.answer_query(query, user_coords=user_coords, top_k=args.top_k)

    # 1. Output Reference Location
    origin_label = result["origin_label"]
    origin_coords = result["origin_coords"]
    print(f"\n[Reference Location]: {origin_label} ({origin_coords[0]:.4f}, {origin_coords[1]:.4f})")

    # 2. Output Caveat if present
    if result["caveat"]:
        print(f"\n[Transparency Warning]: {result['caveat']}")

    # 3. Output Generated LLM Response
    print("\n" + "-" * 70)
    print("RECOMMENDATION:")
    print("-" * 70)
    print(result["response"])

    # 4. Output Provenance / Sources Used
    print("\n" + "-" * 70)
    print("SOURCES USED (FROM OPENSTREETMAP DATA):")
    print("-" * 70)
    for i, m in enumerate(result["matches"], 1):
        print(f"[{i}] {m['name']} ({m['category']}/{m['subcategory']})")
        print(f"    Address: {m['address'] or 'Karlsruhe'}")
        print(f"    Distance: {m['distance_meters']}m (~{m['walking_minutes']} min walk from {origin_label})")
        print(f"    Opening Hours: {m['opening_hours'] or 'Not listed in OSM'}")
        print(f"    Composite Match Score: {m['composite_score']:.4f} (Semantic: {m['score']:.4f})")

    # 5. Render Interactive Map
    if not args.no_map:
        clean_name = "".join(c if c.isalnum() else "_" for c in query[:30]).strip("_").lower()
        map_filename = f"map_{clean_name}_{int(time.time())}.html"
        map_path = os.path.join("outputs", "maps", map_filename)
        render_route_map(
            origin_coords=origin_coords,
            origin_label=origin_label,
            places=result["matches"],
            output_path=map_path,
            query_title=query,
        )
        print(f"\n[Interactive Map]: Saved to {map_path}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Atlas - Grounded Karlsruhe Geo-RAG Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest OpenStreetMap data and populate Milvus Lite")
    ingest_parser.add_argument("--refresh", action="store_true", help="Force refresh from Overpass API instead of cache")

    # Add subcommand (add single custom document)
    add_parser = subparsers.add_parser("add", help="Add a custom place document to Milvus Lite")
    add_parser.add_argument("--name", type=str, required=True, help="Name of the venue or document")
    add_parser.add_argument("--coords", type=str, required=True, help="Coordinates as 'lat,lon' (e.g. 49.0094,8.3942)")
    add_parser.add_argument("--text", type=str, required=True, help="Text description of the venue / document")
    add_parser.add_argument("--category", type=str, default="custom", help="Category (e.g. sustenance, tourism, shopping)")
    add_parser.add_argument("--subcategory", type=str, default="place", help="Subcategory (e.g. cafe, restaurant, museum)")
    add_parser.add_argument("--address", type=str, default="", help="Street address")
    add_parser.add_argument("--hours", type=str, default="", help="Opening hours (e.g. Mo-Fr 08:00-18:00)")

    # Ask subcommand
    ask_parser = subparsers.add_parser("ask", help="Query the Geo-RAG assistant")
    ask_parser.add_argument("query", type=str, help="Natural language question or request")
    ask_parser.add_argument("--coords", type=str, default=None, help="User coordinates as 'lat,lon' (e.g. 49.0094,8.3942)")
    ask_parser.add_argument("--top-k", type=int, default=3, help="Number of places to recommend (default: 3)")
    ask_parser.add_argument("--model", type=str, default="mistral", help="Ollama LLM model name (default: mistral)")
    ask_parser.add_argument("--no-map", action="store_true", help="Disable Folium map generation")

    args = parser.parse_args()
    if args.command == "ingest":
        handle_ingest(args)
    elif args.command == "add":
        handle_add(args)
    elif args.command == "ask":
        handle_ask(args)


if __name__ == "__main__":
    main()
