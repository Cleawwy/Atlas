import os
import sys
import json
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.generator import AtlasRAG
from src.visualization.map_renderer import render_route_map

EXAMPLE_QUERIES = [
    {
        "id": "query_1_restaurant_sushi",
        "category": "Restaurant (Sushi)",
        "query": "I am looking for the best sushi restaurant in Europaplatz.",
        "description": "Tests proximity weighting and cuisine matching centered at Europaplatz.",
    },
    {
        "id": "query_2_situational_kids",
        "category": "Situational (Kids & Family)",
        "query": "Where can I bring my 2 kids to visit in Karlsruhe?",
        "description": "Tests intent extraction for family attractions, museums, and parks.",
    },
    {
        "id": "query_3_shopping_washing_machine",
        "category": "Shopping (Appliances)",
        "query": "I want to buy a new washing machine, where can I go in the city?",
        "description": "Tests product-intent matching in retail and electronics.",
    },
    {
        "id": "query_4_edge_case_peruvian",
        "category": "Edge Case (Unsupported Cuisine)",
        "query": "Where can I find an authentic Peruvian restaurant in Karlsruhe?",
        "description": "Deliberately sparse/absent cuisine in Karlsruhe OSM to test hallucination resistance.",
    },
    {
        "id": "query_5_proximity_station_cafe",
        "category": "Proximity & Time Budget (Station Cafe)",
        "query": "I have an hour free near the station, what's a nice café to relax in?",
        "description": "Tests spatial reasoning around Karlsruhe Hauptbahnhof under a time budget.",
    },
]


def run_benchmark():
    rag = AtlasRAG(model_name="mistral")
    results = []

    os.makedirs("outputs/maps", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    print("================================================================")
    print("Running Atlas 5-Query Benchmark Suite...")
    print("================================================================")

    for item in EXAMPLE_QUERIES:
        qid = item["id"]
        qtext = item["query"]
        print(f"\n---> Executing [{item['category']}]: '{qtext}'")

        start_time = time.time()
        res = rag.answer_query(qtext, top_k=3)
        duration = time.time() - start_time

        # Render map
        map_path = f"outputs/maps/{qid}.html"
        render_route_map(
            origin_coords=res["origin_coords"],
            origin_label=res["origin_label"],
            places=res["matches"],
            output_path=map_path,
            query_title=qtext,
        )

        record = {
            "id": qid,
            "category": item["category"],
            "query": qtext,
            "description": item["description"],
            "origin_label": res["origin_label"],
            "origin_coords": res["origin_coords"],
            "caveat": res["caveat"],
            "response": res["response"],
            "matches": res["matches"],
            "map_path": map_path,
            "duration_seconds": round(duration, 2),
        }
        results.append(record)

        print(f"Generated response in {duration:.1f}s.")
        print(f"Map saved to: {map_path}")

    # Save aggregated JSON
    output_json = "outputs/results/benchmark_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nAll 5 queries executed successfully! Results saved to {output_json}")


if __name__ == "__main__":
    run_benchmark()
