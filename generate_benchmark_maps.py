import os
import subprocess
from src.rag.geo_retriever import HybridGeoRetriever
from src.visualization.map_renderer import render_route_map

BENCHMARKS = [
    {
        "id": "query_1_restaurant_sushi",
        "query": "I am looking for the best sushi restaurant in Europaplatz.",
        "title": "Best Sushi near Europaplatz",
    },
    {
        "id": "query_2_situational_kids",
        "query": "I have a free afternoon with kids, where can we go?",
        "title": "Afternoon with Kids (Europaplatz)",
    },
    {
        "id": "query_3_shopping_washing_machine",
        "query": "I want to buy a new washing machine, where can I go in the city?",
        "title": "Washing Machines & Home Appliances",
    },
    {
        "id": "query_4_edge_case_peruvian",
        "query": "Where can I find an authentic Peruvian restaurant in Karlsruhe?",
        "title": "Peruvian Cuisine (International Alternatives)",
    },
    {
        "id": "query_5_proximity_station_cafe",
        "query": "I have an hour free near the station, what's a nice café to relax in?",
        "title": "1-Hour Time Budget near Station",
    },
]

def main():
    print("[1/2] Initializing retriever...")
    retriever = HybridGeoRetriever()
    
    edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    
    for bm in BENCHMARKS:
        print(f"\n--- Processing: {bm['id']} ---")
        retrieval = retriever.retrieve(bm["query"], top_k=3)
        origin_coords = retrieval["origin_coords"]
        origin_label = retrieval["origin_label"]
        matches = retrieval["matches"]
        
        html_path = os.path.join("outputs", "maps", f"{bm['id']}.html")
        png_path = os.path.abspath(os.path.join("outputs", "maps", f"{bm['id']}.png"))
        
        render_route_map(
            origin_coords=origin_coords,
            origin_label=origin_label,
            places=matches,
            output_path=html_path,
            query_title=bm["title"]
        )
        print(f"Rendered HTML -> {html_path}")
        
        # Capture screenshot with headless Edge
        abs_html_url = "file:///" + os.path.abspath(html_path).replace("\\", "/")
        cmd = [
            edge_exe,
            "--headless=new",
            "--virtual-time-budget=4000",
            f"--screenshot={png_path}",
            "--window-size=1080,650",
            abs_html_url
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(f"Captured PNG -> {png_path} (return code: {res.returncode})")

if __name__ == "__main__":
    main()
