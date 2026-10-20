import json
import os
import time
from typing import Dict, List, Any, Optional
import requests

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Bounding box for Karlsruhe metropolitan area (South, West, North, East)
# Covers City Center, Europaplatz, Marktplatz, Durlach, Oststadt, Weststadt, Hauptbahnhof
DEFAULT_BBOX = (48.970, 8.320, 49.070, 8.480)

DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "raw",
    "karlsruhe_places.json",
)


def build_overpass_query(bbox: tuple = DEFAULT_BBOX) -> str:
    """
    Construct an Overpass QL query covering:
    - Restaurants & Cafes (amenity=restaurant, amenity=cafe)
    - Tourist attractions (tourism=attraction, tourism=museum, tourism=viewpoint)
    - Shopping centers, electronics, and appliance stores (shop=*)
    Restricted to entities that possess a 'name' tag.
    """
    s, w, n, e = bbox
    bbox_str = f"{s},{w},{n},{e}"
    
    query = f"""
    [out:json][timeout:60];
    (
      // Restaurants & Cafes
      node["amenity"="restaurant"]["name"]({bbox_str});
      way["amenity"="restaurant"]["name"]({bbox_str});
      node["amenity"="cafe"]["name"]({bbox_str});
      way["amenity"="cafe"]["name"]({bbox_str});

      // Attractions, Museums & Family Entertainment
      node["tourism"~"attraction|museum|viewpoint|gallery|theme_park|zoo|aquarium"]["name"]({bbox_str});
      way["tourism"~"attraction|museum|viewpoint|gallery|theme_park|zoo|aquarium"]["name"]({bbox_str});
      node["leisure"~"water_park|park"]["name"]({bbox_str});
      way["leisure"~"water_park|park"]["name"]({bbox_str});
      node["amenity"="cinema"]["name"]({bbox_str});

      // Shopping: department stores, malls, electronics, appliances, general retail
      node["shop"~"department_store|mall|electronics|appliances|supermarket|clothes|furniture"]["name"]({bbox_str});
      way["shop"~"department_store|mall|electronics|appliances|supermarket|clothes|furniture"]["name"]({bbox_str});
    );
    out center tags;
    """
    return query


def fetch_karlsruhe_places(
    force_refresh: bool = False,
    cache_path: str = DEFAULT_CACHE_PATH,
    timeout: int = 60,
) -> List[Dict[str, Any]]:
    """
    Fetches Karlsruhe places from Overpass API or loads from local cache.
    Caches the parsed result to ensure offline repeatability and avoid rate limits.
    """
    if not force_refresh and os.path.exists(cache_path):
        print(f"[Ingestion] Loading cached OSM data from: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"[Ingestion] Loaded {len(data)} places from cache.")
            return data

    query = build_overpass_query()
    print("[Ingestion] Fetching fresh OSM data from Overpass API for Karlsruhe...")

    last_error = None
    for endpoint in OVERPASS_URLS:
        try:
            print(f"[Ingestion] Querying endpoint: {endpoint}")
            response = requests.post(
                endpoint,
                data={"data": query},
                headers={"User-Agent": "Atlas-GeoRAG-Assignment/1.0"},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            elements = payload.get("elements", [])
            print(f"[Ingestion] Successfully retrieved {len(elements)} raw elements.")
            
            # Save raw snapshot
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(elements, f, ensure_ascii=False, indent=2)
            print(f"[Ingestion] Cached snapshot saved to {cache_path}")
            return elements

        except Exception as e:
            print(f"[Ingestion] Warning: Endpoint {endpoint} failed: {e}. Trying next...")
            last_error = e
            time.sleep(2)

    raise RuntimeError(f"All Overpass API endpoints failed. Last error: {last_error}")


if __name__ == "__main__":
    elements = fetch_karlsruhe_places(force_refresh=True)
    print(f"Sample raw element count: {len(elements)}")
    if elements:
        print("Sample element:", json.dumps(elements[0], indent=2, ensure_ascii=False))
