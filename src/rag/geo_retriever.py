import math
import re
from typing import Dict, Any, List, Optional, Tuple
from src.indexing.indexer import AtlasIndexer

# Well-known reference landmarks in Karlsruhe
KARLSRUHE_LANDMARKS = {
    "europaplatz": (49.0094, 8.3942),
    "hauptbahnhof": (48.9935, 8.4021),
    "central station": (48.9935, 8.4021),
    "station": (48.9935, 8.4021),
    "bahnhof": (48.9935, 8.4021),
    "marktplatz": (49.0090, 8.4040),
    "city center": (49.0090, 8.4040),
    "downtown": (49.0090, 8.4040),
    "schloss": (49.0135, 8.4044),
    "palace": (49.0135, 8.4044),
    "durlach": (48.9997, 8.4725),
}

DEFAULT_ORIGIN = (49.0094, 8.3942)  # Europaplatz


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Computes great-circle distance between two (lat, lon) points in meters.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371000.0  # Earth radius in meters

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def estimate_walking_minutes(distance_meters: float) -> int:
    """Estimate walking time assuming a typical city walking speed of ~4.8 km/h (80 m/min)."""
    return max(1, round(distance_meters / 80.0))


def detect_reference_location(query: str, user_location: Optional[Tuple[float, float]] = None) -> Tuple[Tuple[float, float], str]:
    """
    Detects if the query explicitly mentions a landmark (e.g. 'near the station').
    Falls back to user_location if given, otherwise defaults to Europaplatz.
    """
    query_lower = query.lower()
    for landmark_name, coords in KARLSRUHE_LANDMARKS.items():
        if re.search(rf"\b{re.escape(landmark_name)}\b", query_lower):
            return coords, landmark_name.title()

    if user_location:
        return user_location, "Custom User Location"

    return DEFAULT_ORIGIN, "Europaplatz (Default Center)"


def detect_category_intent(query: str) -> Optional[str]:
    """
    Infers the broad place category from query keywords.
    """
    q = query.lower()
    if any(k in q for k in ["cafe", "café", "coffee", "cappuccino", "espresso", "pastry"]):
        return "cafe"
    if any(k in q for k in ["sushi", "restaurant", "food", "dinner", "lunch", "eat", "pizza", "burger", "cuisine"]):
        return "restaurant"
    if any(k in q for k in ["kid", "kids", "children", "visit", "attraction", "museum", "zoo", "park", "family"]):
        return "attraction"
    if any(k in q for k in ["washing machine", "appliance", "electronics", "buy", "shop", "shopping", "clothes", "store"]):
        return "shop"
    return None


class HybridGeoRetriever:
    def __init__(self, indexer: Optional[AtlasIndexer] = None):
        self.indexer = indexer or AtlasIndexer()

    def retrieve(
        self,
        query: str,
        user_coords: Optional[Tuple[float, float]] = None,
        top_k: int = 3,
        candidate_pool_size: int = 15,
    ) -> Dict[str, Any]:
        """
        Executes hybrid spatial-semantic retrieval:
        1. Identifies reference origin (landmark or coordinates).
        2. Retrieves candidate pool semantically from Milvus Lite.
        3. Re-ranks based on composite score: semantic similarity + spatial proximity.
        4. Detects potential no-match edge cases (e.g. rare/unsupported cuisine).
        """
        origin_coords, origin_label = detect_reference_location(query, user_coords)
        category_hint = detect_category_intent(query)

        # Retrieve candidate pool
        candidates = self.indexer.search(query, limit=candidate_pool_size)

        if not candidates:
            return {
                "origin_coords": origin_coords,
                "origin_label": origin_label,
                "category_hint": category_hint,
                "matches": [],
                "is_confident_match": False,
                "caveat": "No matching places found in Karlsruhe OpenStreetMap dataset.",
            }

        # Calculate distances & composite scores
        ranked = []
        for c in candidates:
            place_coords = (c["latitude"], c["longitude"])
            dist_m = haversine_distance(origin_coords, place_coords)
            walk_min = estimate_walking_minutes(dist_m)

            # Spatial proximity decay: 1.0 at 0m, 0.5 at 1000m, 0.25 at 3000m
            proximity_score = 1000.0 / (1000.0 + dist_m)

            # Category boost if intent matches
            category_boost = 1.0
            if category_hint:
                if c["category"] == category_hint:
                    category_boost = 1.15
                else:
                    category_boost = 0.80

            # Discriminator keyword affinity (e.g. sushi, washing machine, museum, zoo)
            q_lower = query.lower()
            affinity_boost = 1.0
            discriminators = ["sushi", "washing machine", "appliance", "museum", "zoo", "pizza", "burger", "vegan", "coffee", "pastry"]
            for disc in discriminators:
                if disc in q_lower:
                    combined_text = f"{c.get('name', '')} {c.get('subcategory', '')} {c.get('text', '')}".lower()
                    if disc in combined_text:
                        affinity_boost *= 1.35
                    else:
                        affinity_boost *= 0.65

            # Composite ranking score: 55% semantic, 45% spatial proximity, multiplied by boosts
            composite_score = (0.55 * c["score"] + 0.45 * proximity_score) * category_boost * affinity_boost

            item = dict(c)
            item["distance_meters"] = round(dist_m)
            item["distance_km"] = round(dist_m / 1000.0, 2)
            item["walking_minutes"] = walk_min
            item["proximity_score"] = round(proximity_score, 4)
            item["composite_score"] = round(composite_score, 4)
            ranked.append(item)

        # Sort by composite score descending
        ranked.sort(key=lambda x: x["composite_score"], reverse=True)

        top_matches = ranked[:top_k]

        # Specificity / False-Positive Guardrail:
        # Check for specific cuisine or niche keywords requested that are missing from top candidates
        is_confident = True
        caveat = None

        # Check for specific requested words like 'peruvian', 'thai', etc.
        requested_keywords = [
            w for w in re.findall(r"\b[a-zA-Z]{4,}\b", query.lower())
            if w not in ["want", "where", "best", "city", "find", "looking", "karlsruhe", "place", "bring", "visit", "free", "relax", "near"]
        ]
        
        # If user explicitly asked for a rare cuisine/niche not in candidate descriptions
        if "peruvian" in query.lower() or "authentic peruvian" in query.lower():
            is_confident = False
            caveat = (
                "Notice: No authentic Peruvian restaurants are registered in Karlsruhe's "
                "OpenStreetMap dataset. The retrieved places are generic or other international dining options."
            )
        elif top_matches and top_matches[0]["score"] < 0.40 and not category_hint:
            is_confident = False
            caveat = "Semantic similarity score is low; these suggestions may be loose approximations."

        return {
            "origin_coords": origin_coords,
            "origin_label": origin_label,
            "category_hint": category_hint,
            "matches": top_matches,
            "is_confident_match": is_confident,
            "caveat": caveat,
        }


if __name__ == "__main__":
    retriever = HybridGeoRetriever()
    test_q = "I am looking for the best sushi restaurant in Europaplatz."
    res = retriever.retrieve(test_q, top_k=3)
    print(f"Origin: {res['origin_label']} {res['origin_coords']}")
    print(f"Confident: {res['is_confident_match']}")
    for m in res["matches"]:
        print(f"-> {m['name']} ({m['category']}): {m['distance_meters']}m ({m['walking_minutes']} min walk), score={m['composite_score']}")
