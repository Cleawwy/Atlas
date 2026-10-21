import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class PlaceDocument:
    id: str
    name: str
    category: str  # 'restaurant', 'cafe', 'attraction', 'shop'
    subcategory: str  # e.g., 'electronics', 'greek', 'museum'
    latitude: float
    longitude: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def clean_cuisine_string(cuisine_tag: str) -> str:
    """Normalize cuisine tags like 'greek;mediterranean' to 'Greek and Mediterranean'."""
    items = [c.strip().capitalize() for c in cuisine_tag.replace(";", ",").split(",") if c.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def indefinite_article(word: str) -> str:
    """Return 'an' if the word begins with a vowel sound, otherwise 'a'."""
    word = word.strip().lower()
    if not word:
        return "a"
    return "an" if word[0] in "aeiou" else "a"


def determine_category(tags: Dict[str, str]) -> Tuple[str, str]:
    """
    Identifies the primary category and subcategory from OSM tags.
    Categories: 'restaurant', 'cafe', 'attraction', 'shop'
    """
    amenity = tags.get("amenity", "").lower()
    tourism = tags.get("tourism", "").lower()
    leisure = tags.get("leisure", "").lower()
    shop = tags.get("shop", "").lower()

    if amenity == "restaurant":
        cuisine = tags.get("cuisine", "general dining")
        return "restaurant", cuisine
    elif amenity == "cafe":
        return "cafe", tags.get("cuisine", "coffee and pastries")
    elif tourism in ["attraction", "museum", "viewpoint", "gallery", "theme_park", "zoo", "aquarium"]:
        return "attraction", tourism
    elif leisure in ["park", "water_park", "playground"]:
        return "attraction", leisure
    elif amenity == "cinema":
        return "attraction", "cinema"
    elif shop:
        return "shop", shop
    elif amenity in ["bar", "pub", "fast_food", "ice_cream"]:
        return "restaurant", amenity
    
    return "attraction", "point of interest"


def extract_coordinates(element: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Extract lat/lon from either a node or a way center."""
    if "lat" in element and "lon" in element:
        return float(element["lat"]), float(element["lon"])
    if "center" in element and "lat" in element["center"] and "lon" in element["center"]:
        return float(element["center"]["lat"]), float(element["center"]["lon"])
    return None


def build_natural_language_description(name: str, category: str, subcategory: str, tags: Dict[str, str]) -> str:
    """
    Generates a coherent, natural-language description for a place.
    Gracefully handles missing or sparse metadata by omitting clauses rather
    than interpolating placeholders like 'None' or empty brackets.
    """
    sentences = []

    # 1. Primary identification sentence
    if category == "restaurant":
        cuisine_raw = tags.get("cuisine")
        if cuisine_raw:
            cuisine_clean = clean_cuisine_string(cuisine_raw)
            art = indefinite_article(cuisine_clean)
            sentences.append(f"{name} is {art} {cuisine_clean} restaurant in Karlsruhe.")
        else:
            sentences.append(f"{name} is a restaurant in Karlsruhe.")
    elif category == "cafe":
        sentences.append(f"{name} is a café and coffee spot in Karlsruhe.")
    elif category == "attraction":
        attr_type = subcategory.replace("_", " ")
        art = indefinite_article(attr_type)
        if attr_type == "museum":
            sentences.append(f"{name} is a cultural museum and exhibition space in Karlsruhe.")
        elif attr_type == "viewpoint":
            sentences.append(f"{name} is a scenic viewpoint in Karlsruhe.")
        elif attr_type in ["zoo", "aquarium"]:
            sentences.append(f"{name} is a zoological garden and animal park in Karlsruhe, ideal for families and children.")
        elif attr_type in ["theme park", "water park"]:
            sentences.append(f"{name} is a recreational attraction in Karlsruhe, popular for family outings and kids.")
        elif attr_type == "cinema":
            sentences.append(f"{name} is a movie theater and cinema in Karlsruhe.")
        elif attr_type in ["park", "garden"]:
            sentences.append(f"{name} is a public park and green space in Karlsruhe, great for walking and family leisure.")
        else:
            sentences.append(f"{name} is {art} {attr_type} and visitor attraction in Karlsruhe.")
    elif category == "shop":
        shop_type = subcategory.replace("_", " ")
        art = indefinite_article(shop_type)
        if shop_type in ["department_store", "mall"]:
            sentences.append(f"{name} is a large shopping center and department store in Karlsruhe.")
        elif shop_type in ["electronics", "appliances"]:
            sentences.append(f"{name} is an electronics and home appliance retailer in Karlsruhe offering goods such as washing machines, computers, and home technology.")
        elif shop_type == "supermarket":
            sentences.append(f"{name} is a grocery supermarket in Karlsruhe.")
        elif shop_type == "clothes":
            sentences.append(f"{name} is a clothing and fashion store in Karlsruhe.")
        else:
            sentences.append(f"{name} is {art} {shop_type} shop in Karlsruhe.")

    # 2. Location & Address sentence
    street = tags.get("addr:street")
    housenumber = tags.get("addr:housenumber")
    suburb = tags.get("addr:suburb")
    postcode = tags.get("addr:postcode")

    address_parts = []
    if street:
        if housenumber:
            address_parts.append(f"{street} {housenumber}")
        else:
            address_parts.append(street)
    if suburb:
        address_parts.append(f"in the {suburb} neighborhood")
    if postcode:
        address_parts.append(f"({postcode} Karlsruhe)")

    if address_parts:
        sentences.append(f"It is located at {' '.join(address_parts)}.")

    # 3. Specific highlights (dietary, offerings, features)
    feature_clauses = []
    
    # Dietary options for food/drink
    if tags.get("diet:vegan") in ["yes", "only"]:
        feature_clauses.append("serves dedicated vegan dishes")
    elif tags.get("diet:vegetarian") in ["yes", "only"]:
        feature_clauses.append("offers vegetarian options")

    # Seating & service
    if tags.get("outdoor_seating") == "yes":
        feature_clauses.append("provides outdoor seating")
    if tags.get("takeaway") == "yes":
        feature_clauses.append("supports takeaway orders")
    if tags.get("delivery") == "yes":
        feature_clauses.append("offers delivery service")

    # Accessibility
    if tags.get("wheelchair") == "yes":
        feature_clauses.append("is fully wheelchair accessible")

    if feature_clauses:
        if len(feature_clauses) == 1:
            sentences.append(f"The venue {feature_clauses[0]}.")
        elif len(feature_clauses) == 2:
            sentences.append(f"The venue {feature_clauses[0]} and {feature_clauses[1]}.")
        else:
            first_part = ", ".join(feature_clauses[:-1])
            sentences.append(f"The venue {first_part}, and {feature_clauses[-1]}.")

    # 4. Opening hours (strictly formatted only if present)
    hours = tags.get("opening_hours")
    if hours:
        # Simplify common OSM hour syntax slightly for TTS/LLM clarity
        clean_hours = hours.replace(";", ", ")
        sentences.append(f"Regular opening hours: {clean_hours}.")

    # 5. Description / notes from OSM if editor added them
    description = tags.get("description") or tags.get("note")
    if description and len(description) > 10:
        sentences.append(f"Additional details: {description.strip()}")

    return " ".join(sentences)


def build_place_documents(raw_elements: List[Dict[str, Any]]) -> List[PlaceDocument]:
    """
    Transforms raw Overpass OSM elements into structured PlaceDocument instances
    suitable for vector embeddings and hybrid geospatial queries.
    """
    documents = []

    for el in raw_elements:
        tags = el.get("tags", {})
        name = tags.get("name")
        if not name or not name.strip():
            continue  # Exclude unnamed nodes/ways as they cannot be recommended

        coords = extract_coordinates(el)
        if not coords:
            continue  # Must have valid coordinates for Geo-RAG

        lat, lon = coords
        el_type = el.get("type", "node")
        el_id = el.get("id")
        place_id = f"osm_{el_type}_{el_id}"

        category, subcategory = determine_category(tags)
        nl_text = build_natural_language_description(name, category, subcategory, tags)

        doc = PlaceDocument(
            id=place_id,
            name=name.strip(),
            category=category,
            subcategory=subcategory,
            latitude=lat,
            longitude=lon,
            text=nl_text,
            metadata={
                "place_id": place_id,
                "name": name.strip(),
                "category": category,
                "subcategory": subcategory,
                "latitude": lat,
                "longitude": lon,
                "street": tags.get("addr:street", ""),
                "housenumber": tags.get("addr:housenumber", ""),
                "postcode": tags.get("addr:postcode", ""),
                "cuisine": tags.get("cuisine", ""),
                "opening_hours": tags.get("opening_hours", ""),
                "website": tags.get("website", "") or tags.get("contact:website", ""),
                "phone": tags.get("phone", "") or tags.get("contact:phone", ""),
                "raw_tags": tags,
            },
        )
        documents.append(doc)

    return documents


if __name__ == "__main__":
    from overpass_client import fetch_karlsruhe_places
    raw = fetch_karlsruhe_places()
    docs = build_place_documents(raw)
    print(f"Total documents generated: {len(docs)}")
    for d in docs[:5]:
        print("\n" + "="*50)
        print(f"ID: {d.id} | Category: {d.category} ({d.subcategory})")
        print(f"Coordinates: ({d.latitude:.5f}, {d.longitude:.5f})")
        print(f"Description:\n{d.text}")
