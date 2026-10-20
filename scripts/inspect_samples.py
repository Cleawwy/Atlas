import os
import sys

sys.path.insert(0, os.path.abspath("."))
from src.ingestion.overpass_client import fetch_karlsruhe_places
from src.ingestion.document_builder import build_place_documents

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

raw = fetch_karlsruhe_places()
docs = build_place_documents(raw)

samples = []

# 1. Sushi
for d in docs:
    if "sushi" in d.text.lower():
        samples.append(("Restaurant (Sushi)", d))
        break

# 2. Attraction / Kid-friendly
for d in docs:
    if any(k in d.name.lower() for k in ["naturkunde", "zoo", "stadtgarten"]):
        samples.append(("Attraction (Kid/Family-Friendly)", d))
        break

# 3. Shopping / Appliance
for d in docs:
    if "saturn" in d.name.lower() or "appliance" in d.text.lower():
        samples.append(("Shopping (Appliances / Tech)", d))
        break

# 4. Cafe
for d in docs:
    if d.category == "cafe" and "opening_hours" in d.metadata.get("raw_tags", {}):
        samples.append(("Cafe (Coffee & Pastries)", d))
        break

# 5. Sparse place (minimal tags, graceful fallback)
for d in docs:
    raw_tags = d.metadata.get("raw_tags", {})
    if len(raw_tags) <= 3 and "opening_hours" not in raw_tags and "addr:street" not in raw_tags:
        samples.append(("Sparse Place (Graceful Fallback)", d))
        break

for title, d in samples:
    print(f"### {title}")
    print(f"- **Name:** {d.name}")
    print(f"- **ID:** `{d.id}`")
    print(f"- **Category:** `{d.category}` | **Subcategory:** `{d.subcategory}`")
    print(f"- **Coordinates:** `({d.latitude:.5f}, {d.longitude:.5f})`")
    print(f"- **Address in Metadata:** `{d.metadata.get('street', '')} {d.metadata.get('housenumber', '')}`.strip()")
    print(f"- **Generated Text for Embedding:**\n  \"{d.text}\"\n")
