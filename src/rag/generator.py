import json
from typing import Dict, Any, List, Optional
from llama_index.llms.ollama import Ollama
from src.rag.geo_retriever import HybridGeoRetriever


SYSTEM_PROMPT = """You are Atlas, an expert, honest, and grounded local guide for Karlsruhe, Germany.
You assist residents and visitors by recommending verified places retrieved from OpenStreetMap data.

CRITICAL INSTRUCTIONS:
1. Grounding & Honesty: Base your answers ONLY on the provided retrieved places. Never invent or hallucinate places, addresses, cuisines, or opening hours.
2. Distance & Proximity: Always mention the physical distance (in meters or km) and estimated walking time from the user's starting point.
3. Edge Cases & Missing Matches: If the context includes a caveat or if the retrieved places do not actually match the user's specific request (e.g., asking for Peruvian food when only Vietnamese or generic places exist), be transparent and state clearly that Karlsruhe's OpenStreetMap dataset lacks that exact category, before offering the closest alternative.
4. Actionable Advice: For each recommended place, provide the name, address, key highlights (outdoor seating, wheelchair access, vegan/vegetarian options if present), and opening hours if available.
5. Tone: Helpful, concise, practical, and grounded. No hyperbolic marketing speak."""


class AtlasRAG:
    def __init__(self, model_name: str = "mistral", timeout: float = 90.0):
        self.retriever = HybridGeoRetriever()
        self.model_name = model_name
        self.llm = Ollama(model=model_name, request_timeout=timeout)

    def answer_query(
        self,
        query: str,
        user_coords: Optional[tuple] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Processes a user query end-to-end:
        1. Hybrid geo-retrieval (coordinates, candidates, distances, caveats).
        2. Constructs a grounded prompt.
        3. Calls Mistral via Ollama.
        4. Returns the response with full provenance and map metadata.
        """
        retrieval = self.retriever.retrieve(query, user_coords=user_coords, top_k=top_k)
        matches = retrieval["matches"]
        origin_label = retrieval["origin_label"]
        origin_coords = retrieval["origin_coords"]
        caveat = retrieval["caveat"]

        # Build context prompt
        context_blocks = []
        for i, m in enumerate(matches, 1):
            block = (
                f"[{i}] {m['name']} (Category: {m['category']}, Subcategory: {m['subcategory']})\n"
                f"    Address: {m['address'] or 'Karlsruhe'}\n"
                f"    Distance from {origin_label}: {m['distance_meters']} meters (~{m['walking_minutes']} min walk)\n"
                f"    Opening Hours: {m['opening_hours'] or 'Not specified in OpenStreetMap'}\n"
                f"    Details: {m['text']}"
            )
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks) if context_blocks else "No places found in dataset."

        prompt = f"""[USER QUERY]
"{query}"

[USER REFERENCE LOCATION]
{origin_label} (Coordinates: {origin_coords[0]:.4f}, {origin_coords[1]:.4f})

[DATASET CAVEAT]
{caveat if caveat else "None (High-confidence matches available)"}

[RETRIEVED PLACES FROM KARLSRUHE OPENSTREETMAP DATA]
{context_str}

Please generate a helpful, well-structured response following your instructions."""

        # Generate response
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        
        # Call LLM
        response = self.llm.complete(f"{SYSTEM_PROMPT}\n\n{prompt}")
        response_text = response.text.strip()

        return {
            "query": query,
            "origin_label": origin_label,
            "origin_coords": origin_coords,
            "caveat": caveat,
            "matches": matches,
            "response": response_text,
        }


if __name__ == "__main__":
    rag = AtlasRAG()
    test_q = "I am looking for the best sushi restaurant in Europaplatz."
    res = rag.answer_query(test_q)
    print("\n--- GENERATED RESPONSE ---")
    print(res["response"])
    print("\n--- SOURCES USED ---")
    for m in res["matches"]:
        print(f"* {m['name']} ({m['distance_meters']}m away)")
