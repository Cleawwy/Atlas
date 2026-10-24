import os
from typing import List, Dict, Any, Tuple, Optional
import folium


def create_minimal_pin(color: str, text: str) -> str:
    """
    Generates a crisp, minimalist inline SVG teardrop pin with a bold white label.
    The tip of the needle is at coordinate (14, 36) so route lines anchor precisely.
    """
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 36" width="28" height="36" style="filter: drop-shadow(0 1px 3px rgba(0,0,0,0.22));">
        <path d="M14 0C6.27 0 0 6.27 0 14c0 9.5 14 22 14 22s14-12.5 14-22C28 6.27 21.73 0 14 0z" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>
        <circle cx="14" cy="13" r="8" fill="#ffffff"/>
        <text x="14" y="17" font-size="10" font-weight="700" fill="{color}" text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">{text}</text>
    </svg>
    """


def render_route_map(
    origin_coords: Tuple[float, float],
    origin_label: str,
    places: List[Dict[str, Any]],
    output_path: str,
    query_title: Optional[str] = None,
) -> str:
    """
    Renders an interactive route map with a minimalist architectural split layout:
    - Left: Full-bleed Leaflet map with zero unrendered margins
    - max_native_zoom=16 prevents Esri 'Map data not yet available' tiles
    - Right: Ultra-clean, minimalist editorial typography (no chunky cards or loud colors)
    - All markers guaranteed framed within viewport
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Initialize map with max_native_zoom=16 so zooming never requests missing tile levels
    m = folium.Map(
        location=origin_coords,
        zoom_start=15,
        max_zoom=18,
        tiles=None,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri, HERE, Garmin",
        max_native_zoom=16,
        max_zoom=18,
    ).add_to(m)

    # Start Pin (Crimson)
    folium.Marker(
        location=origin_coords,
        icon=folium.DivIcon(
            icon_size=(28, 36),
            icon_anchor=(14, 36),
            html=create_minimal_pin("#dc2626", "S")
        ),
        tooltip=f"Start: {origin_label}"
    ).add_to(m)

    # Muted, sophisticated palette for destinations
    palette = ["#16a34a", "#2563eb", "#7c3aed", "#d97706", "#475569"]

    for i, place in enumerate(places, 1):
        place_coords = (place["latitude"], place["longitude"])
        color = palette[(i - 1) % len(palette)]
        dist_m = place.get("distance_meters", 0)
        walk_min = place.get("walking_minutes", 1)
        dist_str = f"{dist_m} m (~{walk_min} min walk)"

        # Venue Pin
        folium.Marker(
            location=place_coords,
            icon=folium.DivIcon(
                icon_size=(28, 36),
                icon_anchor=(14, 36),
                html=create_minimal_pin(color, str(i))
            ),
            tooltip=f"#{i} {place['name']} ({dist_str})"
        ).add_to(m)

        # Route Line
        folium.PolyLine(
            locations=[origin_coords, place_coords],
            color=color,
            weight=2.5,
            opacity=0.8,
            dash_array="5, 7",
            tooltip=f"Walk to #{i} {place['name']} ({dist_str})"
        ).add_to(m)

    # Frame ALL coordinates, capping initial auto-zoom to 16 so it never over-zooms on close venues
    all_coords = [origin_coords] + [(p["latitude"], p["longitude"]) for p in places]
    m.fit_bounds(all_coords, padding=(48, 48), max_zoom=16)

    # Save initial html
    m.save(output_path)

    # Read and inject minimalist sidebar + resize script
    with open(output_path, "r", encoding="utf-8") as f:
        raw_html = f.read()

    # Generate minimalist rows
    items_html = ""
    for i, place in enumerate(places, 1):
        color = palette[(i - 1) % len(palette)]
        dist_m = place.get("distance_meters", 0)
        walk_min = place.get("walking_minutes", 1)
        subcat = place.get("subcategory") or place.get("category") or ""
        addr = place.get("address") or "Karlsruhe"
        sub_info = f"{subcat} &bull; {addr}" if subcat else addr

        items_html += f"""
        <div style="padding: 11px 0; border-bottom: 1px solid #f1f5f9;">
            <div style="display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 2px;">
                <div style="display: flex; align-items: baseline; gap: 7px; min-width: 0;">
                    <span style="font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; font-weight: 700; color: {color}; flex-shrink: 0;">0{i}</span>
                    <span style="font-size: 13px; font-weight: 600; color: #0f172a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 190px;">{place['name']}</span>
                </div>
            </div>
            <div style="display: flex; align-items: center; justify-content: space-between; font-size: 11px; color: #64748b; padding-left: 20px;">
                <span style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 150px;">{sub_info}</span>
                <span style="font-weight: 600; color: #334155; white-space: nowrap;">{dist_m} m &bull; {walk_min} min</span>
            </div>
        </div>
        """

    display_title = query_title or "Karlsruhe Route Guide"

    sidebar_css = """
    <style>
        * { box-sizing: border-box; }
        html, body {
            margin: 0 !important;
            padding: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            overflow: hidden !important;
            display: flex !important;
            flex-direction: row !important;
            background: #f8fafc !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }
        .folium-map {
            flex: 1 1 0% !important;
            height: 100vh !important;
            width: calc(100vw - 310px) !important;
            background: #f7f9fa !important;
        }
        .leaflet-container {
            background: #f7f9fa !important;
            width: 100% !important;
            height: 100% !important;
        }
        .atlas-sidebar {
            width: 310px !important;
            flex-shrink: 0 !important;
            height: 100vh !important;
            background: #ffffff !important;
            border-left: 1px solid #e2e8f0 !important;
            padding: 22px 18px !important;
            display: flex !important;
            flex-direction: column !important;
            overflow-y: auto !important;
            z-index: 2000 !important;
        }
    </style>
    <script>
        // Force Leaflet to invalidate container size and load all edge tiles
        window.addEventListener('load', function() {
            setTimeout(function() {
                window.dispatchEvent(new Event('resize'));
                for (var k in window) {
                    if (window[k] && typeof window[k].invalidateSize === 'function') {
                        window[k].invalidateSize(true);
                    }
                }
            }, 150);
        });
    </script>
    """

    sidebar_html = f"""
    <div class="atlas-sidebar">
        <div style="margin-bottom: 18px;">
            <div style="font-size: 9px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 4px;">
                ATLAS &bull; KARLSRUHE
            </div>
            <div style="font-size: 14px; font-weight: 700; color: #0f172a; line-height: 1.3;">
                {display_title}
            </div>
        </div>
        
        <div style="padding-bottom: 12px; margin-bottom: 12px; border-bottom: 1px solid #f1f5f9;">
            <div style="font-size: 9px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">
                START POINT
            </div>
            <div style="display: flex; align-items: center; gap: 7px;">
                <span style="font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; font-weight: 700; color: #dc2626;">S</span>
                <span style="font-size: 12px; font-weight: 600; color: #1e293b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{origin_label}</span>
            </div>
        </div>

        <div style="font-size: 9px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">
            RECOMMENDATIONS
        </div>
        {items_html}
    </div>
    """

    final_html = raw_html.replace("</head>", f"{sidebar_css}</head>").replace("</body>", f"{sidebar_html}</body>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    return output_path


if __name__ == "__main__":
    print("Map renderer ready.")
