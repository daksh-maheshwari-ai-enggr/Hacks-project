"""
==============================================================================
  CITY ROAD NETWORK SIMULATION ENGINE
  Emergency Response Simulator — Synthetic City Generator
  
  Generates a large (1000–1500 node) perturbed-grid city road network with
  typed nodes (houses, hospitals, police/fire stations, evacuation centers),
  realistic road attributes (traffic, blockages), emergency routing, and a
  rich interactive Folium map with animated dispatch visualisation.
==============================================================================
"""

import networkx as nx
import folium
from folium import plugins
import random
import math
import json

# ─── Reproducibility (optional; remove for true randomness) ─────────────────
random.seed(42)

# =============================================================================
# 1.  CITY GRAPH GENERATION
# =============================================================================

def create_city_graph(
    grid_rows: int = 38,
    grid_cols: int = 40,
    edge_removal_prob: float = 0.12,
    num_hospitals: int = 15,
    num_police_stations: int = 10,
    num_fire_stations: int = 8,
    num_evacuation_centers: int = 5,
    house_fraction: float = 0.60,
    center_lat: float = 28.6139,      # New Delhi centre (cosmetic)
    center_lon: float = 77.2090,
    spacing: float = 0.0012,          # ≈ 130 m per grid cell
    jitter: float = 0.25,            # perturbation factor (0 = perfect grid)
) -> nx.Graph:
    """
    Build a synthetic city graph on a perturbed grid.

    Returns a connected NetworkX Graph with ≥ 1000 nodes and typed /
    labelled nodes plus weighted, traffic-attributed edges.
    """

    # --- 1a. Build raw grid and perturb coordinates -------------------------
    raw_grid = nx.grid_2d_graph(grid_rows, grid_cols)

    # Perturbed (row, col) → (lat_offset, lon_offset) in grid units
    raw_coords: dict[tuple, tuple] = {}
    for r, c in raw_grid.nodes():
        raw_coords[(r, c)] = (
            r + random.uniform(-jitter, jitter),
            c + random.uniform(-jitter, jitter),
        )

    # --- 1b. Build the actual graph, randomly dropping edges ----------------
    G = nx.Graph()
    G.add_nodes_from(raw_grid.nodes())
    for u, v in raw_grid.edges():
        if random.random() > edge_removal_prob:
            G.add_edge(u, v)

    # --- 1c. Keep only the largest connected component ----------------------
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    print(f"[city_simulation] Graph has {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges after pruning.")

    # --- 1d. Compute real-world lat/lon for each surviving node -------------
    surviving = list(G.nodes())
    lats = [raw_coords[n][0] for n in surviving]
    lons = [raw_coords[n][1] for n in surviving]
    mid_lat = (min(lats) + max(lats)) / 2
    mid_lon = (min(lons) + max(lons)) / 2

    for n in surviving:
        lat = center_lat + (raw_coords[n][0] - mid_lat) * spacing
        lon = center_lon + (raw_coords[n][1] - mid_lon) * spacing
        G.nodes[n]["coord"] = (lat, lon)

    # --- 1e. Assign node types + labels -------------------------------------
    nodes_shuffled = surviving.copy()
    random.shuffle(nodes_shuffled)

    # Fixed-count facilities first
    facility_specs = [
        ("hospital",          num_hospitals),
        ("police_station",    num_police_stations),
        ("fire_station",      num_fire_stations),
        ("evacuation_center", num_evacuation_centers),
    ]

    idx = 0
    counters: dict[str, int] = {}
    for ftype, count in facility_specs:
        counters[ftype] = 0
        for _ in range(count):
            if idx >= len(nodes_shuffled):
                break
            node = nodes_shuffled[idx]
            counters[ftype] += 1
            G.nodes[node]["type"] = ftype
            G.nodes[node]["label"] = _label(ftype, counters[ftype])
            idx += 1

    # Houses and road intersections for the rest
    remaining = nodes_shuffled[idx:]
    num_houses = int(len(remaining) * (house_fraction / (1.0 - 0.0)))
    # (house_fraction is relative to all nodes, but facilities already set)
    # Recompute to honour ~60 % of TOTAL nodes being houses:
    target_houses = int(len(nodes_shuffled) * house_fraction)
    num_houses = min(target_houses, len(remaining))

    house_counter = 0
    intersection_counter = 0
    for j, node in enumerate(remaining):
        if j < num_houses:
            house_counter += 1
            G.nodes[node]["type"] = "house"
            G.nodes[node]["label"] = f"House {house_counter}"
        else:
            intersection_counter += 1
            G.nodes[node]["type"] = "road_intersection"
            G.nodes[node]["label"] = f"Intersection {intersection_counter}"

    # --- 1f. Assign edge attributes -----------------------------------------
    _assign_edge_attributes(G)

    return G


def _label(node_type: str, index: int) -> str:
    """Human-readable label for a facility node."""
    names = {
        "hospital":          "Hospital",
        "police_station":    "Police Station",
        "fire_station":      "Fire Station",
        "evacuation_center": "Evacuation Center",
    }
    return f"{names[node_type]} {index}"


def _assign_edge_attributes(G: nx.Graph) -> None:
    """Assign weight, traffic_level, and blocked status to every edge."""
    for u, v in G.edges():
        traffic = random.choices(
            ["green", "orange", "red"],
            weights=[0.70, 0.20, 0.10],
            k=1,
        )[0]
        weight = _traffic_weight(traffic)
        # Add small Euclidean component so distances are meaningful
        lat1, lon1 = G.nodes[u]["coord"]
        lat2, lon2 = G.nodes[v]["coord"]
        eucl = math.hypot(lat1 - lat2, lon1 - lon2) * 1000  # scale up
        G.edges[u, v]["traffic_level"] = traffic
        G.edges[u, v]["blocked"] = False
        G.edges[u, v]["weight"] = weight + eucl


def _traffic_weight(traffic: str) -> float:
    """Base weight for a traffic level."""
    return {"green": 1.0, "orange": 3.0, "red": 8.0}.get(traffic, 1.0)


# =============================================================================
# 2.  TRAFFIC & ROAD MANIPULATION
# =============================================================================

def increase_traffic(G: nx.Graph, edge: tuple) -> None:
    """
    Escalate traffic on *edge* by one level:
      green → orange → red → (stays red)
    Updates the edge weight accordingly.
    """
    if not G.has_edge(*edge):
        return
    d = G.edges[edge]
    if d.get("blocked", False):
        return  # blocked roads are not escalated
    current = d.get("traffic_level", "green")
    if current == "green":
        d["traffic_level"] = "orange"
        d["weight"] = 3.0 + _euclidean_component(G, edge)
    elif current == "orange":
        d["traffic_level"] = "red"
        d["weight"] = 8.0 + _euclidean_component(G, edge)
    # red stays red


def collapse_road(G: nx.Graph, edge: tuple) -> None:
    """
    Mark a road as collapsed / impassable.
    Sets blocked=True, traffic_level='blocked', weight=999999.
    """
    if not G.has_edge(*edge):
        return
    d = G.edges[edge]
    d["blocked"] = True
    d["traffic_level"] = "blocked"
    d["weight"] = 999_999


def _euclidean_component(G: nx.Graph, edge: tuple) -> float:
    """Small Euclidean distance contribution for an edge."""
    u, v = edge
    lat1, lon1 = G.nodes[u]["coord"]
    lat2, lon2 = G.nodes[v]["coord"]
    return math.hypot(lat1 - lat2, lon1 - lon2) * 1000


# =============================================================================
# 3.  EMERGENCY ROUTING
# =============================================================================

# Mapping: incident type → facility node type to dispatch FROM
INCIDENT_FACILITY_MAP = {
    "Fire":       "fire_station",
    "Medical":    "hospital",
    "Crime":      "police_station",
    "Evacuation": "evacuation_center",
}


def compute_routes(
    G: nx.Graph,
    incident_type: str,
    incident_node: tuple,
) -> list[dict]:
    """
    Compute shortest weighted paths from every facility of the relevant type
    to *incident_node*.

    Returns a list of dicts sorted by distance:
        [{"facility": node, "label": str, "route": [nodes], "distance": float}, ...]
    """
    facility_type = INCIDENT_FACILITY_MAP.get(incident_type)
    if facility_type is None:
        raise ValueError(f"Unknown incident type: {incident_type!r}. "
                         f"Choose from {list(INCIDENT_FACILITY_MAP)}")

    facilities = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == facility_type
    ]

    results = []
    for fac in facilities:
        try:
            path = nx.shortest_path(G, source=fac, target=incident_node, weight="weight")
            dist = nx.shortest_path_length(G, source=fac, target=incident_node, weight="weight")
            results.append({
                "facility": fac,
                "label": G.nodes[fac].get("label", str(fac)),
                "route": path,
                "distance": round(dist, 2),
            })
        except nx.NetworkXNoPath:
            continue

    results.sort(key=lambda r: r["distance"])
    return results


# =============================================================================
# 4.  SELECTION HELPERS (click → nearest node / edge)
# =============================================================================

def find_nearest_node(G: nx.Graph, click_lat: float, click_lng: float) -> tuple:
    """Return the graph node closest to (click_lat, click_lng)."""
    best, best_dist = None, float("inf")
    for n, d in G.nodes(data=True):
        lat, lon = d["coord"]
        dist = math.hypot(lat - click_lat, lon - click_lng)
        if dist < best_dist:
            best_dist = dist
            best = n
    return best


def find_nearest_edge(G: nx.Graph, click_lat: float, click_lng: float) -> tuple:
    """Return the graph edge (u, v) closest to (click_lat, click_lng)."""
    best, best_dist = None, float("inf")
    for u, v in G.edges():
        lat1, lon1 = G.nodes[u]["coord"]
        lat2, lon2 = G.nodes[v]["coord"]
        dist = _point_to_segment_dist(click_lat, click_lng, lat1, lon1, lat2, lon2)
        if dist < best_dist:
            best_dist = dist
            best = (u, v)
    return best


def _point_to_segment_dist(px, py, x1, y1, x2, y2) -> float:
    """Euclidean distance from point (px, py) to segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


# =============================================================================
# 5.  FOLIUM MAP RENDERING
# =============================================================================

# Visual configuration ---
_NODE_STYLE = {
    "house":              {"radius": 3,  "color": "#3B82F6",  "fill_opacity": 0.35, "opacity": 0.35},
    "road_intersection":  {"radius": 2,  "color": "#6B7280",  "fill_opacity": 0.20, "opacity": 0.20},
    "hospital":           {"icon": "plus",          "fa": True,  "marker_color": "red"},
    "police_station":     {"icon": "shield",        "fa": True,  "marker_color": "darkblue"},
    "fire_station":       {"icon": "fire-extinguisher", "fa": True, "marker_color": "orange"},
    "evacuation_center":  {"icon": "home",          "fa": True,  "marker_color": "purple"},
}

_TRAFFIC_COLORS = {
    "green":   "#22C55E",
    "orange":  "#F59E0B",
    "red":     "#EF4444",
    "blocked": "#EF4444",
}


def render_map(
    G: nx.Graph,
    incident_node: tuple | None = None,
    dispatch_route: list | None = None,
    vehicle_index: int | None = None,
    selected_node: tuple | None = None,
    selected_edge: tuple | None = None,
    tile_style: str = "CartoDB dark_matter",
) -> folium.Map:
    """
    Build a rich Folium map from the city graph.

    Parameters
    ----------
    G               : city graph
    incident_node   : node to mark with a pulsing red marker
    dispatch_route  : list of nodes forming the dispatch path (AntPath)
    vehicle_index   : index into dispatch_route for the vehicle marker
    selected_node   : highlight a node (yellow ring)
    selected_edge   : highlight an edge (yellow)
    tile_style      : Folium tile layer name

    Returns
    -------
    folium.Map
    """

    # --- Determine map centre from graph bounding box -----------------------
    all_coords = [d["coord"] for _, d in G.nodes(data=True)]
    lats = [c[0] for c in all_coords]
    lons = [c[1] for c in all_coords]
    center = ((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2)

    m = folium.Map(
        location=center,
        zoom_start=14,
        tiles=tile_style,
        control_scale=True,
        prefer_canvas=True,        # much faster for large graphs
    )

    # ---- Layer groups for toggling -----------------------------------------
    road_layer      = folium.FeatureGroup(name="Roads", show=True)
    house_layer     = folium.FeatureGroup(name="Houses", show=True)
    facility_layer  = folium.FeatureGroup(name="Facilities", show=True)
    incident_layer  = folium.FeatureGroup(name="Incident", show=True)
    route_layer     = folium.FeatureGroup(name="Dispatch Route", show=True)

    # ── 5a. Draw roads (edges) ──────────────────────────────────────────────
    for u, v, d in G.edges(data=True):
        c1 = G.nodes[u]["coord"]
        c2 = G.nodes[v]["coord"]
        is_sel = selected_edge and ((u, v) == selected_edge or (v, u) == selected_edge)
        traffic = d.get("traffic_level", "green")
        blocked = d.get("blocked", False)

        if blocked:
            folium.PolyLine(
                [c1, c2],
                color="yellow" if is_sel else _TRAFFIC_COLORS["blocked"],
                weight=7 if is_sel else 4,
                opacity=0.85,
                dash_array="8, 6",
                tooltip="BLOCKED",
            ).add_to(road_layer)
        else:
            color = "yellow" if is_sel else _TRAFFIC_COLORS.get(traffic, "#22C55E")
            w = 6 if is_sel else (2 if traffic == "green" else 3)
            folium.PolyLine(
                [c1, c2],
                color=color,
                weight=w,
                opacity=0.75,
                tooltip=f"Traffic: {traffic} | Weight: {d.get('weight', '?'):.1f}",
            ).add_to(road_layer)

    # ── 5b. Draw nodes ──────────────────────────────────────────────────────
    for n, d in G.nodes(data=True):
        coord = d["coord"]
        ntype = d.get("type", "road_intersection")
        label = d.get("label", str(n))
        is_sel = (n == selected_node)
        style = _NODE_STYLE.get(ntype, _NODE_STYLE["road_intersection"])

        if ntype in ("house", "road_intersection"):
            # Circle markers for houses / intersections
            folium.CircleMarker(
                location=coord,
                radius=6 if is_sel else style["radius"],
                color="yellow" if is_sel else style["color"],
                fill=True,
                fill_color="yellow" if is_sel else style["color"],
                fill_opacity=0.9 if is_sel else style["fill_opacity"],
                opacity=0.9 if is_sel else style["opacity"],
                tooltip=label,
            ).add_to(house_layer)
        else:
            # Icon markers for facilities
            if is_sel:
                folium.CircleMarker(
                    location=coord, radius=14,
                    color="yellow", fill=True, fill_opacity=0.55,
                ).add_to(facility_layer)
            folium.Marker(
                location=coord,
                icon=folium.Icon(
                    icon=style["icon"],
                    color=style["marker_color"],
                    prefix="fa" if style.get("fa") else "glyphicon",
                ),
                tooltip=label,
            ).add_to(facility_layer)

    # ── 5c. Incident pulsing marker ─────────────────────────────────────────
    if incident_node is not None and incident_node in G:
        ic = G.nodes[incident_node]["coord"]
        pulse_html = """
        <div style="
            background: radial-gradient(circle, #ff0000 30%, transparent 70%);
            width: 28px; height: 28px; border-radius: 50%;
            border: 2px solid #ffffff;
            animation: pulse-anim 1.2s ease-in-out infinite;
            box-shadow: 0 0 12px 4px rgba(255,0,0,0.6);
        "></div>
        <style>
        @keyframes pulse-anim {
            0%   { opacity: 1;   transform: scale(1);   }
            50%  { opacity: 0.5; transform: scale(1.6);  }
            100% { opacity: 1;   transform: scale(1);   }
        }
        </style>
        """
        folium.Marker(
            location=ic,
            icon=folium.DivIcon(html=pulse_html, icon_size=(28, 28), icon_anchor=(14, 14)),
            tooltip=f"🚨 INCIDENT at {G.nodes[incident_node].get('label', incident_node)}",
        ).add_to(incident_layer)

    # ── 5d. Dispatch route (AntPath) + vehicle marker ───────────────────────
    if dispatch_route:
        route_coords = [G.nodes[n]["coord"] for n in dispatch_route if n in G]
        if len(route_coords) >= 2:
            plugins.AntPath(
                locations=route_coords,
                color="#06B6D4",
                pulse_color="#FFFFFF",
                weight=5,
                opacity=0.85,
                delay=600,
            ).add_to(route_layer)

            # Start marker (facility)
            folium.Marker(
                location=route_coords[0],
                icon=folium.Icon(icon="play", color="green", prefix="fa"),
                tooltip="Dispatch origin",
            ).add_to(route_layer)

            # End marker (incident)
            folium.Marker(
                location=route_coords[-1],
                icon=folium.Icon(icon="flag-checkered", color="red", prefix="fa"),
                tooltip="Incident location",
            ).add_to(route_layer)

        # Vehicle marker
        if vehicle_index is not None and 0 <= vehicle_index < len(dispatch_route):
            vc = G.nodes[dispatch_route[vehicle_index]]["coord"]
            vehicle_html = """
            <div style="
                font-size: 22px; text-shadow: 0 0 6px rgba(0, 200, 255, 0.9);
            ">🚒</div>
            """
            folium.Marker(
                location=vc,
                icon=folium.DivIcon(html=vehicle_html, icon_size=(28, 28), icon_anchor=(14, 14)),
                tooltip="Emergency vehicle en route",
            ).add_to(route_layer)

    # ── Add layers + control ────────────────────────────────────────────────
    road_layer.add_to(m)
    house_layer.add_to(m)
    facility_layer.add_to(m)
    incident_layer.add_to(m)
    route_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # ── Custom legend (HTML overlay) ────────────────────────────────────────
    legend_html = _build_legend_html()
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def _build_legend_html() -> str:
    """Generate a floating HTML legend for the map."""
    return """
    <div id="map-legend" style="
        position: fixed; bottom: 30px; left: 14px; z-index: 9999;
        background: rgba(17, 24, 39, 0.92); backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.12); border-radius: 12px;
        padding: 16px 20px; font-family: 'Segoe UI', system-ui, sans-serif;
        color: #e5e7eb; font-size: 13px; line-height: 1.7;
        box-shadow: 0 8px 32px rgba(0,0,0,0.5); max-width: 220px;
    ">
        <div style="font-weight:700; font-size:14px; margin-bottom:8px;
             border-bottom:1px solid rgba(255,255,255,0.15); padding-bottom:6px;">
            🗺️ Map Legend
        </div>
        <div><span style="color:#3B82F6;">●</span> House</div>
        <div><span style="color:#6B7280;">●</span> Intersection</div>
        <div><span style="color:#EF4444;">✚</span> Hospital</div>
        <div><span style="color:#3B82F6;">🛡</span> Police Station</div>
        <div><span style="color:#F59E0B;">🧯</span> Fire Station</div>
        <div><span style="color:#A855F7;">🏠</span> Evacuation Center</div>
        <hr style="border-color:rgba(255,255,255,0.15); margin:6px 0;">
        <div><span style="color:#22C55E;">━</span> Green traffic</div>
        <div><span style="color:#F59E0B;">━</span> Orange traffic</div>
        <div><span style="color:#EF4444;">━</span> Red traffic</div>
        <div><span style="color:#EF4444;">╌</span> Blocked road</div>
        <div><span style="color:#06B6D4;">⇢</span> Dispatch route</div>
    </div>
    """


# =============================================================================
# 6.  STATISTICS HELPER
# =============================================================================

def graph_summary(G: nx.Graph) -> dict:
    """Return a summary dict of the graph composition."""
    types = {}
    for _, d in G.nodes(data=True):
        t = d.get("type", "unknown")
        types[t] = types.get(t, 0) + 1

    traffic_counts = {"green": 0, "orange": 0, "red": 0, "blocked": 0}
    for _, _, d in G.edges(data=True):
        tl = d.get("traffic_level", "green")
        traffic_counts[tl] = traffic_counts.get(tl, 0) + 1

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "node_types": types,
        "traffic_distribution": traffic_counts,
        "is_connected": nx.is_connected(G),
    }


# =============================================================================
# 7.  MAIN — Generate graph + map and save
# =============================================================================

def main():
    print("=" * 70)
    print("  CITY SIMULATION — Generating synthetic road network …")
    print("=" * 70)

    # ── Build the city graph ────────────────────────────────────────────────
    G = create_city_graph()
    summary = graph_summary(G)
    print(f"\n📊  Graph Summary")
    print(f"   Nodes : {summary['total_nodes']}")
    print(f"   Edges : {summary['total_edges']}")
    print(f"   Types : {json.dumps(summary['node_types'], indent=10)}")
    print(f"   Traffic: {summary['traffic_distribution']}")
    print(f"   Connected: {summary['is_connected']}")

    # ── Demo: pick an incident and compute routes ───────────────────────────
    # Select a random house as the incident location
    houses = [n for n, d in G.nodes(data=True) if d.get("type") == "house"]
    incident_node = random.choice(houses) if houses else list(G.nodes())[0]
    incident_type = "Fire"

    print(f"\n🔥  Simulated incident: {incident_type}")
    print(f"   Location: {G.nodes[incident_node].get('label', incident_node)}")
    print(f"   Coordinates: {G.nodes[incident_node]['coord']}")

    routes = compute_routes(G, incident_type, incident_node)
    print(f"\n🚒  Found {len(routes)} route(s) from fire stations:")
    for i, r in enumerate(routes):
        print(f"   {i+1}. {r['label']}  —  distance {r['distance']:.1f}  "
              f"({len(r['route'])} hops)")

    # Pick the best (shortest) route for visualisation
    best_route = routes[0]["route"] if routes else None

    # ── Demo: collapse a random road + increase traffic on a few edges ─────
    edges = list(G.edges())
    if len(edges) > 5:
        collapse_road(G, edges[0])
        for e in edges[1:4]:
            increase_traffic(G, e)
        print(f"\n🚧  Collapsed road: {edges[0]}")
        print(f"   Increased traffic on {edges[1:4]}")

    # ── Render the interactive Folium map ───────────────────────────────────
    print("\n🗺️   Rendering Folium map …")
    m = render_map(
        G,
        incident_node=incident_node,
        dispatch_route=best_route,
        vehicle_index=len(best_route) // 3 if best_route else None,
    )

    output_path = "city_simulation_map.html"
    m.save(output_path)
    print(f"✅  Map saved to  →  {output_path}")
    print("   Open the file in your browser to explore the simulation.\n")

    return G, m


def graph_to_citystate(G):

    nodes=[]
    edges=[]

    for n,d in G.nodes(data=True):

        lat,lon = d["coord"]

        nodes.append({
            "id":str(n),
            "lat":lat,
            "lng":lon,
            "type":d["type"],
            "label":d.get("label","")
        })

    for i,(u,v,d) in enumerate(G.edges(data=True)):

        c1 = G.nodes[u]["coord"]
        c2 = G.nodes[v]["coord"]

        traffic_map={
            "green":0.2,
            "orange":0.6,
            "red":0.9
        }

        edges.append({
            "id":f"edge_{i}",
            "source":str(u),
            "target":str(v),
            "traffic_level":traffic_map.get(d.get("traffic_level","green"),0.2),
            "collapsed":d.get("blocked",False),
            "coordinates":[
                [c1[0],c1[1]],
                [c2[0],c2[1]]
            ]
        })

    return {
        "nodes":nodes,
        "edges":edges,
        "incidents":[]
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    city_graph, city_map = main()



