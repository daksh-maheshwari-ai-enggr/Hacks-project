import networkx as nx
import folium
from folium import plugins
import random
import math

def create_city_graph(grid_size=16):
    G_grid = nx.grid_2d_graph(grid_size, grid_size)
    G = nx.Graph()
    amenity_types = ['grocery_store', 'school', 'park', 'gas_station']
    coords = {}
    for x, y in G_grid.nodes():
        coords[(x,y)] = (x + random.uniform(-0.3, 0.3), y + random.uniform(-0.3, 0.3))
    
    for u, v in G_grid.edges():
        if random.random() > 0.15:
            G.add_edge(u, v, weight=1, traffic='green', blocked=False)
            
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    nodes = list(G.nodes())
    random.shuffle(nodes)
    
    num_amenities = int(len(nodes) * 0.08)
    num_homes = int(len(nodes) * 0.50)
    
    node_lats, node_lons = [coords[n][0] for n in G.nodes()], [coords[n][1] for n in G.nodes()]
    mid_lat, mid_lon = (min(node_lats) + max(node_lats)) / 2, (min(node_lons) + max(node_lons)) / 2
    spacing = 0.003
    
    for i, node in enumerate(nodes):
        G.nodes[node]['coord'] = ((coords[node][0] - mid_lat) * spacing, (coords[node][1] - mid_lon) * spacing)
        if i < num_amenities: G.nodes[node]['type'] = random.choice(amenity_types)
        elif i < num_amenities + num_homes: G.nodes[node]['type'] = 'home'
        else: G.nodes[node]['type'] = 'road'
            
    reqs = ['fire_station', 'hospital', 'police_station']
    for req in reqs:
        G.nodes[random.choice(nodes)]['type'] = req
        
    return G

def point_to_segment_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.hypot(px - closest_x, py - closest_y)

def process_click(G, click_lat, click_lng):
    min_node_dist = float('inf')
    best_node = None
    for n, d in G.nodes(data=True):
        lat, lng = d['coord']
        dist = math.hypot(lat - click_lat, lng - click_lng)
        if dist < min_node_dist:
            min_node_dist = dist
            best_node = n
            
    min_edge_dist = float('inf')
    best_edge = None
    for u, v, d in G.edges(data=True):
        lat1, lng1 = G.nodes[u]['coord']
        lat2, lng2 = G.nodes[v]['coord']
        dist = point_to_segment_dist(click_lat, click_lng, lat1, lng1, lat2, lng2)
        if dist < min_edge_dist:
            min_edge_dist = dist
            best_edge = (u, v)

    threshold_node, threshold_edge = 0.001, 0.001
    selected_node, selected_edge = None, None
    if min_node_dist <= threshold_node and min_node_dist <= min_edge_dist:
        selected_node = best_node
    elif min_edge_dist <= threshold_edge:
        selected_edge = best_edge
        
    return selected_node, selected_edge

def compute_all_routes(G, incident_type, incident_node):
    fac_map = {'Fire': 'fire_station', 'Medical emergency': 'hospital', 'Crime': 'police_station'}
    tgt = fac_map.get(incident_type)
    facs = [n for n, d in G.nodes(data=True) if d.get('type') == tgt]
    
    all_routes = []
    for fac in facs:
        try:
            route = nx.shortest_path(G, source=fac, target=incident_node, weight='weight')
            dist = nx.shortest_path_length(G, source=fac, target=incident_node, weight='weight')
            all_routes.append({"facility": str(fac), "route": route, "distance": dist})
        except: continue
    return all_routes

def increase_traffic(G, edge):
    if G.has_edge(*edge):
        d = G.edges[edge]
        if not d.get('blocked', False):
            if d.get('traffic', 'green') == 'green': d['traffic'], d['weight'] = 'orange', 3
            else: d['traffic'], d['weight'] = 'red', 8

def block_road(G, edge):
    if G.has_edge(*edge):
        d = G.edges[edge]
        d['blocked'], d['traffic'], d['weight'] = True, 'blocked', 999999

def render_folium_map(G, incident=None, route=None, vehicle_idx=None, selected_node=None, selected_edge=None):
    m = folium.Map(location=[0, 0], zoom_start=14, tiles='CartoDB dark_matter', control_scale=True)
    icon_map = {
        'hospital': ('plus', 'red'), 'police_station': ('shield', 'darkblue'),
        'fire_station': ('fire', 'orange'), 'grocery_store': ('shopping-cart', 'purple'),
        'school': ('book', 'green'), 'park': ('tree', 'darkgreen'), 'gas_station': ('car', 'darkred')
    }
    
    for u, v, d in G.edges(data=True):
        c1, c2 = G.nodes[u]['coord'], G.nodes[v]['coord']
        is_selected = (selected_edge) and ((u, v) == selected_edge or (v, u) == selected_edge)
        
        if d.get('blocked', False):
             folium.PolyLine([c1, c2], color=('yellow' if is_selected else 'red'), weight=(8 if is_selected else 5), opacity=0.9, dash_array='5, 5').add_to(m)
        else:
            t = d.get('traffic', 'green')
            base_color = '#2ecc71' if t == 'green' else '#f39c12' if t == 'orange' else '#e74c3c'
            folium.PolyLine([c1, c2], color=('yellow' if is_selected else base_color), weight=(8 if is_selected else (3 if t != 'green' else 1)), opacity=0.8).add_to(m)
            
    for n, d in G.nodes(data=True):
        c, ntype = d['coord'], d.get('type', 'road')
        is_selected = (n == selected_node)
        
        if ntype == 'home': 
            folium.CircleMarker(location=c, radius=(6 if is_selected else 3), color=('yellow' if is_selected else '#3498DB'), fill=True, fill_color=('yellow' if is_selected else '#3498DB'), fill_opacity=(0.9 if is_selected else 0.3), opacity=(0.9 if is_selected else 0.3)).add_to(m)
        elif ntype in icon_map:
            ic, col = icon_map[ntype]
            if is_selected: folium.CircleMarker(location=c, radius=12, color='yellow', fill=True, fill_opacity=0.6).add_to(m)
            folium.Marker(location=c, icon=folium.Icon(icon=ic, color=col, prefix='fa' if ic != 'info-sign' else 'glyphicon')).add_to(m)
        else:
            if is_selected: folium.CircleMarker(location=c, radius=6, color='yellow', fill=True, fill_opacity=0.9, opacity=0.9).add_to(m)

    if incident:
        ic = G.nodes[incident['node']]['coord']
        html = """<div style="background-color: red; width: 16px; height: 16px; border-radius: 50%; border: 2px solid white; animation: pulse 1s infinite;"></div>
        <style>@keyframes pulse { 0% {opacity: 1; transform: scale(1);} 50% {opacity: 0.5; transform: scale(1.5);} 100% {opacity: 1; transform: scale(1);} }</style>"""
        folium.Marker(location=ic, icon=folium.DivIcon(html=html)).add_to(m)
        
    if route:
        r_coords = [G.nodes[n]['coord'] for n in route]
        plugins.AntPath(locations=r_coords, color="cyan", weight=5, opacity=0.8, delay=400).add_to(m)
        if vehicle_idx is not None and vehicle_idx < len(route):
            vc = G.nodes[route[vehicle_idx]]['coord']
            folium.Marker(location=vc, icon=folium.Icon(icon='car', color='black', prefix='fa')).add_to(m)
            
    return m

G = create_city_graph()
m = render_folium_map(G)
m.save("city_map.html")