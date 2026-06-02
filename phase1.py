import osmnx as ox
import networkx as nx 
import folium 

# --- 1. SIMULATION PARAMETERS (Toggle these to test different routes) ---
IS_RAINING = True
IS_RUSH_HOUR = True

print(f"Simulating Route | Rain: {IS_RAINING} | Rush Hour: {IS_RUSH_HOUR}")

# --- 2. LOCATION SETUP ---
source_lat, source_lon = 12.9698, 77.7500
dest_lat, dest_lon = 12.9990, 77.7100
center_lat, center_lon = 12.9840, 77.7300

print("Downloading the map graph (this is faster now)...")
# Using only graph_from_point to save memory and time
G = ox.graph_from_point(
    (center_lat, center_lon),
    dist=10000,
    network_type="drive"
)

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# --- 3. DYNAMIC EDGE WEIGHT CALCULATION (Phase 1.5 Logic) ---
print("Calculating travel times based on road type, traffic, and weather...")



for u, v, k, data in G.edges(keys=True, data=True):
    # OSM 'highway' tags can sometimes be a list. We extract the primary type.
    highway = data.get("highway", "unclassified")
    if isinstance(highway, list):
        highway = highway[0]

    # Step A: Assign base speed limits based on road type
    if highway in ["motorway", "motorway_link"]:
        speed_kmh = 80
    elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
        speed_kmh = 60
    elif highway in ["secondary", "secondary_link", "tertiary"]:
        speed_kmh = 40
    else:
        speed_kmh = 25 # Residential and unclassified

    # Step B: Apply environmental and traffic penalties
    if IS_RUSH_HOUR:

        if highway in ["motorway", "trunk"]:
         speed_kmh *= 0.8

        elif highway in ["primary", "secondary"]:
         speed_kmh *= 0.6

        else:
            speed_kmh *= 0.4
        
    if IS_RAINING:

        if highway in ["motorway", "trunk"]:
         speed_kmh *= 0.9

        else:
         speed_kmh *= 0.7
    

    # Step C: Calculate travel time (Weight)
    distance_km = data.get("length", 0) / 1000
    
    # Avoid division by zero
    if speed_kmh > 0:
        travel_time_minutes = (distance_km / speed_kmh) * 60
    else:
        travel_time_minutes = float('inf')

    # Assign this new metric to the edge
    data["travel_time"] = travel_time_minutes

# --- 4. ROUTING ENGINE ---
source_node = ox.distance.nearest_nodes(G, X=source_lon, Y=source_lat)
dest_node = ox.distance.nearest_nodes(G, X=dest_lon, Y=dest_lat)

print("Calculating fastest route...")
route = ox.routing.shortest_path(
    G,
    source_node,
    dest_node,
    weight="travel_time" # <--- Now routing based on time, not distance
)

print(f"Route contains {len(route)} nodes")

# --- 5. VISUALIZATION AND MAP BUILDING ---
m = folium.Map(
    location=[(source_lat + dest_lat) / 2, (source_lon + dest_lon) / 2],
    zoom_start=10
)

route_points = []

total_length = 0
total_time = 0

# Optimized loop: extracting geometry, length, and time simultaneously
for u, v in zip(route[:-1], route[1:]):
    # Get the specific edge data that minimizes travel time
    edge = min(
        G.get_edge_data(u, v).values(),
        key=lambda x: x["travel_time"]
    )
    
    total_length += edge["length"]
    total_time += edge["travel_time"]

    if "geometry" in edge:
        xs, ys = edge["geometry"].xy
        for x, y in zip(xs, ys):
            route_points.append((y, x))
    else:
        route_points.append((G.nodes[u]["y"], G.nodes[u]["x"]))

print(f"Total Route Distance: {total_length/1000:.2f} km")
print(f"Estimated Travel Time: {total_time:.1f} minutes")

# Draw the route
folium.PolyLine(
    route_points,
    weight=5,
    color="blue",
    opacity=0.8
).add_to(m)

# Markers
folium.Marker([source_lat, source_lon], popup="Start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker([dest_lat, dest_lon], popup="End", icon=folium.Icon(color="red")).add_to(m)

m.fit_bounds(route_points)
m.save("predictive_route_map.html")
print("Map successfully saved as 'predictive_route_map.html'")