import osmnx as ox
import networkx as nx 
import folium 

place = "Whitefield,Bengaluru,India"

print("Downloading the location...")

G = ox.graph_from_place(place,network_type="drive")

print(f"number of nodes: {G.number_of_edges()}")
print(f"number of edges: {G.number_of_nodes()}")

source_lat, source_lon = 12.9698, 77.7500
dest_lat, dest_lon = 12.9900, 77.7300

source_node = ox.distance.nearest_nodes(G, X = source_lon , Y = source_lat)

dest_node = ox.distance.nearest_nodes(G, X = dest_lon , Y = dest_lat)

route = nx.astar_path(G,source_node,dest_node, weight = "length")

print(f"route contains {len(route)} nodes")

m = folium.Map(location=[source_lat,source_lon],zoom_start = 13)

route_points = []

for u, v in zip(route[:-1], route[1:]):

    edge = min(
        G.get_edge_data(u, v).values(),
        key=lambda x: x["length"]
    )

    if "geometry" in edge:
        xs, ys = edge["geometry"].xy

        for x, y in zip(xs, ys):
            route_points.append((y, x))

    else:
        route_points.append(
            (
                G.nodes[u]["y"],
                G.nodes[u]["x"]
            )
        )



folium.PolyLine(
    route_points,
    weight=5,
    color="blue"
).add_to(m)

folium.Marker([source_lat, source_lon],popup="Start").add_to(m)

folium.Marker([dest_lat,dest_lon],popup = "End").add_to(m)

m.save("route_map_for_sample.html")

print("Map successfully saved as html")




