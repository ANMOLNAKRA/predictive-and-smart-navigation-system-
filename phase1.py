import osmnx as ox
import networkx as nx 
import folium 

place = "Whitefield","Bengaluru","India"

print("Downloading the location...")

G = ox.graph_from_place(place,network_type="drive")

print(f"number of nodes: {G.number_of_edges()}")
print(f"number of edges: {G.number_of_nodes()}")

source_lat, source_lon = 12.9698, 77.7500
dest_lat, dest_lon = 12.9900, 77.7300

source_node = ox.distance.nearest_nodes(G,x = source_lat , y = source_lon)

dest_node = ox.distance.nearest_nodes(G, x = dest_lat , y= dest_lon)

route = nx.astar_path(G,source_node,dest_node,weight = "length")

print(f"route contains {len(route)} nodes")

m = folium.Map(location=[source_lat,source_lon],zoom_start = 13)

route_coords =[]

for node in route:
    route_coords.append((G.nodes[node]["y"],G.nodes[node]["x"]))



folium.PolyLine(route_coords,weight=5).add_to(m)

folium.Marker([source_lat, source_lon],popup="Start").add_to(m)

folium.Marker([dest_lat,dest_lon],popup = "End".add_to(m))

m.save("route_map_for_sample.html")

print("Map successfully saved as html")




