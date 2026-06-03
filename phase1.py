import random
from datetime import datetime, timedelta

import folium
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd


# --- 1. SIMULATION PARAMETERS ---
IS_RAINING = True
IS_RUSH_HOUR = True
GENERATE_SYNTHETIC_DATA = True

SOURCE_LAT, SOURCE_LON = 12.9698, 77.7500
DEST_LAT, DEST_LON = 12.9990, 77.7100
CENTER_LAT, CENTER_LON = 12.9840, 77.7300

GRAPH_DISTANCE_METERS = 10000
SYNTHETIC_EDGE_SAMPLE_SIZE = 500
SYNTHETIC_DAYS = 30
SYNTHETIC_START_DATE = datetime(2024, 1, 1)
RANDOM_SEED = 42


def normalize_highway_type(highway):
    """OSM highway tags may be lists; use the first value as the primary type."""
    if isinstance(highway, list):
        return highway[0]
    return highway or "unclassified"


def base_speed_for_route(highway):
    if highway in ["motorway", "motorway_link"]:
        return 80
    if highway in ["trunk", "trunk_link", "primary", "primary_link"]:
        return 60
    if highway in ["secondary", "secondary_link", "tertiary"]:
        return 40
    return 25


def apply_route_conditions(speed_kmh, highway, is_rush_hour, is_raining):
    if is_rush_hour:
        if highway in ["motorway", "motorway_link", "trunk", "trunk_link"]:
            speed_kmh *= 0.8
        elif highway in ["primary", "primary_link", "secondary", "secondary_link"]:
            speed_kmh *= 0.6
        else:
            speed_kmh *= 0.4

    if is_raining:
        if highway in ["motorway", "motorway_link", "trunk", "trunk_link"]:
            speed_kmh *= 0.9
        else:
            speed_kmh *= 0.7

    return max(speed_kmh, 1)


def base_speed_for_synthetic_data(highway):
    if highway in ["motorway", "motorway_link", "trunk", "trunk_link"]:
        return 80
    if highway in ["primary", "primary_link", "secondary", "secondary_link"]:
        return 50
    return 30


def calculate_synthetic_speed(highway, is_rush_hour, weather, rng):
    base_speed = base_speed_for_synthetic_data(highway)
    speed_modifier = 1.0

    if is_rush_hour:
        if highway in ["motorway", "motorway_link", "trunk", "trunk_link"]:
            speed_modifier -= 0.40
        else:
            speed_modifier -= 0.15

    if weather == "Rain":
        speed_modifier -= 0.10
    elif weather == "Heavy Rain":
        speed_modifier -= 0.25

    theoretical_speed = base_speed * speed_modifier
    final_speed = theoretical_speed + rng.normal(loc=0, scale=3.0)
    return max(5.0, round(final_speed, 2))


def add_travel_time_weights(G, is_rush_hour, is_raining):
    for _, _, _, data in G.edges(keys=True, data=True):
        highway = normalize_highway_type(data.get("highway", "unclassified"))
        speed_kmh = base_speed_for_route(highway)
        speed_kmh = apply_route_conditions(speed_kmh, highway, is_rush_hour, is_raining)

        distance_km = data.get("length", 0) / 1000
        data["travel_time"] = (distance_km / speed_kmh) * 60


def extract_route_points_and_totals(G, route):
    route_points = []
    total_length = 0
    total_time = 0

    for u, v in zip(route[:-1], route[1:]):
        edge = min(
            G.get_edge_data(u, v).values(),
            key=lambda edge_data: edge_data["travel_time"],
        )

        total_length += edge.get("length", 0)
        total_time += edge["travel_time"]

        if "geometry" in edge:
            xs, ys = edge["geometry"].xy
            route_points.extend((y, x) for x, y in zip(xs, ys))
        else:
            route_points.append((G.nodes[u]["y"], G.nodes[u]["x"]))

    route_points.append((G.nodes[route[-1]]["y"], G.nodes[route[-1]]["x"]))
    return route_points, total_length, total_time


def save_route_map(route_points):
    m = folium.Map(
        location=[(SOURCE_LAT + DEST_LAT) / 2, (SOURCE_LON + DEST_LON) / 2],
        zoom_start=10,
    )

    folium.PolyLine(
        route_points,
        weight=5,
        color="blue",
        opacity=0.8,
    ).add_to(m)

    folium.Marker(
        [SOURCE_LAT, SOURCE_LON],
        popup="Start",
        icon=folium.Icon(color="green"),
    ).add_to(m)
    folium.Marker(
        [DEST_LAT, DEST_LON],
        popup="End",
        icon=folium.Icon(color="red"),
    ).add_to(m)

    m.fit_bounds(route_points)
    m.save("predictive_route_map.html")


def generate_synthetic_traffic_csv(G):
    print("Starting Synthetic Data Generation...")

    random_picker = random.Random(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    edges_data = []
    for u, v, k, data in G.edges(keys=True, data=True):
        highway = normalize_highway_type(data.get("highway", "unclassified"))
        edges_data.append(
            {
                "edge_id": f"{u}_{v}_{k}",
                "highway_type": highway,
            }
        )

    if len(edges_data) > SYNTHETIC_EDGE_SAMPLE_SIZE:
        edges_data = random_picker.sample(edges_data, SYNTHETIC_EDGE_SAMPLE_SIZE)

    print(f"Tracking {len(edges_data)} road segments over {SYNTHETIC_DAYS} days...")

    weather_conditions = ["Clear", "Rain", "Heavy Rain"]
    weather_weights = [0.75, 0.20, 0.05]
    synthetic_records = []

    for hour_offset in range(SYNTHETIC_DAYS * 24):
        current_time = SYNTHETIC_START_DATE + timedelta(hours=hour_offset)
        hour_of_day = current_time.hour
        day_of_week = current_time.weekday()
        is_weekend = day_of_week >= 5
        is_rush_hour = (
            not is_weekend
            and (8 <= hour_of_day <= 10 or 17 <= hour_of_day <= 19)
        )
        current_weather = random_picker.choices(
            weather_conditions,
            weights=weather_weights,
        )[0]

        for edge in edges_data:
            highway = edge["highway_type"]
            final_speed = calculate_synthetic_speed(
                highway,
                is_rush_hour,
                current_weather,
                rng,
            )

            synthetic_records.append(
                {
                    "edge_id": edge["edge_id"],
                    "highway_type": highway,
                    "hour_of_day": hour_of_day,
                    "day_of_week": day_of_week,
                    "is_weekend": int(is_weekend),
                    "is_rush_hour": int(is_rush_hour),
                    "weather": current_weather,
                    "target_speed_kmh": final_speed,
                }
            )

    df = pd.DataFrame(synthetic_records)
    df.to_csv("synthetic_bengaluru_traffic.csv", index=False)

    print("Success! Generated CSV with", len(df), "rows.")
    print(df.head())


def main():
    print(f"Simulating Route | Rain: {IS_RAINING} | Rush Hour: {IS_RUSH_HOUR}")
    print("Downloading the map graph...")

    G = ox.graph_from_point(
        (CENTER_LAT, CENTER_LON),
        dist=GRAPH_DISTANCE_METERS,
        network_type="drive",
    )

    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    print("Calculating travel times based on road type, traffic, and weather...")
    add_travel_time_weights(G, IS_RUSH_HOUR, IS_RAINING)

    source_node = ox.distance.nearest_nodes(G, X=SOURCE_LON, Y=SOURCE_LAT)
    dest_node = ox.distance.nearest_nodes(G, X=DEST_LON, Y=DEST_LAT)

    print("Calculating fastest route...")
    route = ox.routing.shortest_path(
        G,
        source_node,
        dest_node,
        weight="travel_time",
    )

    if route is None:
        raise nx.NetworkXNoPath("No drivable route found between source and destination.")

    print(f"Route contains {len(route)} nodes")

    route_points, total_length, total_time = extract_route_points_and_totals(G, route)

    print(f"Total Route Distance: {total_length / 1000:.2f} km")
    print(f"Estimated Travel Time: {total_time:.1f} minutes")

    save_route_map(route_points)
    print("Map successfully saved as 'predictive_route_map.html'")

    if GENERATE_SYNTHETIC_DATA:
        generate_synthetic_traffic_csv(G)


if __name__ == "__main__":
    main()
