"""
This script contains the helper functions to
extract pois, road, and nearest intersections 
within 150m radius of a stop from OpenStreetMap.
These features are added to stop_records in columns: POIs, Stop Road Info, Nearest Intersections
"""

import overpy
import folium
import geopy.distance as gd
import csv
import time
import sys

maxInt = sys.maxsize

while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

# Initialize Overpass API
api = overpy.Overpass()


def retry_overpass_query(query, max_retries=5, initial_wait=5):
    """
    Retry the overpass query if get the error too many request/server load too high.
    An alternative is to decrease the num_cores in parallel_process_csvs_3.py from multiprocessing.cpu_count() - 1 to multiprocessing.cpu_count() // 2
    :param query: the overpass api query
    :param max_retries: default 5
    :param initial_wait: default 5 seconds
    :return: the api query result / None
    """
    wait = initial_wait
    for attempt in range(max_retries):
        try:
            return api.query(query)
        except (overpy.exception.OverpassTooManyRequests, overpy.exception.OverpassGatewayTimeout) as e:
            print(f"⚠️ Overpass busy: {e}. Retrying in {wait} seconds...")
            time.sleep(wait)
            wait *= 2  # Exponential backoff
        except Exception as e:
            print(f"❌ Error in Overpass query: {e}")
            break
    return None  # Return None if all retries fail


def get_all_pois(lat, lon, radius=150, filter_types=None):
    """
    Fetch all Points of Interest (POIs) of certain types
    around a given latitude and longitude within a radius or 150 meters.
    :param lat: latitude of the stop point
    :param lon: longitude of the stop point
    :param radius: 150 meters
    :param filter_types: None
    :return: a list of pois sorted in increasing order of distance from the stop point / None
    """
    query = f"""
    [out:json][timeout:25];
    (
      node(around:{radius},{lat},{lon})["amenity"];
      node(around:{radius},{lat},{lon})["shop"];
      node(around:{radius},{lat},{lon})["public_transport"];
      node(around:{radius},{lat},{lon})["tourism"];
      node(around:{radius},{lat},{lon})["leisure"];
    );
    out body;
    >;
    out skel qt;
    """

    try:
        result = retry_overpass_query(query)
        if result is None:
            return []

        pois = []

        for node in result.nodes:

            # Determine POI type
            poi_type = (
                    node.tags.get("amenity") or
                    node.tags.get("shop") or
                    node.tags.get("public_transport") or
                    node.tags.get("tourism") or
                    node.tags.get("leisure") or
                    "Unknown"
            )

            # Apply filtering if needed
            if filter_types and poi_type not in filter_types:
                continue

            poi = {
                "name": node.tags.get("name", "Unknown"),
                "type": poi_type,
                "latitude": node.lat,
                "longitude": node.lon,
                "distance_m": round(gd.geodesic((lat, lon), (node.lat, node.lon)).meters, 3)  # Calculate distance in meters
            }
            pois.append(poi)
        return sorted(pois, key=lambda p: p["distance_m"])  # Sort POIs by distance

    except Exception as e:
        print(f"Error fetching POIs: {e}")
        return []


def get_road_info(lat, lon, radius=150):
    """
    Overpass Query to get roads, intersections, traffic signs, and road conditions
    of a given latitude and longitude within 150 meters radius
    :param lat: latitude of the stop point
    :param lon: longitude of the stop point
    :param radius: 150
    :return: 2 lists: 1.closest road information of the stop point within 150 meters,
    and 2.nearest <= 10 intersections around the stop point within 150 meters sorted in
    increasing order of distance to the stop point
    """
    query = f"""
    (
    way(around:{radius},{lat},{lon})["highway"];
    node(around: {radius}, {lat}, {lon})["highway"];
    node(around:{radius},{lat},{lon})["traffic_sign"];
    );
    out body;
    >;
    out skel qt;
    """

    # try:
    result = retry_overpass_query(query)
    if result is None:
        return [], []
    road_conditions, intersections = {}, {}

    node_counts = {}  # To track intersections
    node_way_map = {}
    node_distance = {}

    closest_way_ids = []
    closest_distance = float("inf")

    for way in result.ways:
        road_type = way.tags.get("highway", "")
        max_speed = way.tags.get("maxspeed", "")
        access = way.tags.get("access", "")
        lanes = way.tags.get("lanes", "")

        road_conditions[way.id] = {
            "road_type": road_type,
            "max_speed": max_speed,
            "access": access,
            "distance_m": "",
            "lanes": lanes
        }

        # Track nodes for intersection detection
        way_nodes = way.get_nodes(resolve_missing=True)
        for node in way_nodes:
            node_counts[node.id] = node_counts.get(node.id, 0) + 1
            node_way_map[node.id] = way.id
            distance = round(gd.geodesic((lat, lon), (node.lat, node.lon)).meters, 3)
            road_conditions[way.id]["distance_m"] = closest_distance
            node_distance[node.id] = distance
            if distance < closest_distance:
                closest_distance = distance
                closest_way_ids = [way.id]
            elif distance == closest_distance:
                closest_way_ids.append(way.id)

    road_conditions_closest = []
    for way_id in closest_way_ids:
        road_conditions_closest.append(road_conditions[way_id])

    # Identify intersections & compute distances, excludes small intersections that the same node appears twice
    for node in result.nodes:
        if node_counts.get(node.id, 0) >= 3:
            distance = node_distance[node.id]
            traffic_signs = []
            node_tags = node.tags
            if "traffic_sign" in node_tags:
                traffic_signs.append(node_tags["traffic_sign"])

            node_tags_crossing = node.tags.get("crossing", '')
            if node_tags_crossing == 'traffic_signals':
                traffic_signs.append("pedestrian_traffic_signals")

            node_tags_highway = node_tags.get("highway", "")
            if node_tags_highway in ["stop", "crossing", "traffic_signals", "give_way"]:
                traffic_signs.append(node_tags_highway)

            node_tags_railway = node_tags.get("railway", "")
            if node_tags_railway in ['crossing', 'level_crossing', 'tram_level_crossing']:
                traffic_signs.append("railway_crossing")

            intersections[(node.lat, node.lon)] = {"distance_m": distance,
                                                   "signs": traffic_signs,
                                                   "intersection_type": node_tags_highway,
                                                   "intersection_lanes": node_tags.get("lanes", ""),
                                                   "max_speed": node_tags.get("maxspeed", ""),
                                                   "access": node_tags.get("access", ""),
                                                   "crossing": node_tags_crossing,
                                                   "railway": node_tags_railway,
                                                   "barrier": node_tags.get("barrier", ""),
                                                   "road_type": road_conditions.get(node_way_map.get(node.id, ""), ""),
                                                   }

    # Sort and limit to 10 nearest intersections
    sorted_intersections = sorted(intersections.values(), key=lambda x: x["distance_m"])[:10]

    #print("sorted intersections", sorted_intersections)

    return road_conditions_closest, sorted_intersections

    # except Exception as e:
    #     print("Error fetching road info: {}, {}, {}".format(lat, lon, e))
    #     return {}


def process_csv(input_csv, output_csv):
    """
    Process a csv, fetch POIs & road info, and save the original + 3 new columns to a new csv.
    :param input_csv: the input csv to be processed to get poi and road, and nearest intersections info
    :param output_csv: the processed csv with newly added poi, stop road info and nearest intersections info
    :return:
    """
    with open(input_csv, newline='', encoding='utf-8') as infile, open(output_csv, "w", newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["POIs", "Stop Road Info", "Nearest Intersections"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # uncomment to creating map
        # locations, pois_all, intersections_all = [], [], []

        for row in reader:
            lat, lon = float(row["lat"]), float(row["lon"])

            # Fetch POIs and Road Info
            pois = get_all_pois(lat, lon)
            road_conditions, sorted_intersections = get_road_info(lat, lon)

            # Format POIs
            row["POIs"] = "; ".join([f"{p['type']} - {p['distance_m']}m" for p in pois])

            row["Stop Road Info"] = road_conditions

            row["Nearest Intersections"] = sorted_intersections

            writer.writerow(row)

            # uncomment creating map
            # locations.append((lat, lon))
            # pois_all.extend(pois)
            # intersections_all.extend(sorted_intersections)

        # uncomment creating map
        # create_map(locations, pois_all, intersections_all, "map_{}.html".format(output_csv))

        #print("Processing csv complete. outputted to {}".format(output_csv))


def create_map(locations, pois, intersections, filename):
    """Plot locations, POIs, and intersections on a map."""
    if not locations:
        print("No locations found.")
        return

    m = folium.Map(location=locations[0], zoom_start=15)

    # Add locations (Red Markers with Lat/Lon Popup)
    for lat, lon in locations:
        folium.Marker([lat, lon], popup=f"{lat}, {lon}", icon=folium.Icon(color="red")).add_to(m)

    # Add POIs (Blue Markers with Distance)
    for poi in pois:
        folium.Marker([poi["latitude"], poi["longitude"]],
                      popup=f"{poi['type']} ({poi['name']}) - {poi['distance_m']}m",
                      icon=folium.Icon(color="blue")).add_to(m)

    # Add Intersections (Green Markers with Distance & Traffic Signs)
    for int in intersections:
        intx = int[-1]
        popup_text = f"Intersection - {intx['distance_m']}m"
        if intx["signs"] is not None:
            print("intx[signs]", intx["signs"])
            popup_text += f" | Signs: {', '.join(intx['signs'])}"
        folium.Marker(
            [int[0][0], int[0][1]],
            popup=popup_text,
            icon=folium.Icon(color="green")
        ).add_to(m)

    m.save(filename)


# # Run the script
# start_time = time.time()
# process_csv("../input/training_merged_with_local_time.csv", "output3.csv")
# print("Total running time {} sec".format((time.time() - start_time)))