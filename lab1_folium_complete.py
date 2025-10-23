# Lab 1 — Interactive Visual Analytics with Folium (Complete)
# -----------------------------------------------------------
# Drop-in cells for the Skills Network Folium lab.
# Assumes the notebook already installed folium & pandas and loaded the SpaceX dataset into spacex_df.
# This script provides the missing TODO completions end-to-end.

# Imports (keep consistent with your notebook)
import folium
import pandas as pd
from folium.plugins import MarkerCluster, MousePosition
from folium.features import DivIcon

# ---------- Helpers ----------
def calculate_distance(lat1, lon1, lat2, lon2):
    # Haversine (km)
    from math import sin, cos, sqrt, atan2, radians
    R = 6373.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def add_distance_line(map_obj, site_name: str, launch_sites_df: pd.DataFrame, target_lat: float, target_lon: float, label: str):
    row = launch_sites_df.loc[launch_sites_df['Launch Site'] == site_name].iloc[0]
    slat, slon = float(row['Lat']), float(row['Long'])
    d_km = calculate_distance(slat, slon, target_lat, target_lon)
    # line
    folium.PolyLine([[slat, slon], [target_lat, target_lon]], weight=1).add_to(map_obj)
    # label
    folium.Marker(
        [target_lat, target_lon],
        icon=DivIcon(icon_size=(20,20), icon_anchor=(0,0),
                     html=f'<div style="font-size:12px; color:#d35400;"><b>{label}: {d_km:.2f} KM</b></div>')
    ).add_to(map_obj)
    return d_km

# ---------- Task 0: Load/prepare launch_sites_df ----------
# Expect spacex_df with columns: 'Launch Site','Lat','Long','class'
# If your notebook already created launch_sites_df, skip this cell.
if 'launch_sites_df' not in globals():
    launch_sites_df = (spacex_df[['Launch Site','Lat','Long']]
                       .groupby('Launch Site', as_index=False).first())

# ---------- Task 1: Mark all launch sites on a map ----------
# Start centered at NASA JSC (as in lab)
nasa_coordinate = [29.55968488888889, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)

for _, r in launch_sites_df.iterrows():
    folium.Circle([r['Lat'], r['Long']], radius=1000, color="#d35400", fill=True)\
        .add_child(folium.Popup(r['Launch Site'])).add_to(site_map)
    folium.map.Marker([r['Lat'], r['Long']],
                      icon=DivIcon(icon_size=(20,20), icon_anchor=(0,0),
                                   html=f'<div style="font-size:12px; color:#d35400;"><b>{r["Launch Site"]}</b></div>')
                     ).add_to(site_map)

# Display
site_map

# ---------- Task 2: Mark success/fail launches with MarkerCluster ----------
marker_cluster = MarkerCluster()
site_map.add_child(marker_cluster)

# color by 'class' (1=success -> green, 0=fail -> red)
spacex_df['marker_color'] = spacex_df['class'].apply(lambda x: 'green' if x==1 else 'red')

for _, row in spacex_df.iterrows():
    marker = folium.Marker(location=[row['Lat'], row['Long']],
                           icon=folium.Icon(color=row['marker_color'],
                                            icon='ok-sign' if row['marker_color']=='green' else 'remove'))
    marker_cluster.add_child(marker)

# Display cluster
site_map

# ---------- Task 3: Proximity distances (coastline, railway, highway, city) ----------
# Show mouse position readout to pick coordinates on the map
formatter = "function(num){return L.Util.formatNum(num, 5);}"
mouse_position = MousePosition(position='topright', separator=' | ',
                               empty_string='NaN', lng_first=False,
                               num_digits=20, prefix='lat:',
                               lat_formatter=formatter, lng_formatter=formatter)
site_map.add_child(mouse_position)

# Example target coordinates near CCAFS SLC-40 (replace as needed with your own readings)
site_name = "CCAFS SLC-40"
coast_lat, coast_lon = 28.56367, -80.57163
rail_lat,  rail_lon  = 28.56310, -80.57100
hwy_lat,   hwy_lon   = 28.56410, -80.56990
city_lat,  city_lon  = 28.39220, -80.60770  # Cape Canaveral vicinity

d_coast = add_distance_line(site_map, site_name, launch_sites_df, coast_lat, coast_lon, "Coastline")
d_rail  = add_distance_line(site_map, site_name, launch_sites_df, rail_lat, rail_lon, "Railway")
d_hwy   = add_distance_line(site_map, site_name, launch_sites_df, hwy_lat, hwy_lon, "Highway")
d_city  = add_distance_line(site_map, site_name, launch_sites_df, city_lat, city_lon, "City")

# Final map with lines & labels
site_map