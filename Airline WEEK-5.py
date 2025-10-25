# Databricks notebook source
# Week 5: Route and Airport-Level Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the data
df = pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay.csv")

# Ensure Date is in datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Create derived columns
df['Route'] = df['Origin'] + "-" + df['Dest']
df['Month'] = df['Date'].dt.month
df['Season'] = df['Month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
})

# Total delay (sum of all delay components)
delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
df['TotalDelay'] = df[delay_cols].sum(axis=1)

# COMMAND ----------

# Top 10 Origin-Destination (Route) pairs by flight count

top_routes = df['Route'].value_counts().head(10).reset_index()
top_routes.columns = ['Route', 'FlightCount']

plt.figure(figsize=(10,6))
sns.barplot(x='FlightCount', y='Route', data=top_routes, palette='mako')
plt.title("Top 10 Busiest Routes by Flight Count")
plt.xlabel("Number of Flights")
plt.ylabel("Route")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Delay Heatmaps by Airport and Route

# Average delay by Origin and Destination

airport_delay = df.groupby(['Origin', 'Dest'])['TotalDelay'].mean().reset_index()

pivot_airport = airport_delay.pivot(index='Origin', columns='Dest', values='TotalDelay')
plt.figure(figsize=(12,8))
sns.heatmap(pivot_airport, cmap="coolwarm", linewidths=0.5)
plt.title("Average Delay Heatmap by Route (Origin-Destination)")
plt.xlabel("Destination Airport")
plt.ylabel("Origin Airport")
plt.show()

# COMMAND ----------

#Map: Busiest Airports & Average Delays

# Get airport-level data
origin_stats = df.groupby('Origin').agg({
    'FlightNum': 'count',
    'TotalDelay': 'mean'
}).reset_index().rename(columns={'FlightNum': 'FlightCount', 'TotalDelay': 'AvgDelay'})

# Example static mapping (replace or extend for your airports)
airport_coords = {
    'ATL': (33.6407, -84.4277), 'LAX': (33.9416, -118.4085),
    'ORD': (41.9742, -87.9073), 'DFW': (32.8998, -97.0403),
    'DEN': (39.8561, -104.6737), 'JFK': (40.6413, -73.7781),
    'SFO': (37.6213, -122.3790), 'SEA': (47.4502, -122.3088),
    'LAS': (36.0840, -115.1537), 'PHX': (33.4342, -112.0116)
}
coords_df = pd.DataFrame(airport_coords).T.reset_index()
coords_df.columns = ['Origin', 'Lat', 'Lon']

# Merge with stats
airport_map = pd.merge(origin_stats, coords_df, on='Origin', how='inner')

fig = px.scatter_mapbox(
    airport_map,
    lat='Lat',
    lon='Lon',
    size='FlightCount',
    color='AvgDelay',
    color_continuous_scale='RdYlGn_r',
    hover_name='Origin',
    title="Busiest Airports & Average Delays",
    mapbox_style='carto-positron',
    zoom=3
)
fig.show()

# COMMAND ----------

#Seasonal Summary Visualization

season_summary = df.groupby('Season')['TotalDelay'].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(x='Season', y='TotalDelay', data=season_summary, palette='viridis')
plt.title("Average Delay by Season")
plt.xlabel("Season")
plt.ylabel("Average Delay (minutes)")
plt.show()

# COMMAND ----------

import numpy as np
#assign random cooordinates for visualiazation

unique_airports = df['Org_Airport'].unique()
coords = {a:(np.random.uniform(25,49),np.random.uniform(-125,-67)) for a in unique_airports}

df["Latitutde"]= df["Org_Airport"].map(lambda x: coords[x][0])
df["Longitude"]= df["Org_Airport"].map(lambda x: coords[x][1])

# COMMAND ----------

# ASSIGN RANDOM COORDINATES FOR AIRPORTS

unique_airports = df['Origin'].unique()

# Randomly generate lat/lon for visualization (approx US bounds)
coords = {
    a: (np.random.uniform(25, 49), np.random.uniform(-125, -67))
    for a in unique_airports
}

df['Latitude'] = df['Origin'].map(lambda x: coords[x][0])
df['Longitude'] = df['Origin'].map(lambda x: coords[x][1])

# CREATE GEO MAP VISUALIZATION

fig = px.scatter_geo(
    df,
    lat="Latitude",
    lon="Longitude",
    hover_name="Origin",
    color="ArrDelay",
    color_continuous_scale="RdYlGn_r",
    projection="natural earth",
    title="Airport Average Delays (Simulated Coordinates)"
)

fig.show()
