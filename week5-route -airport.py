# Databricks notebook source
import pandas as pd
df1 = pd.read_csv('/Volumes/workspace/default/airlines/Flight_delay_cleaned.csv')
display(df1.info())

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Create 'Route' column if not present
if 'Route' not in df1.columns:
    df1['Route'] = df1['Origin'] + '-' + df1['Destination']

# Get top 10 busiest routes
top_routes = df1['Route'].value_counts().head(10).reset_index()
top_routes.columns = ['Route', 'FlightCount']

# Gradient colors
colors = sns.color_palette("viridis", n_colors=10)

plt.figure(figsize=(12, 6))
bars = plt.bar(top_routes['Route'], top_routes['FlightCount'], color=colors)
plt.xlabel('Route')
plt.ylabel('Number of Flights')
plt.title('Top 10 Busiest Routes')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
display(plt.gcf())
plt.close()

# COMMAND ----------

route_delays = df1.groupby('Route')['TotalDelay'].mean().loc[top_routes['Route']]
colors = sns.color_palette("rocket", n_colors=len(route_delays))
plt.figure(figsize=(10,6))
sns.barplot(x=route_delays.values, y=route_delays.index, palette=colors)
plt.title("Average Total Delay for Top 10 Routes")
plt.xlabel("Average Delay (minutes)")
plt.ylabel("Route")
plt.tight_layout()
display(plt.gcf())
plt.close()

# COMMAND ----------

# Create 'Month' column if not present


# Pivot table: rows=Route, columns=Month, values=FlightCount
route_month_counts = df1.pivot_table(index='Route', columns='Month', values='FlightNum', aggfunc='count', fill_value=0)

# Select top 10 busiest routes
top_routes_list = top_routes['Route'].tolist()
route_month_counts = route_month_counts.loc[top_routes_list]

plt.figure(figsize=(14, 6))
sns.heatmap(route_month_counts, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Flight Counts by Route and Month (Top 10 Routes)')
plt.xlabel('Month')
plt.ylabel('Route')
plt.tight_layout()
display(plt.gcf())
plt.close()

# COMMAND ----------

# Heatmap: Average Total Delay by Route and DayOfWeek (Top 10 Routes)
route_day_delay = df1.pivot_table(index='Route', columns='DayOfWeek', values='TotalDelay', aggfunc='mean', fill_value=0)
route_day_delay = route_day_delay.loc[top_routes['Route']]
plt.figure(figsize=(14, 6))
sns.heatmap(route_day_delay, annot=True, fmt='.1f', cmap='Oranges')
plt.title('Average Total Delay by Route and DayOfWeek (Top 10 Routes)')
plt.xlabel('DayOfWeek')
plt.ylabel('Route')
plt.tight_layout()
display(plt.gcf())
plt.close()

# COMMAND ----------

# Heatmap: Flight Count by Route and DayOfWeek (Top 10 Routes)
route_day_counts = df1.pivot_table(index='Route', columns='DayOfWeek', values='FlightNum', aggfunc='count', fill_value=0)
route_day_counts = route_day_counts.loc[top_routes['Route']]
plt.figure(figsize=(14, 6))
sns.heatmap(route_day_counts, annot=True, fmt='d', cmap='Blues')
plt.title('Flight Count by Route and DayOfWeek (Top 10 Routes)')
plt.xlabel('DayOfWeek')
plt.ylabel('Route')
plt.tight_layout()
display(plt.gcf())
plt.close()

# COMMAND ----------

# Heatmap: Average Total Delay by Route and Month (Top 10 Routes)
route_month_delay = df1.pivot_table(index='Route', columns='Month', values='TotalDelay', aggfunc='mean', fill_value=0)
route_month_delay = route_month_delay.loc[top_routes['Route']]
plt.figure(figsize=(14, 6))
sns.heatmap(route_month_delay, annot=True, fmt='.1f', cmap='YlOrRd')
plt.title('Average Total Delay by Route and Month (Top 10 Routes)')
plt.xlabel('Month')
plt.ylabel('Route')
plt.tight_layout()
display(plt.gcf())
plt.close()

# COMMAND ----------

# MAGIC     %pip install geopandas

# COMMAND ----------

unique_airports = df1["Org_Airport"].unique()
import numpy as np
coords = {a: (np.random.uniform(25, 49), np.random.uniform(-125, -67)) for a in unique_airports}
df1["Latitude"] = df1["Org_Airport"].map(lambda x: coords[x][0])
df1["Longitude"] = df1["Org_Airport"].map(lambda x: coords[x][1])

# Display sample to verify
df1[["Org_Airport", "Latitude", "Longitude"]].head(10)

# COMMAND ----------

# MAGIC %pip install folium
# MAGIC

# COMMAND ----------

import folium
import seaborn as sns

# Calculate busiest airports by flight count
airport_counts = df1['Org_Airport'].value_counts().reset_index()
airport_counts.columns = ['Org_Airport', 'FlightCount']

# Merge with coordinates
airport_counts = airport_counts.merge(df1[['Org_Airport', 'Latitude', 'Longitude']].drop_duplicates(), on='Org_Airport')

# Assign a unique color to each airport
colors = sns.color_palette("husl", n_colors=len(airport_counts))
color_map = {airport: f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for airport, (r, g, b) in zip(airport_counts['Org_Airport'], colors)}

# Center map on US
m = folium.Map(location=[37.8, -96], zoom_start=4)

# Add airport markers as small points with different colors
for _, row in airport_counts.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=3,  # Small points
        popup=f"{row['Org_Airport']}: {row['FlightCount']} flights",
        color=color_map[row['Org_Airport']],
        fill=True,
        fill_color=color_map[row['Org_Airport']],
        fill_opacity=0.8
    ).add_to(m)

display(m)

# COMMAND ----------

import folium
import seaborn as sns

# Calculate average delay per airport
airport_delay = df1.groupby('Org_Airport')['TotalDelay'].mean().reset_index()
airport_delay.columns = ['Org_Airport', 'AvgDelay']

# Merge with coordinates
airport_delay = airport_delay.merge(df1[['Org_Airport', 'Latitude', 'Longitude']].drop_duplicates(), on='Org_Airport')

# Assign a unique color to each airport
colors = sns.color_palette("husl", n_colors=len(airport_delay))
color_map = {airport: f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for airport, (r, g, b) in zip(airport_delay['Org_Airport'], colors)}
airport_delay['color'] = airport_delay['Org_Airport'].map(color_map)

# Center map on US
m = folium.Map(location=[37.8, -96], zoom_start=4)

# Add airport markers colored by unique color
for _, row in airport_delay.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        popup=f"{row['Org_Airport']}: {row['AvgDelay']:.1f} min",
        color=row['color'],
        fill=True,
        fill_color=row['color'],
        fill_opacity=0.8
    ).add_to(m)

# Add least and most delay annotation at top right
least = airport_delay.loc[airport_delay['AvgDelay'].idxmin()]
most = airport_delay.loc[airport_delay['AvgDelay'].idxmax()]
html = f"""
<div style='position: absolute; top: 10px; right: 10px; background: white; padding: 10px; border-radius: 8px; box-shadow: 2px 2px 8px #888; z-index:9999;'>
<b>Least Delay:</b> {least['Org_Airport']} ({least['AvgDelay']:.1f} min)<br>
<b>Most Delay:</b> {most['Org_Airport']} ({most['AvgDelay']:.1f} min)
</div>
"""
m.get_root().html.add_child(folium.Element(html))

display(m)