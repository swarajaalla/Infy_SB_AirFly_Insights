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

