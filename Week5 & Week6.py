# Databricks notebook source
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df = pd.read_csv("/Volumes/workspace/default/airdelay/Flight_delay_cleaned.csv")

# COMMAND ----------


top10_pairs = df.groupby(['Org_Airport', 'Dest_Airport']).size().reset_index(name='count').sort_values('count', ascending=False).head(10)
display(top10_pairs)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top10_pairs,
    x='count',
    y=top10_pairs['Org_Airport'] + " → " + top10_pairs['Dest_Airport'],
    palette='viridis'
)
plt.xlabel('Number of Flights')
plt.ylabel('Route (Origin → Destination)')
plt.title('Top 10 Most Frequent Flight Routes')
plt.tight_layout()
plt.show()

# COMMAND ----------

#Heatmap: Average departure delay by origin and destination airport (route)
if all(col in df.columns for col in ["Org_Airport", 'Dest_Airport', 'DepDelay']):
    pivot_route = df.pivot_table(
        index="Org_Airport",
        columns="Dest_Airport",
        values="DepDelay",
        aggfunc="mean"
    )

    plt.figure(figsize=(max(8, 0.5 * len(pivot_route.columns)), max(6, 0.4 * len(pivot_route.index))))
    sns.heatmap(
        pivot_route,
        cmap='coolwarm',
        annot=True,
        fmt=".1f",
        cbar_kws={'label': "Avg Departure Delay (min)"},
        xticklabels=True,
        yticklabels=True
    )
    plt.title('Heatmap of Average Departure Delay by Route (Origin-Destination)', fontsize=14)
    plt.xlabel('Destination Airport', fontsize=12)
    plt.ylabel('Origin Airport', fontsize=12)
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# Heatmap: Average departure delay by origin airport
if "Org_Airport" in df.columns and "DepDelay" in df.columns:
    pivot_origin = df.pivot_table(
        index="Org_Airport",
        values="DepDelay",
        aggfunc="mean"
    ).sort_values("DepDelay", ascending=True)

    plt.figure(figsize=(8, max(6, 0.4 * len(pivot_origin.index))))
    sns.heatmap(
        pivot_origin,
        cmap='coolwarm',  # low intensity to high (yellow to red)
        annot=True,
        fmt=".1f",
        cbar_kws={'label': "Avg Departure Delay (min)"},
        yticklabels=True
    )
    plt.title('Heatmap of Average Departure Delay by Origin Airport', fontsize=14)
    plt.xlabel('Average Departure Delay (min)', fontsize=12)
    plt.ylabel('Origin Airport', fontsize=12)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# Heatmap: Average departure delay by airport (both origin and destination)
if all(col in df.columns for col in ["Org_Airport", "Dest_Airport", "DepDelay"]):
    # Calculate average departure delay for each airport (as origin and as destination)
    origin_delay = df.groupby("Org_Airport")["DepDelay"].mean().rename("AvgDepDelay_Origin")
    dest_delay = df.groupby("Dest_Airport")["DepDelay"].mean().rename("AvgDepDelay_Dest")
    avg_delay = pd.concat([origin_delay, dest_delay], axis=1).fillna(0)
    avg_delay["AvgDepDelay"] = avg_delay.mean(axis=1)
    avg_delay = avg_delay[["AvgDepDelay"]].sort_values("AvgDepDelay", ascending=True)

    plt.figure(figsize=(8, max(6, 0.4 * len(avg_delay.index))))
    sns.heatmap(
        avg_delay,
        cmap='coolwarm',
        annot=True,
        fmt=".1f",
        cbar_kws={'label': "Avg Departure Delay (min)"},
        yticklabels=True
    )
    plt.title('Heatmap of Average Departure Delay by Airport', fontsize=14)
    plt.xlabel('Average Departure Delay (min)', fontsize=12)
    plt.ylabel('Airport', fontsize=12)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# Assign two columns: latitude and longitude in the dataset
df['latitude'] = np.nan
df['longitude'] = np.nan

# COMMAND ----------

# Assign random coordinates to each unique airport and map them to the dataset
unique_airports = pd.Index(df['Org_Airport'].unique()).union(df['Dest_Airport'].unique())
airport_coords = pd.DataFrame({
    'airport': unique_airports,
    'latitude': np.random.uniform(-90, 90, len(unique_airports)),
    'longitude': np.random.uniform(-180, 180, len(unique_airports))
})

airport_coords_dict = airport_coords.set_index('airport')[['latitude', 'longitude']].to_dict(orient='index')
df['latitude'] = df['Org_Airport'].map(lambda x: airport_coords_dict[x]['latitude'])
df['longitude'] = df['Org_Airport'].map(lambda x: airport_coords_dict[x]['longitude'])

# COMMAND ----------

#  Calculate busiest airport (by total flights as origin or destination) and average delay
airport_counts = pd.concat([
    df['Org_Airport'].value_counts(),
    df['Dest_Airport'].value_counts()
], axis=1, keys=['origin_count', 'dest_count']).fillna(0)
airport_counts['total_flights'] = airport_counts['origin_count'] + airport_counts['dest_count']

# Average delay per airport (as origin)
avg_delay = df.groupby('Org_Airport')['DepDelay'].mean().rename('avg_delay')

# Merge with coordinates
airport_stats = airport_counts.join(avg_delay, how='left')
airport_stats = airport_stats.join(airport_coords.set_index('airport'), how='left')
airport_stats = airport_stats.reset_index().rename(columns={'index': 'airport'})

plt.figure(figsize=(12, 8))
sc = plt.scatter(
    airport_stats['longitude'],
    airport_stats['latitude'],
    s=airport_stats['total_flights'] * 0.5,  # scale for visibility
    c=airport_stats['avg_delay'],
    cmap='coolwarm',
    alpha=0.7,
    edgecolor='k'
)
plt.colorbar(sc, label='Average Departure Delay (min)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Busiest Airports and Their Average Departure Delay')
for _, row in airport_stats.nlargest(10, 'total_flights').iterrows():
    plt.text(row['longitude'], row['latitude'], row['airport'], fontsize=9, ha='center', va='bottom')
plt.tight_layout()
plt.show()