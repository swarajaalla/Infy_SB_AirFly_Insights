# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

df=pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay_processed.csv")
display(df)

# COMMAND ----------

# Top 10 routes by number of flights
top_routes = df['Route'].value_counts().head(10)

plt.style.use('dark_background')
plt.figure(figsize=(10,6))
sns.barplot(
    x=top_routes.values,
    y=top_routes.index,
    palette="Purples"
)
plt.title('Top 10 Busiest Routes by Flight Count')
plt.xlabel('Number of Flights')
plt.ylabel('Route')
plt.show()

# COMMAND ----------

#Average Total Delay for Top 10 Routes
plt.style.use('dark_background')
route_delays = df.groupby('Route')['TotalDelay'].mean().loc[top_routes.index]
plt.figure(figsize=(10,6))
sns.barplot(x=route_delays.values, y=route_delays.index, palette="magma")
plt.title("Average Total Delay for Top 10 Routes")
plt.xlabel("Average Delay (minutes)")
plt.ylabel("Route")
plt.show()

# COMMAND ----------

#Top 10 Origin-Destination Pairs
df['R'] = df['Org_Airport'] + '-' + df['Dest_Airport']
top_routes = df['R'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_routes.values, y=top_routes.index, palette="magma")
plt.title("Top 10 Origin-Destination Pairs")
plt.xlabel("Number of Flights")
plt.ylabel("Route")
plt.show()

# COMMAND ----------

#Heatmap of Average Delay Between Top Airports
#Selecting Top 20 Airports
top_airports = df['Origin'].value_counts().head(20).index
route_delay = (
    df[df['Origin'].isin(top_airports) & df['Dest'].isin(top_airports)]
    .groupby(['Origin','Dest'])['TotalDelay']
    .mean().unstack().fillna(0)
)

# COMMAND ----------

import plotly.express as px
# Create mappings from airport code to airport name
origin_code_to_name = df.set_index('Origin')['Org_Airport'].to_dict()
dest_code_to_name = df.set_index('Dest')['Dest_Airport'].to_dict()
# Top 10 origin and destination airports by flight count
top10_origin = df['Origin'].value_counts().head(10).index
top10_dest = df['Dest'].value_counts().head(10).index
# Group and compute mean delay for top 10 origin airports
origin_delay = (
    df[df['Origin'].isin(top10_origin)]
    .groupby(['Origin', 'DayOfWeek'])['TotalDelay']
    .mean().unstack()
)
origin_delay.index = origin_delay.index.map(origin_code_to_name)
# Group and compute mean delay for top 10 destination airports
dest_delay = (
    df[df['Dest'].isin(top10_dest)]
    .groupby(['Dest', 'DayOfWeek'])['TotalDelay']
    .mean().unstack()
)
dest_delay.index = dest_delay.index.map(dest_code_to_name)
#Heatmap for Origin Delay
fig1 = px.imshow(
    origin_delay,
    labels=dict(x="Day of Week (1=Mon ... 7=Sun)", y="Origin Airport", color="Avg Total Delay"),
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Average Total Delay by Top 10 Origin Airports and Day of Week"
)
fig1.update_layout(template="plotly_dark")
fig1.update_xaxes(type='category')
fig1.update_yaxes(type='category')
fig1.show()
#Heatmap for Destination Delay
fig2 = px.imshow(
    dest_delay,
    labels=dict(x="Day of Week (1=Mon ... 7=Sun)", y="Destination Airport", color="Avg Total Delay"),
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Average Total Delay by Top 10 Destination Airports and Day of Week"
)
fig2.update_layout(template="plotly_dark")
fig2.update_xaxes(type='category')
fig2.update_yaxes(type='category')
fig2.show()

# COMMAND ----------

#Heatmap of Average Delay by Route (Top 10 Airports)
plt.style.use('dark_background')
top10_airports = df['Origin'].value_counts().head(10).index
route_delay = (
    df[df['Origin'].isin(top10_airports) & df['Dest'].isin(top10_airports)]
    .groupby(['Origin','Dest'])['TotalDelay']
    .mean().unstack()
    .fillna(0)
)
plt.figure(figsize=(12,8))
sns.heatmap(route_delay, cmap="YlOrRd", cbar_kws={'label': 'Avg Total Delay (mins)'})
plt.title("Average Total Delay by Route (Top 10 Airports)")
plt.xlabel("Destination")
plt.ylabel("Origin")
plt.show()

# COMMAND ----------

#Average Delay by Hour for Top Airports
plt.style.use('dark_background')
top10_airports = df['Origin'].value_counts().head(10).index
hour_delay = (
    df[df['Origin'].isin(top10_airports)]
    .groupby(['Origin','DepHour'])['TotalDelay']
    .mean().unstack()
)
plt.figure(figsize=(12,6))
sns.heatmap(hour_delay, cmap="viridis")
plt.title("Average Delay by Hour and Top 10 Airports")
plt.xlabel("Departure Hour")
plt.ylabel("Origin Airport")
plt.show()

# COMMAND ----------

#Top 10 Busiest Airports (Departures)
airport_flights = df['Origin'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=airport_flights.values, y=airport_flights.index, palette="magma")
plt.title("Top 10 Busiest Airports (by Departures)")
plt.xlabel("Number of Flights")
plt.ylabel("Airport")
plt.show()

# COMMAND ----------

#Top 10 Airports by Average Delay
airport_delays = df.groupby('Origin')['TotalDelay'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=airport_delays.values, y=airport_delays.index, palette="magma")
plt.title("Top 10 Airports by Average Delay")
plt.xlabel("Average Total Delay (minutes)")
plt.ylabel("Airport")
plt.show()

# COMMAND ----------

#Causes of Delays
plt.style.use('dark_background')
cause_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
top15_airports = df['Origin'].value_counts().head(15).index
delay_causes = df[df['Origin'].isin(top15_airports)].groupby('Origin')[cause_cols].mean().reset_index()
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']  # 5 distinct colors
delay_causes.set_index('Origin').plot(
    kind='bar',
    stacked=True,
    figsize=(12,6),
    color=colors
)
plt.title("Average Delay Causes by Top 15 Airports")
plt.ylabel("Average Delay (minutes)")
plt.xlabel("Origin Airport")
plt.legend(title="Delay Type")
plt.tight_layout()
plt.show()

# COMMAND ----------

#Delay Distribution for Top 5 Routes
plt.style.use('dark_background')
top_routes_list = df['Route'].value_counts().head(5).index
plt.figure(figsize=(12,6), facecolor='#222222')
ax = sns.boxplot(data=df[df['Route'].isin(top_routes_list)], x='Route', y='TotalDelay', palette="Set2")
plt.title("Delay Distribution for Top 5 Routes", color='white')
plt.xlabel("Route", color='white')
plt.ylabel("Total Delay (minutes)", color='white')
plt.xticks(rotation=45, color='white')
plt.yticks(color='white')
ax.set_facecolor('#222222')
plt.gca().spines['bottom'].set_color('white')
plt.gca().spines['left'].set_color('white')
plt.gca().spines['top'].set_color('white')
plt.gca().spines['right'].set_color('white')
plt.show()

# COMMAND ----------

#Airport Performance (Flights vs Average Delay)
plt.style.use('dark_background')
airport_perf = df.groupby('Origin').agg(
    Flights=('FlightNum', 'count'),
    AvgDelay=('TotalDelay', 'mean')
).reset_index()
plt.figure(figsize=(8,6))
sns.scatterplot(data=airport_perf, x='Flights', y='AvgDelay', s=100)
plt.title("Airport Performance: Volume vs Average Delay")
plt.xlabel("Number of Flights")
plt.ylabel("Average Delay (minutes)")
plt.show()

# COMMAND ----------

# Airport coordinates (IATA -> [Latitude, Longitude])
airport_coords = {
    'ATL': [33.6407, -84.4277],
    'LAX': [33.9416, -118.4085],
    'ORD': [41.9742, -87.9073],
    'DFW': [32.8998, -97.0403],
    'DEN': [39.8561, -104.6737],
    'JFK': [40.6413, -73.7781],
    'SFO': [37.6213, -122.3790],
    'SEA': [47.4502, -122.3088],
    'LAS': [36.0840, -115.1537],
    'MCO': [28.4312, -81.3081],
    'PHX': [33.4350, -112.0000],
    'IAH': [29.9902, -95.3368],
    'MIA': [25.7959, -80.2870],
    'BOS': [42.3656, -71.0096],
    'MSP': [44.8820, -93.2218],
    'DTW': [42.2124, -83.3534],
    'PHL': [39.8744, -75.2424],
    'BWI': [39.1754, -76.6684],
    'SLC': [40.7899, -111.9791],
    'SAN': [32.7338, -117.1933],
    'TPA': [27.9755, -82.5332],
    'HNL': [21.3245, -157.9251],
    'PDX': [45.5898, -122.5951],
    'STL': [38.7487, -90.3700],
    'AUS': [30.2020, -97.6664],
    'BNA': [36.1317, -86.6689],
    'CLT': [35.2140, -80.9431],
    'DCA': [38.8512, -77.0402],
    'MDW': [41.7850, -87.7524],
    'FLL': [26.0726, -80.1527],
    'OAK': [37.7126, -122.2197],
    'SJC': [37.3639, -121.9289],
    'RDU': [35.8776, -78.7875],
    'MCI': [39.2976, -94.7139],
    'MSY': [29.9934, -90.2580],
    'PIT': [40.4914, -80.2329],
    'CLE': [41.4117, -81.8498],
    'IND': [39.7173, -86.2944],
    'JAX': [30.4941, -81.6879],
    'SAT': [29.5337, -98.4698],
    'OMA': [41.3030, -95.8941],
    'OKC': [35.3931, -97.6007],
    'ABQ': [35.0496, -106.6170],
    'ELP': [31.8072, -106.3778],
    'ANC': [61.1743, -149.9983],
    'RNO': [39.4986, -119.7681],
    'ONT': [34.0560, -117.6012],
    'SDF': [38.1744, -85.7360],
    'CMH': [39.9979, -82.8919],
    'BUF': [42.9405, -78.7322],
    'BUR': [34.2007, -118.3587],
    'BHM': [33.5629, -86.7535],
    'CHS': [32.8987, -80.0405],
    'TUS': [32.1161, -110.9410],
    'PBI': [26.6832, -80.0956],
    'RIC': [37.5061, -77.3208],
    'SAV': [32.1276, -81.2021],
    'HOU': [29.6454, -95.2789],
    'BTV': [44.4720, -73.1503],
    'PWM': [43.6462, -70.3093],
    'MHT': [42.9326, -71.4357],
    'BIS': [46.7741, -100.7467],
    'TYS': [35.8110, -83.9940],
    'DAY': [39.9024, -84.2194],
    'FWA': [40.9785, -85.1951],
    'MSN': [43.1399, -89.3375],
    'GRR': [42.8808, -85.5228],
    'ALB': [42.7483, -73.8017],
    'BDL': [41.9389, -72.6832],
    'HPN': [41.0670, -73.7076],
    'MKE': [42.9472, -87.8966],
    'ICT': [37.6499, -97.4331],
    'FAT': [36.7762, -119.7181],
    'GEG': [47.6253, -117.5367],
    'EUG': [44.1233, -123.2186],
    'LBB': [33.6636, -101.8230],
    'LEX': [38.0365, -84.6059],
    'MEM': [35.0424, -89.9767],
    'BTR': [30.5328, -91.1496],
    'SHV': [32.4466, -93.8256],
    'MOB': [30.6914, -88.2428],
    'TLH': [30.3965, -84.3503],
    'DSM': [41.5340, -93.6631],
    'XNA': [36.2819, -94.3068],
    'GSP': [34.8956, -82.2189],
    'CAE': [33.9388, -81.1195],
    'AVL': [35.4362, -82.5418],
    'MYR': [33.6827, -78.9275],
    'SJU': [18.4394, -66.0018],
    'STT': [18.3373, -64.9734],
    'STX': [17.7019, -64.7986],
    'OGG': [20.8986, -156.4305],
    'KOA': [19.7388, -156.0456],
    'LIH': [21.9760, -159.3390],
    'ITO': [19.7203, -155.0480]
}

# COMMAND ----------

#Mapping Airport Coordinates
coords = df['Origin'].map(airport_coords)

# COMMAND ----------

#Seperating Latitude and Longitudes
df[['Latitude', 'Longitude']] = coords.apply(pd.Series)

# COMMAND ----------

#Filling Missing Coordinates
df['Latitude'] = df['Latitude'].fillna(np.random.uniform(25, 49))
df['Longitude'] = df['Longitude'].fillna(np.random.uniform(-124, -67))

# COMMAND ----------

#Maps: Busiest Airports and Average Delays
#Aggregating Airport Statistics
airport_stats = df.groupby('Origin').agg(
    Flights=('FlightNum', 'count'),
    AvgDelay=('TotalDelay', 'mean')
).reset_index()

# COMMAND ----------

#Adding Coordinates to Airport Stats
airport_stats['Latitude'] = airport_stats['Origin'].map(lambda x: airport_coords.get(x, [None, None])[0])
airport_stats['Longitude'] = airport_stats['Origin'].map(lambda x: airport_coords.get(x, [None, None])[1])

# COMMAND ----------

#Map of Busiest Airports and Average Delays
import plotly.express as px
fig = px.scatter_geo(
    airport_stats,
    lat='Latitude',
    lon='Longitude',
    size='Flights',
    color='AvgDelay',
    hover_name='Origin',
    projection='natural earth',
    title='Busiest Airports and Average Delays'
)
fig.update_layout(template='plotly_dark')
fig.show()

# COMMAND ----------

