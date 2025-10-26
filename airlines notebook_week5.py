# Databricks notebook source
import pandas as pd
import numpy as np
df=pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay.csv")



# COMMAND ----------

print("Schema & Data Types:")
print(df.dtypes)


# COMMAND ----------

print("\nMissing Values:")
print(df.isnull().sum())

# COMMAND ----------

print("\nShape (rows, cols):", df.shape)

print("\nSample Records:")
print(df.head())

# COMMAND ----------

print("\nRandom Sample (5 rows):")
print(df.sample(5, random_state=42))   # random sampling

print("\nFractional Sample (10% of data):")
df_sample = df.sample(frac=0.1, random_state=42)
print(df_sample.shape)

# COMMAND ----------

def optimize_dataframe(df):
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type == "object":
            # convert to category if unique values are relatively low
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype("category")
                
        elif np.issubdtype(col_type, np.integer):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        
        elif np.issubdtype(col_type, np.floating):
            df[col] = pd.to_numeric(df[col], downcast="float")
    
    return df

print("\nMemory Usage Before Optimization:")
print(df.memory_usage(deep=True).sum() / 1024**2, "MB")

df_optimized = optimize_dataframe(df)

print("\nMemory Usage After Optimization:")
print(df_optimized.memory_usage(deep=True).sum() / 1024**2, "MB")

print("\nOptimized Dtypes:")
print(df_optimized.dtypes)

# COMMAND ----------

df.display()

# COMMAND ----------

#WEEK 2
delay_cols = ["ArrDelay", "DepDelay", "CarrierDelay", "WeatherDelay", 
              "NASDelay", "SecurityDelay", "LateAircraftDelay", "Cancelled"]

for col in delay_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

print("\n--- After Handling Nulls in Delay & Cancellation Columns ---")
print(df[delay_cols].head())


# COMMAND ----------

# Convert Date column to proper datetime format (DD-MM-YYYY)
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce')
print("\n--- After Formatting Date ---")
print(df[['Date']].head())

# COMMAND ----------

# Extract month and day of week from Date column
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
print("\n--- Derived Features: Month & DayOfWeek ---")
print(df[['Date', 'Month', 'DayOfWeek']].head())


# COMMAND ----------

# Convert DepTime to numeric and extract the hour (e.g., 1829 -> 18)
if 'DepTime' in df.columns:
    df['DepTime'] = pd.to_numeric(df['DepTime'], errors='coerce')
    df['Hour'] = (df['DepTime'] // 100).astype('Int64')

print("\n--- Extracted Hour from DepTime ---")
print(df[['DepTime', 'Hour']].head())

# COMMAND ----------

# Create Route column (convert categorical to string first)
if 'Origin' in df.columns and 'Dest' in df.columns:
    df['Route'] = df['Origin'].astype(str) + "-" + df['Dest'].astype(str)

print("\n--- Created Route Column ---")
print(df[['Origin', 'Dest', 'Route']].head())

# COMMAND ----------

df.to_csv("flights_preprocessed.csv", index=False)
print("\n✅ Preprocessed data saved to 'flights_preprocessed.csv'")

# COMMAND ----------

import matplotlib.pyplot as plt 
import seaborn as sns

# Top airlines by number of flights

plt.figure(figsize =(10,5))

sns.countplot(

data=df,

x='Airline',

order=df[ 'Airline'].value_counts().index
)
plt.title('Top Airlines by Number of Flights') 
plt.xticks(rotation=45) 
plt.show()

# COMMAND ----------

# Convert categorical columns to strings before combining
df['ROUTE'] = df['Org_Airport'].astype(str) + " → " + df['Dest_Airport'].astype(str)

# Count and plot the top 10 routes
top_routes = df['ROUTE'].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_routes.values, y=top_routes.index)
plt.title("Top 10 Routes by Number of Flights")
plt.xlabel("Number of Flights")
plt.ylabel("Route")
plt.show()


# COMMAND ----------

# Delay distribution (Histogram and Boxplot)

if 'DepDelay' in df.columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(df['DepDelay'].dropna(), bins=50, kde=True)
    plt.title('Departure Delay Distribution')
    plt.xlabel('Departure Delay (minutes)')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(8, 2))
    sns.boxplot(x=df['DepDelay'].dropna())
    plt.title('Departure Delay Boxplot')
    plt.xlabel('Departure Delay (minutes)')
    plt.show()
else:
    print("⚠️ Column 'DepDelay' not found in DataFrame.")


# COMMAND ----------

plt.figure(figsize=(6, 3))
sns.countplot(data=df, x='Month', order=sorted(df['Month'].dropna().unique()))
plt.title('Flight Count by Month')
plt.xlabel('Month')
plt.ylabel('Number of Flights')
plt.show()


# COMMAND ----------

plt.figure(figsize=(10,5))
sns.countplot(data=df, x='Airline', order=df['Airline'].value_counts().index)
plt.title('Number of Flights by Airline')
plt.xlabel('Airline')
plt.ylabel('Number of Flights')
plt.show()


# COMMAND ----------



# COMMAND ----------

plt.figure(figsize=(8,4))
sns.countplot(data=df, x='DayOfWeek', order=sorted(df['DayOfWeek'].dropna().unique()))
plt.title('Flight Count by Day of Week')
plt.xlabel('Day of Week (1=Mon, 7=Sun)')
plt.ylabel('Number of Flights')
plt.show()


# COMMAND ----------

avg_delay = df.groupby('Airline')['DepDelay'].mean().sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=avg_delay.values, y=avg_delay.index)
plt.title('Average Departure Delay by Airline')
plt.xlabel('Average Delay (minutes)')
plt.ylabel('Airline')
plt.show()


# COMMAND ----------

plt.figure(figsize=(10,4))
sns.histplot(df['ArrDelay'].dropna(), bins=50, kde=True)
plt.title('Arrival Delay Distribution')
plt.xlabel('Arrival Delay (minutes)')
plt.show()

plt.figure(figsize=(8,2))
sns.boxplot(x=df['ArrDelay'].dropna())
plt.title('Arrival Delay Boxplot')
plt.show()


# COMMAND ----------

plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Distance', y='DepDelay')
plt.title('Departure Delay vs Flight Distance')
plt.xlabel('Distance (miles)')
plt.ylabel('Departure Delay (minutes)')
plt.show()


# COMMAND ----------

monthly_delay = df.groupby('Month')['DepDelay'].mean()
plt.figure(figsize=(8,4))
sns.lineplot(x=monthly_delay.index, y=monthly_delay.values, marker='o')
plt.title('Average Departure Delay by Month')
plt.xlabel('Month')
plt.ylabel('Average Delay (minutes)')
plt.show()


# COMMAND ----------

sns.pairplot(df[['DepDelay', 'ArrDelay', 'Distance', 'CRSElapsedTime']].dropna())
plt.show()


# COMMAND ----------

plt.figure(figsize=(8,4))
sns.countplot(data=df, x='Month', order=sorted(df['Month'].dropna().unique()))
plt.title('Flight Count by Month (1–12)')
plt.xlabel('Month')
plt.ylabel('Number of Flights')
plt.show()


# COMMAND ----------

df['DepHour'] = (df['DepTime'] // 100).astype('Int64')

plt.figure(figsize=(8,4))
sns.countplot(x='DepHour', data=df, palette='crest')
plt.title('Flight Distribution by Departure Time (Hour of Day)')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Flights')
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# COMMAND ----------

# Example mapping dictionary (replace with your actual airport codes and names)
airport_name_map = {
    'ATL': 'Hartsfield–Jackson Atlanta International',
    'ORD': 'Chicago O\'Hare International',
    'DFW': 'Dallas/Fort Worth International',
    'DEN': 'Denver International',
    'LAX': 'Los Angeles International',
    'JFK': 'John F. Kennedy International',
    'SFO': 'San Francisco International',
    'SEA': 'Seattle–Tacoma International',
    'LAS': 'McCarran International',
    'MCO': 'Orlando International',
    'CLT': 'Charlotte Douglas International',
    'PHX': 'Phoenix Sky Harbor International',
    'MIA': 'Miami International',
    'IAH': 'George Bush Intercontinental',
    'EWR': 'Newark Liberty International'
}

df['OriginFullName'] = df['Origin'].map(airport_name_map).fillna(df['Origin'])

top_airports = (
    df['OriginFullName']
    .value_counts()
    .head(15)
)

plt.figure(figsize=(10, 8))
sns.barplot(
    y=top_airports.index,
    x=top_airports.values,
    palette='magma'
)
plt.title('Top 15 Busiest Airports by Number of Flights')
plt.ylabel('Airport')
plt.xlabel('Number of Flights')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# COMMAND ----------


df['OriginFullName'] = df['Origin'].map(airport_name_map).fillna(df['Origin'])

airport_counts = df['OriginFullName'].value_counts()

plt.figure(
    figsize=(10, max(6, 0.3 * len(airport_counts)))  
)
sns.barplot(
    y=airport_counts.index,
    x=airport_counts.values,
    palette='magma'
)
plt.title('Flight Distribution by Airport')
plt.ylabel('Airport')
plt.xlabel('Number of Flights')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay']

available_delay_cols = [col for col in delay_cols if col in df.columns and 'Airline' in df.columns]

mean_delays_by_airline = df.groupby('Airline')[available_delay_cols].mean()

mean_delays_by_airline = mean_delays_by_airline.loc[df['Airline'].value_counts().index]

mean_delays_by_airline.plot(kind='bar', figsize=(14, 6))

# Customize plot
plt.title('Mean Delay by Cause and Airline')
plt.xlabel('Airline')
plt.ylabel('Mean Delay (minutes)')
plt.xticks(rotation=45)
plt.legend(title='Delay Cause')
plt.tight_layout()

plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay']

available_delay_cols = [col for col in delay_cols if col in df.columns]

if available_delay_cols:
    # Histograms for each delay type
    df[available_delay_cols].hist(bins=40, figsize=(12, 4), layout=(1, len(available_delay_cols)))
    plt.suptitle('Distribution of Delay Types')
    plt.tight_layout()
    plt.show()

    # Boxplots for each delay type
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df[available_delay_cols])
    plt.title('Boxplot of Delay Types')
    plt.ylabel('Delay (minutes)')
    plt.tight_layout()
    plt.show()

    # Mean delay per type
    mean_delays = df[available_delay_cols].mean()
    plt.figure(figsize=(8, 4))
    sns.barplot(x=mean_delays.index, y=mean_delays.values)
    plt.title('Mean Delay by Type')
    plt.ylabel('Mean Delay (minutes)')
    plt.tight_layout()
    plt.show()
else:
    print("No delay columns found in the dataset.")


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

if all(col in df.columns for col in ['DepTime', 'Org_Airport', 'DepDelay']):
    
    df['DEP_HOUR'] = df['DepTime'] // 100

    top_airports = df['Org_Airport'].value_counts().head(5).index

    df_top = df[df['Org_Airport'].isin(top_airports)].copy()

    grouped = df_top.groupby(['DEP_HOUR', 'Org_Airport'])['DepDelay'].mean().reset_index()

   
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=grouped,
        x='DEP_HOUR',
        y='DepDelay',
        hue='Org_Airport',
        marker='o'
    )

    plt.title('Average Departure Delay by Hour and Top 5 Origin Airports')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Departure Delay (minutes)')
    plt.xticks(range(0, 24))

    plt.legend(
        title='Origin Airport',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,            
        frameon=False      
    )

    plt.tight_layout()
    plt.show()

else:
    print("Required columns not found in the dataset.")


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Heatmap: average delay by hour and airport
if all(col in df_top.columns for col in ['Org_Airport', 'DEP_HOUR', 'DepDelay']):
    
    pivot = df_top.pivot_table(
        index='Org_Airport',
        columns='DEP_HOUR',
        values='DepDelay',
        aggfunc='mean'
    )

    # Plot heatmap
    plt.figure(figsize=(14, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        cbar_kws={"label": "Avg Departure Delay (min)"}
    )

    plt.title('Heatmap of Average Departure Delay by Hour and Airport')
    plt.xlabel('Hour of Day')
    plt.ylabel('Origin Airport')
    plt.tight_layout()
    plt.show()

else:
    print("Required columns not found in df_top.")


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
sns.boxplot(data=df_top, x='DEP_HOUR', y='DepDelay', palette='coolwarm')
plt.title('Variation in Departure Delay by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Departure Delay (minutes)')
plt.tight_layout()
plt.show()


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

if all(col in df.columns for col in ['WeatherDelay', 'CarrierDelay']):
    plt.figure(figsize=(6, 5))
    sns.regplot(
        data=df,
        x='WeatherDelay',
        y='CarrierDelay',
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red'}
    )
    plt.title('Relationship Between Weather and Carrier Delays')
    plt.xlabel('Weather Delay (minutes)')
    plt.ylabel('Carrier Delay (minutes)')
    plt.tight_layout()
    plt.show()
else:
    print("Required columns for correlation plot not found in the dataset.")


# COMMAND ----------


delay_by_airline = df_top.groupby('UniqueCarrier')[['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']].mean()

plt.figure(figsize=(12,6))
delay_by_airline.plot(kind='bar', stacked=True, figsize=(12,6), colormap='coolwarm')

plt.title("Average Delay Type by Airline")
plt.xlabel("Airline")
plt.ylabel("Average Delay (minutes)")
plt.legend(title="Delay Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# COMMAND ----------


df_top['TotalDelay'] = df_top[['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']].sum(axis=1)

avg_delay_by_hour = df_top.groupby('DEP_HOUR')['TotalDelay'].mean().reset_index()

plt.figure(figsize=(10,5))
sns.lineplot(data=avg_delay_by_hour, x='DEP_HOUR', y='TotalDelay', marker='o', color='teal')

plt.title("Average Total Delay by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Total Delay (minutes)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

corr = df_top[delay_cols].corr()

plt.figure(figsize=(7,5))
sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)

plt.title("Correlation Between Different Delay Causes")
plt.tight_layout()
plt.show()


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Check if required columns exist
required_cols = ['Org_Airport', 'Dest_Airport']
if all(col in df.columns for col in required_cols):

    # Group by origin and destination to count number of flights
    top_routes = (
        df.groupby(['Org_Airport', 'Dest_Airport'])
          .size()
          .reset_index(name='Flight_Count')
    )

    # Sort and take top 10
    top_routes = top_routes.sort_values('Flight_Count', ascending=False).head(10)

    display(top_routes)

    # Optional: Plot Top Routes
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Flight_Count', y=top_routes.index, data=top_routes)
    plt.title('Top 10 Origin-Destination Flight Routes')
    plt.xlabel('Number of Flights')
    plt.ylabel('Route')
    plt.show()

else:
    print("❌ Required columns not found in DataFrame.")
    print("Columns required:", required_cols)
    print("Columns available:", df.columns.tolist())


# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Check if required columns exist
if all(col in df.columns for col in ['Org_Airport', 'Dest_Airport', 'DepDelay']):

    # ✅ Pivot table to compute average delay per route
    pivot_route = df.pivot_table(
        index='Org_Airport',
        columns='Dest_Airport',
        values='DepDelay',
        aggfunc='mean'
    )

    plt.figure(
        figsize=(
            max(8, 0.5 * len(pivot_route.columns)),
            max(6, 0.4 * len(pivot_route.index))
        )
    )

    sns.heatmap(
        pivot_route,
        cmap='coolwarm',
        cbar_kws={'label': 'Avg Departure Delay (min)'},
        xticklabels=True,
        yticklabels=True
    )

    plt.title("Heatmap of Average Departure Delay by Route (Origin → Destination)", fontsize=14)
    plt.xlabel("Destination Airport", fontsize=12)
    plt.ylabel("Origin Airport", fontsize=12)
    plt.xticks(fontsize=10, rotation=45, ha='right')
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

else:
    print("❌ Required columns missing for heatmap!")


# COMMAND ----------

# ✅ Check if required columns exist
if all(col in df.columns for col in ['Org_Airport', 'DepDelay']):

    pivot_airport = df.groupby('Org_Airport')['DepDelay'].mean().sort_values(ascending=False)

    plt.figure(figsize=(10, max(4, 0.4 * len(pivot_airport))))

    sns.barplot(
        x=pivot_airport.values,
        y=pivot_airport.index
    )

    plt.title("Average Departure Delay by Origin Airport", fontsize=14)
    plt.xlabel("Avg Departure Delay (minutes)", fontsize=12)
    plt.ylabel("Origin Airport", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

else:
    print("❌ Required columns missing for origin delay chart!")


# COMMAND ----------

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ✅ Assign random coordinates for visualization (USA bounds for demo)
unique_airports = df['Org_Airport'].unique()
coords = {
    a: (
        np.random.uniform(25, 49),   # Latitude range in USA
        np.random.uniform(-125, -67) # Longitude range in USA
    )
    for a in unique_airports
}

# ✅ Map coordinates to original DataFrame
df["Latitude"] = df["Org_Airport"].map(lambda x: coords[x][0])
df["Longitude"] = df["Org_Airport"].map(lambda x: coords[x][1])

# ✅ Compute flight count per origin airport
airport_stats = df.groupby('Org_Airport').size().reset_index(name='Flight_Count')

# ✅ Add coordinates into airport stats
airport_stats['Latitude'] = airport_stats['Org_Airport'].map(lambda x: coords[x][0])
airport_stats['Longitude'] = airport_stats['Org_Airport'].map(lambda x: coords[x][1])

# ✅ Convert necessary columns to string to avoid TypeError
airport_stats["Org_Airport"] = airport_stats["Org_Airport"].astype(str)
airport_stats["Flight_Count"] = airport_stats["Flight_Count"].astype(int)

# ✅ Make interactive map
fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lon = airport_stats['Longitude'],
    lat = airport_stats['Latitude'],
    text = (
        airport_stats['Org_Airport'] +
        "<br>Flights: " + airport_stats['Flight_Count'].astype(str)
    ),
    mode = 'markers',
    marker=dict(
        size=(airport_stats['Flight_Count'] / airport_stats['Flight_Count'].max()) * 30,
        color='red',
        opacity=0.75,
        line=dict(width=0.6, color='white')
    )
))

# ✅ Styling map layout
fig.update_layout(
    title="Airport Visualization (Randomized Coordinates for Demo)",
    geo=dict(
        scope="usa",  # Change to "world" if global data
        landcolor="rgb(235, 235, 235)",
        coastlinecolor="black",
        projection_scale=2.5,
        lataxis=dict(range=[20, 50]),
        lonaxis=dict(range=[-130, -65])
    ),
    width=1100,
    height=750
)

fig.show()


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

if all(col in df.columns for col in ['Airline', 'DepDelay']):
    airline_delay = df.groupby('Airline')['DepDelay'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, max(4, 0.4 * len(airline_delay))))
    sns.barplot(x=airline_delay.values, y=airline_delay.index)
    plt.title("Worst Airlines by Average Departure Delay", fontsize=14)
    plt.xlabel("Avg Departure Delay (min)")
    plt.ylabel("Airline")
    plt.tight_layout()
    plt.show()
else:
    print("❌ Missing columns for Airline delay chart")


# COMMAND ----------

delay_columns = ['WeatherDelay', 'CarrierDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

if all(col in df.columns for col in ['Org_Airport'] + delay_columns):
    cause_stats = df.groupby('Org_Airport')[delay_columns].mean()

    cause_stats.plot(
        kind='bar',
        stacked=True,
        figsize=(15, max(6, 0.4 * len(cause_stats))),
        alpha=0.85
    )

    plt.title("Delay Cause Contribution by Origin Airport", fontsize=14)
    plt.xlabel("Origin Airport")
    plt.ylabel("Avg Delay (min)")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Delay Type")
    plt.tight_layout()
    plt.show()
else:
    print("❌ Missing delay cause columns for stacked bar chart")


# COMMAND ----------

# Create hour column if you only have timestamps
if 'DepHour' not in df.columns and 'DepTime' in df.columns:
    df['DepHour'] = df['DepTime'] // 100  # e.g., 1340 → 13

if all(col in df.columns for col in ['DepHour', 'DepDelay']):
    hourly_delay = df.groupby('DepHour')['DepDelay'].mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=hourly_delay.index, y=hourly_delay.values, marker="o")
    plt.title("Average Departure Delay by Time of Day", fontsize=14)
    plt.xlabel("Departure Hour (0-23)")
    plt.ylabel("Avg Delay (min)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
else:
    print("❌ Missing DepHour or DepDelay for time-of-day delay chart")


# COMMAND ----------

# Extract month from date if needed
if 'Month' not in df.columns and 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month

if all(col in df.columns for col in ['Month', 'DepDelay']):
    monthly_delay = df.groupby('Month')['DepDelay'].mean()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=monthly_delay.index, y=monthly_delay.values, marker="o")
    plt.title("Monthly Average Departure Delay", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel("Avg Delay (min)")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
else:
    print("❌ Missing Month or DepDelay for monthly delay chart")
