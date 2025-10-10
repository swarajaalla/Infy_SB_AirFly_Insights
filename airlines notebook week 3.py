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

# Map airport codes to full names
df['OriginFullName'] = df['Origin'].map(airport_name_map).fillna(df['Origin'])

# Get top 15 airports by flight count (using full names)
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

# Map airport codes to full names
df['OriginFullName'] = df['Origin'].map(airport_name_map).fillna(df['Origin'])

# Count flights for all airports
airport_counts = df['OriginFullName'].value_counts()

plt.figure(
    figsize=(10, max(6, 0.3 * len(airport_counts)))  # Dynamic height
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