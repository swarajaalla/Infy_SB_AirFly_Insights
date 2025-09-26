# Databricks notebook source
# Week 1: Project Initialization and Dataset Setup
## -Define goals, KPIs, and workflow
## -Load CSVs using pandas
## -Explore schema, types, size, and nulls
## -Perform sampling and memory optimizations
 
### Goals
#### -Analyze airline data to uncover operational trends and delay #patterns.
#### -Help improve airline/airport performance through visual insights.
#### -Serve stakeholders: airlines, airports, analysts, travelers.

### KPIs
#### -On-time performance rate
#### -Average Departure/Arrival Delay per Airline, airport and Route (min)
#### -Cancellation Rate %
#### -Delay Causes (Carrier, Weather, NAS, Late Aircraft, Security)
#### -Top Busiest/Most Delayed Routes & Airports
#### -Seasonal/Hourly Trends

### Workflow
#### -Load & Understand Data
#### -Clean Data & Engineer Features (e.g., Route, Season, Delay Flags)
#### -Univariate Analysis – Distributions of key metrics
#### -Bivariate Analysis – Relationships between variables
#### -Delay Cause Analysis – Why delays happen
#### -Route & Airport Analysis – Identify hotspots
#### -Seasonal & Cancellation Trends – Time-based patterns
#### -Build Interactive Dashboard/Report

# COMMAND ----------

import pandas as pd
# 1. Load csv
df = pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay1.csv",na_values=['#N/A'])
display(df)

# COMMAND ----------

# DBTITLE 1,#2.Explore Schema
df.info()

# COMMAND ----------

df.shape

# COMMAND ----------

df.size

# COMMAND ----------

df.columns

# COMMAND ----------

df.dtypes

# COMMAND ----------

df.describe()

# COMMAND ----------

df1 = df[['AirTime', 'Distance', 'ActualElapsedTime', 'CRSElapsedTime', 'DepTime', 'ArrTime']]
df1.describe()

# COMMAND ----------

df.head(10)

# COMMAND ----------

df.tail(10)

# COMMAND ----------

#identifying nulls
print(df.isnull().sum())

# COMMAND ----------

#identifying Duplicates
print(df.duplicated().sum())

# COMMAND ----------

#Perform sampling and memory optimizations
sample_df = df.sample(frac=0.01, random_state=42)
print(sample_df.shape)

# COMMAND ----------

for col in sample_df.select_dtypes(include=['float']):
    sample_df[col]=pd.to_numeric(sample_df[col],downcast='float')
for col in sample_df.select_dtypes(include=['int']):
    sample_df[col]=pd.to_numeric(sample_df[col],downcast='integer')
for col in sample_df.select_dtypes(include=['object']):
    num_unique=sample_df[col].nunique()
    num_total=len(sample_df[col])
    if num_unique/num_total < 0.5:
        sample_df[col]=sample_df[col].astype('category')

# COMMAND ----------

# DBTITLE 1,No. of Flights by airline by DaysofWeek
flight_count_by_airline_day = df.groupby(['UniqueCarrier', 'DayOfWeek']).size().reset_index(name='FlightCount')
display(flight_count_by_airline_day)

# COMMAND ----------

flight_count_by_airline = df['Airline'].value_counts()
display(flight_count_by_airline)

# COMMAND ----------

avg_delay_by_airline = df.groupby('UniqueCarrier')[['ArrDelay', 'DepDelay']].mean().reset_index()
display(avg_delay_by_airline)

# COMMAND ----------

# Min, max, avg distance travelled by each airline
distance_stats_by_airline = df.groupby('Airline')['Distance'].agg(['min', 'max', 'mean'])
display(distance_stats_by_airline)

# COMMAND ----------

N = int(input("Enter N value: "))
# Exclude flights where Origin == Dest
filtered_df = df[df['Origin'] != df['Dest']].copy()

# Remove duplicate routes regardless of direction (e.g., WRG-PSG and PSG-WRG are considered the same)
filtered_df['RouteKey'] = filtered_df.apply(lambda x: '-'.join(sorted([x['Origin'], x['Dest']])), axis=1)
filtered_df = filtered_df.drop_duplicates(subset=['RouteKey'])

# Get top N longest flights
top_n_longest_flights = (
    filtered_df.nlargest(N, 'Distance')
      .loc[:, ['UniqueCarrier', 'FlightNum', 'Origin', 'Dest', 'Distance']]
      .rename(columns={'UniqueCarrier': 'Airline'})
)
display(top_n_longest_flights)

# COMMAND ----------

N = int(input("Enter N value: "))
# Exclude flights where Origin == Dest
filtered_df = df[df['Origin'] != df['Dest']].copy()

# Remove duplicate routes regardless of direction (e.g., WRG-PSG and PSG-WRG are considered the same)
filtered_df['RouteKey'] = filtered_df.apply(lambda x: '-'.join(sorted([x['Origin'], x['Dest']])), axis=1)
filtered_df = filtered_df.drop_duplicates(subset=['RouteKey'])

# Get top N longest flights
top_n_longest_flights = (
    filtered_df.nlargest(N, 'Distance')
      .loc[:, ['UniqueCarrier', 'FlightNum', 'Origin', 'Dest', 'Distance']]
      .rename(columns={'UniqueCarrier': 'Airline'})
)
display(top_n_longest_flights)

# COMMAND ----------

# DBTITLE 1,top N shortest flight
N = int(input("Enter N value: "))
# Exclude flights where Origin == Dest
filtered_df = df[df['Origin'] != df['Dest']].copy()

# Remove duplicate routes regardless of direction (e.g., WRG-PSG and PSG-WRG are considered the same)
filtered_df['RouteKey'] = filtered_df.apply(lambda x: '-'.join(sorted([x['Origin'], x['Dest']])), axis=1)
filtered_df = filtered_df.drop_duplicates(subset=['RouteKey'])

# Get top N shortest flights
top_n_shortest_flights = (
    filtered_df.nsmallest(N, 'Distance')
      .loc[:, ['UniqueCarrier', 'FlightNum', 'Origin', 'Dest', 'Distance']]
      .rename(columns={'UniqueCarrier': 'Airline'})
)
display(top_n_shortest_flights)

# COMMAND ----------

