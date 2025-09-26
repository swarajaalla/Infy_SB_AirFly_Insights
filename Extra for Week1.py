# Databricks notebook source
import pandas as pd
# 1. Load csv
df = pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay1.csv",na_values=['#N/A'])
display(df)

# COMMAND ----------

# DBTITLE 1,Flights count by DayOfWeek
flight_count_by_day = df['DayOfWeek'].value_counts().sort_index()
display(flight_count_by_day)

# COMMAND ----------

# DBTITLE 1,No. of Flights by airline by DaysofWeek
flight_count_by_airline_day = df.groupby(['UniqueCarrier', 'DayOfWeek']).size().reset_index(name='FlightCount')
display(flight_count_by_airline_day)

# COMMAND ----------

# DBTITLE 1,Flights Count by Airline
flight_count_by_airline = df['Airline'].value_counts()
display(flight_count_by_airline)


# COMMAND ----------

# DBTITLE 1,Average Delay of each airline
avg_delay_by_airline = df.groupby('Airline')[['ArrDelay', 'DepDelay']].mean().reset_index()
display(avg_delay_by_airline)

# COMMAND ----------

# DBTITLE 1,Distance Stats by Airline
# Min, max, avg distance travelled by each airline
distance_stats_by_airline = df.groupby('Airline')['Distance'].agg(['min', 'max', 'mean']).reset_index()
display(distance_stats_by_airline)

# COMMAND ----------

# DBTITLE 1,Top N Longest Flights
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

# DBTITLE 1,Top N Shortest Flights
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