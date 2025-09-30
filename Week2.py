# Databricks notebook source
# DBTITLE 1,Load Dataset
import pandas as pd
df = pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay1.csv",na_values=['#N/A'])
display(df)

# COMMAND ----------

# DBTITLE 1,Duplicates Handling
print(df.shape)
df = df.drop_duplicates()
print(df.shape)

# COMMAND ----------

# DBTITLE 1,Handling Nulls in Delays and CancellationCode
delay_cols = ['ArrDelay', 'DepDelay', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
df[delay_cols] = df[delay_cols].fillna(0)
df['CancellationCode'] = df['CancellationCode'].fillna('None')

# COMMAND ----------

# DBTITLE 1,Fill the Nulls in Cancellation
#Fill Empty cells of Cancelled column with 0 or 1 according to the CancellationCode if N then 0 otherwise 1
df['Cancelled'] = df['CancellationCode'].fillna('N').apply(lambda x: 0 if x == 'N' else 1)

# COMMAND ----------

# DBTITLE 1,show the nulls in Airports
#Display the rows that have Null Values in Org_Airport and Dest_Airport
print(df["Org_Airport"].isnull().sum())
print(df["Dest_Airport"].isnull().sum())

# COMMAND ----------

# DBTITLE 1,create a file for airport's IATA codes and Fill the Missing Values
# Load the airport codes CSV
airport_df = pd.read_csv("/Volumes/workspace/default/airlines/airport_codes_mapping.csv")

# Convert CSV to dictionary for mapping
airport_dict = dict(zip(airport_df["IATA_Code"], airport_df["Airport_Name"]))

# Fill missing values in Org_Airport and Dest_Airport
df["Org_Airport"] = df["Origin"].map(airport_dict)
df["Dest_Airport"] = df["Dest"].map(airport_dict)

# COMMAND ----------

# DBTITLE 1,Show Count of Null Values in Airports
print(df["Org_Airport"].isnull().sum())
print(df["Dest_Airport"].isnull().sum())

# COMMAND ----------

# DBTITLE 1,Format datetime columns
df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)
display(df)

# COMMAND ----------

# DBTITLE 1,Create derived features: Month, Day of Week, Hour, Route
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek + 1  # 1=Monday, 7=Sunday
df['Hour'] = (df['DepTime'] // 100).astype(int)
df['Route'] = df['Origin'] + '-' + df['Dest']
df['Duration'] = df['AirTime'].apply(lambda x: f"{x//60}h {x%60}m" if pd.notna(x) else "0h 0m")

# COMMAND ----------

# DBTITLE 1,Display Data
display(df)

# COMMAND ----------

# DBTITLE 1,Calculating Total Delay
# Calculate total delay as sum of ArrivalDelay and DepartureDelay, store in 'TotalDelay' column
df["TotalDelay"] = df["ArrDelay"].fillna(0) + df["DepDelay"].fillna(0)
display(df.head(10))

# COMMAND ----------

# DBTITLE 1,Handling Nulls in Delays
delay_cols = ['ArrDelay', 'DepDelay', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
df[delay_cols] = df[delay_cols].fillna(0)

# COMMAND ----------

# DBTITLE 1,Checking for Nulls
df.isnull().sum()

# COMMAND ----------

# DBTITLE 1,Save cleaned data for fast reuse
df.to_csv("/Volumes/workspace/default/airlines/Flight_delay_cleaned.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC #Flight Delay Analysis

# COMMAND ----------

# DBTITLE 1,1. Load CSV
#Load Dataset
import pandas as pd
df1=pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay_cleaned.csv")

# COMMAND ----------

# DBTITLE 1,2. Basic Exploration
#Basic Exploration
print("Shape of dataset:", df1.shape)
print("Size of dataset:", df1.size)
print("\n--- Info ---")
print(df1.info())
print("\n--- Columns ---")
print(df1.columns)

# COMMAND ----------

print("\n--- Data Types ---")
print(df1.dtypes)
print("\n--- Describe ---")
print(df1.describe())

# COMMAND ----------

print("\n--- Head ---")
print(df1.head(10))
print("\n--- Tail ---")
print(df1.tail(10))

# COMMAND ----------

# DBTITLE 1,3. Data Quality Check
# 3. Data Quality Checks
print("\n--- Null Values ---")
print(df1.isnull().sum())

print("\n--- Duplicate Rows ---")
print(df1.duplicated().sum())

# COMMAND ----------

# DBTITLE 1,4. Flight Counts
# 4. Flight Counts
#  Flight Count by Day
day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday',5: 'Friday', 6: 'Saturday', 7: 'Sunday'}

flight_count_by_day = df1['DayOfWeek'].value_counts().sort_index()
flight_count_by_day = flight_count_by_day.rename(index=day_names)
print("\n--- Flight Count by Day ---")
display(flight_count_by_day)

# COMMAND ----------

#Flight Count by Airline & Day
flight_count_by_airline_day = df1.groupby(['Airline', 'DayOfWeek']).size().reset_index(name='FlightCount')
print("\n--- Flight Count by Airline & Day ---")
display(flight_count_by_airline_day)

# COMMAND ----------

#Flight count by Airline
flight_count_by_airline = df1['Airline'].value_counts()
print("\n--- Flight Count by Airline ---")
display(flight_count_by_airline)

# COMMAND ----------

# DBTITLE 1,5. Delay Analysis by Airline
#Average Delay by Airline
avg_delay_by_airline = df1.groupby('Airline')[['ArrDelay', 'DepDelay']].mean().round(2).reset_index()
print("\n--- Average Delay by Airline ---")
display(avg_delay_by_airline)

# COMMAND ----------

#Distance Stats by Airline
distance_stats_by_airline = df1.groupby('Airline')['Distance'].agg(['min', 'max', 'mean']).round(2)
print("\n--- Distance Stats by Airline ---")
display(distance_stats_by_airline)

# COMMAND ----------

# DBTITLE 1,6. Time Performance
#On-time Performance by Airline
df1['OnTime'] = (df1['ArrDelay'] <= 0).astype(int)
ontime_stats = df1.groupby('Airline')['OnTime'].mean().round(2).sort_values(ascending=False)
print("\n--- On-time Performance by Airline ---")
display(ontime_stats)

# COMMAND ----------

# DBTITLE 1,7. Route Analysis
#Worst Routes (Highest Avg Arrival Delay)
avg_delay_by_route = df1.groupby(['Origin', 'Dest'])['ArrDelay'].mean().round(2).sort_values(ascending=False).head(10)
print("\n--- Worst Routes (Highest Avg Arrival Delay) ---")
display(avg_delay_by_route)

# COMMAND ----------

#Best Routes (Lowest Avg Arrival Delay)
best_routes = df1.groupby(['Origin', 'Dest'])['ArrDelay'].mean().round(2).sort_values().head(10)
print("\n--- Best Routes (Lowest Avg Arrival Delay) ---")
display(best_routes)

# COMMAND ----------

# DBTITLE 1,8. Delay Correlation
#Correlation between DepDelay and ArrDelay
dep_arr_delay_corr = round(df['DepDelay'].corr(df['ArrDelay']), 2)
print("\nCorrelation between DepDelay and ArrDelay:", dep_arr_delay_corr)

# COMMAND ----------

# DBTITLE 1,9. Airline Comparisons
#Delay Variability (Std Dev) by Airline
delay_variability = df1.groupby('Airline')['ArrDelay'].std().round(2).sort_values()
print("\n--- Delay Variability (Std Dev) by Airline ---")
display(delay_variability)

# COMMAND ----------

#Average Distance per Flight by Airline
avg_distance_by_airline = df1.groupby('Airline')['Distance'].mean().round(2).sort_values(ascending=False)
print("\n--- Avg Distance per Flight by Airline ---")
display(avg_distance_by_airline)

# COMMAND ----------

# DBTITLE 1,10. Airport-Level Analysis
# Busiest Airports
busiest_airports = df1['Origin'].value_counts()
print("\n--- Busiest Airports (Origin) ---")
display(busiest_airports)

# COMMAND ----------

#Airports with Highest Avg Departure Delay
avg_dep_delay_airport = df1.groupby('Origin')['DepDelay'].mean().round(2).sort_values(ascending=False)
print("\n--- Airports with Highest Avg Departure Delay ---")
display(avg_dep_delay_airport)

# COMMAND ----------

#Airports with Highest Avg Arrival Delay
avg_arr_delay_airport = df1.groupby('Dest')['ArrDelay'].mean().round(2).sort_values(ascending=False)
print("\n--- Airports with Highest Avg Arrival Delay ---")
display(avg_arr_delay_airport)

# COMMAND ----------

# DBTITLE 1,11. Time-Based Analysis
# 11. Time-Based Analysis
#Average Delay by Day of Week
avg_delay_by_day = df1.groupby('DayOfWeek')[['DepDelay','ArrDelay']].mean().round(2)
print("\n--- Avg Delay by Day of Week ---")
display(avg_delay_by_day)

# COMMAND ----------

# Extract hour from DepTime (HHMM format -> hour)
df1['DepHour'] = (df1['DepTime'] // 100).astype(int)
avg_delay_by_hour = df1.groupby('DepHour')[['DepDelay','ArrDelay']].mean().round(2)
print("\n--- Avg Delay by Hour of Day ---")
display(avg_delay_by_hour)

# COMMAND ----------

# Monthly trends
avg_delay_by_month = df1.groupby('Month')[['DepDelay','ArrDelay']].mean().round(2)
print("\n--- Avg Delay by Month ---")
display(avg_delay_by_month)

# COMMAND ----------

# DBTITLE 1,12. Flight Duration Analysis
#Show Top N longest Flights.
N = int(input("Enter N value: "))
# Exclude flights where Origin == Dest
filtered_df = df1[df1['Origin'] != df1['Dest']].copy()

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

#show Top N shortest Flights.
N = int(input("Enter N value: "))
# Exclude flights where Origin == Dest
filtered_df = df1[df1['Origin'] != df1['Dest']].copy()

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

# DBTITLE 1,13. Cancellations & Diversions
# 13. Cancellations & Diversions
# Cancellation & Diversion Rates
cancellation_rate = df1['Cancelled'].mean().round(2)
print("\nCancellation rate:", cancellation_rate)

diversion_rate = df1['Diverted'].mean().round(2)
print("Diversion rate:", diversion_rate)

# COMMAND ----------

# DBTITLE 1,14. Delay Causes
# 14. Delay Causes
#Average Contribution of Delay Causes
delay_cols = ['CarrierDelay','WeatherDelay','NASDelay','SecurityDelay','LateAircraftDelay']
available_delay_cols = [col for col in delay_cols if col in df1.columns]

delay_causes = df1[available_delay_cols].mean().round(2)
print("\n--- Average Contribution of Delay Causes ---")
display(delay_causes)

# COMMAND ----------

df1.isnull().sum()

# COMMAND ----------

display(df1)

# COMMAND ----------

df1=df1.drop(df1.columns[0],axis=1)

# COMMAND ----------

print(df1.shape)
print(df1.size)

# COMMAND ----------

df1.to_csv("/Volumes/workspace/default/airlines/Flight_delay_processed.csv")
