# Databricks notebook source
import pandas as pd
df = spark.table("adf_databricks_nyc.default.flight_delay").toPandas()
display(df)

# COMMAND ----------

df.head(10)

# COMMAND ----------

df.tail(10)

# COMMAND ----------

df.columns

# COMMAND ----------

df.dtypes

# COMMAND ----------

df.info()

# COMMAND ----------

print(df.shape)
df = df.drop_duplicates()
print(df.shape)

# COMMAND ----------

df.describe()

# COMMAND ----------

print("Minimum Distance:", df['Distance'].min())
print("Maximum Distance:", df['Distance'].max())
print("Average Distance:", df['Distance'].mean())

# COMMAND ----------

delay_cols = ['ArrDelay', 'DepDelay', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
df[delay_cols] = df[delay_cols].fillna(0)
df['CancellationCode'] = df['CancellationCode'].fillna('None')

# COMMAND ----------

df['Cancelled'] = df['CancellationCode'].fillna('N').apply(lambda x: 0 if x == 'N' else 1)

# COMMAND ----------

#Display the rows that have Null Values in Org_Airport and Dest_Airport
print(df["Org_Airport"].isnull().sum())
print(df["Dest_Airport"].isnull().sum())

# COMMAND ----------

#Flight count by Day Week
flights_by_day = df['DayOfWeek'].value_counts().sort_index()
print("\nFlights by Day of Week")
print(flights_by_day)

# COMMAND ----------

#Average arrival and departure delay by airline
avg_delays_by_airline = df.groupby('Airline')[['ArrDelay', 'DepDelay']].mean()
print("\nAverage Delays by Airline")
print(avg_delays_by_airline)

# COMMAND ----------

display(avg_delays_by_airline)

# COMMAND ----------

#Top 10 longest flights (by Distance)
longest_flights = df.nlargest(10, 'Distance')[['Airline', 'FlightNum', 'Origin', 'Dest', 'Distance']]
display(longest_flights)

# COMMAND ----------

#Taxi times statistics
print("\nAverage Taxi In/Out Time")
print(df[['TaxiIn', 'TaxiOut']].mean())

# COMMAND ----------

#Cancellation analysis
cancellations = df['Cancelled'].value_counts(normalize=True) * 100
print("\nCancellation Rate (%)")
print(cancellations)

# COMMAND ----------

#FLIGHT COUNTS & BASIC STATS
print(df['Airline'].value_counts())

# COMMAND ----------

display(df['DayOfWeek'].value_counts().sort_index())

# COMMAND ----------

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Verify conversion
display(df['Date'].head())
display(df['Date'].dtype)

# COMMAND ----------

# df.to_csv("Airfly_insights.csv", index=True) or False