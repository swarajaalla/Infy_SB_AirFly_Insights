# Databricks notebook source
#WEEK 1:

import pandas as pd

# Load the dataset into Pandas
df = pd.read_csv("/Volumes/workspace/default/flight_delay_data_file/Flight_delay.csv")

# Show basic info
print(df.shape)
print(df.dtypes)
df.head()


# COMMAND ----------

# Schema and data types
print(pdf.info())

# Memory usage
print("Memory usage (MB):", round(pdf.memory_usage(deep=True).sum() / 1024**2, 2))

# Count nulls
print(pdf.isnull().sum())


# COMMAND ----------

# Sampling 1% of data for faster testing
sample_df = df.sample(frac=0.01, random_state=42)

# Optimize memory usage
for col in df.select_dtypes(include='int64').columns:
    df[col] = pd.to_numeric(df[col], downcast='integer')
for col in df.select_dtypes(include='float64').columns:
    df[col] = pd.to_numeric(df[col], downcast='float')

print("Optimized Memory (MB):", round(df.memory_usage(deep=True).sum() / 1024**2, 2))


# COMMAND ----------

#WEEK 2:

# Fill delays with 0 if missing
delay_cols = ["DepDelay", "ArrDelay", "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
for col in delay_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Fill cancellations with 0
if "Cancelled" in df.columns:
    df["Cancelled"] = df["Cancelled"].fillna(0)

print(df.display())


# COMMAND ----------

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Month, Day of Week, Hour
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['DepHour'] = (df['DepTime'] // 100).astype('Int64')

# Route = Origin-Destination
if "Origin" in df.columns and "Dest" in df.columns:
    df['Route'] = df['Origin'] + "_" + df['Dest']


# COMMAND ----------

# Ensure all datetime fields are properly formatted
if "CRSDepTime" in df.columns:
    df['CRSDepTime'] = pd.to_datetime(df['CRSDepTime'], format='%H%M', errors='coerce')
if "CRSArrTime" in df.columns:
    df['CRSArrTime'] = pd.to_datetime(df['CRSArrTime'], format='%H%M', errors='coerce')


# COMMAND ----------

# Save preprocessed dataset for reuse
df.to_csv("/Volumes/workspace/default/flight_delay_data_file/Flight_delay.csv", index=False)

print("✅ Preprocessed dataset saved!")


# COMMAND ----------

# Save cleaned dataframe as CSV
df.to_csv("/Volumes/workspace/default/flight_delay_data_file/Flight_delay.csv", index=False)

# Reload the saved file
df_check = pd.read_csv("/Volumes/workspace/default/flight_delay_data_file/Flight_delay.csv")

print(df_check.shape)
df_check.head()


# COMMAND ----------

#WEEK 3:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("/Volumes/workspace/default/flight_delay_data_file/Flight_delay.csv")

# Display first 5 rows
print(df.shape)
#print(df.display())
df.head()


# COMMAND ----------

#Top 10 Airlines with Most Flights:

plt.figure(figsize=(10,5))
top_airlines = df['Airline'].value_counts().head(10)
sns.barplot(y=top_airlines.index, x=top_airlines.values, palette="coolwarm")
plt.title("Top 10 Airlines with Most Flights")
plt.ylabel("Airline")
plt.xlabel("Number of Flights")
plt.show()


# COMMAND ----------

#Top 10 Busiest Routes:

df['Route'] = df['Org_Airport'] + " → " + df['Dest_Airport']
top_routes = df['Route'].value_counts().head(10)

plt.figure(figsize=(12,6))
sns.barplot(y=top_routes.index, x=top_routes.values, palette="mako")
plt.title("Top 10 Busiest Flight Routes")
plt.xlabel("Number of Flights")
plt.ylabel("Route")
plt.show()


# COMMAND ----------

#Busiest Months (Flight Distribution by Month):

# Extract month from date column
df['Month'] = pd.to_datetime(df['Date']).dt.month

plt.figure(figsize=(10,5))
sns.countplot(x='Month', data=df, palette='viridis')
plt.title("Number of Flights by Month")
plt.xlabel("Month")
plt.ylabel("Flight Count")
plt.show()


# COMMAND ----------

#Flight Distribution by Day of Week:

plt.figure(figsize=(8,5))
sns.countplot(x='DayOfWeek', data=df, palette='cubehelix')
plt.title("Flight Distribution by Day of the Week")
plt.xlabel("Day of Week (0=Mon, 6=Sun)")
plt.ylabel("Number of Flights")
plt.show()


# COMMAND ----------

#Flight Distribution by Time of Day:
def get_time_of_day(dep_time):
    if dep_time < 600:
        return 'Early Morning'
    elif dep_time < 1200:
        return 'Morning'
    elif dep_time < 1800:
        return 'Afternoon'
    else:
        return 'Evening'

df['DepTimePeriod'] = df['DepTime'].apply(get_time_of_day)

plt.figure(figsize=(8,5))
sns.countplot(x='DepTimePeriod', data=df, order=['Early Morning','Morning','Afternoon','Evening'], palette='cool')
plt.title("Flight Distribution by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Number of Flights")
plt.show()


# COMMAND ----------

#Delay Analysis by Airline:
plt.figure(figsize=(10,6))
sns.boxplot(x='UniqueCarrier', y='ArrDelay', data=df, palette='Set3')
plt.title("Arrival Delay by Airline")
plt.xlabel("Airline")
plt.ylabel("Arrival Delay (minutes)")
plt.show()


# COMMAND ----------

#Compare Delay Causes:
delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
df[delay_cols].mean().plot(kind='bar', figsize=(8,5), color='coral')
plt.title("Average Delay Time by Cause")
plt.ylabel("Average Delay (minutes)")
plt.show()


# COMMAND ----------

#Visualize Delay by Time of Day and Airport:
plt.figure(figsize=(12,6))
sns.barplot(x='DepTimePeriod', y='ArrDelay', data=df, ci=None, palette='muted')
plt.title("Average Arrival Delay by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Average Delay (minutes)")
plt.show()

plt.figure(figsize=(12,6))
top_airports = df['Origin'].value_counts().head(10).index
sns.barplot(x='Origin', y='ArrDelay', data=df[df['Origin'].isin(top_airports)], ci=None, palette='crest')
plt.title("Average Delay by Top 10 Airports")
plt.xlabel("Origin Airport")
plt.ylabel("Average Arrival Delay (minutes)")
plt.show()


# COMMAND ----------

#