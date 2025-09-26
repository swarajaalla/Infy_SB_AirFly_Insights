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
df['Date'] = pd.to_datetime(df['Date'])
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

# DBTITLE 1,Save preprocessed data for fast reuse
df.to_csv("/Volumes/workspace/default/airlines/Flight_delay_cleaned.csv")