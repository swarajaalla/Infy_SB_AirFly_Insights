# Databricks notebook source
import pandas as pd
raw_path = '/Volumes/airfly_workspace/default/airfly_insights/airfly _raw_data.csv'
df = pd.read_csv(raw_path)
print("Raw shape:", df.shape)


# COMMAND ----------

# Handle Nulls in Delay and Cancellation Columns

import numpy as np

# Fill delay columns with 0
delay_cols = ['ArrDelay', 'DepDelay', 'CarrierDelay', 'WeatherDelay', 
              'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
df[delay_cols] = df[delay_cols].fillna(0)

# Standardize cancellation columns
df['Cancelled'] = df['Cancelled'].map({'Y': 1, 'N': 0})
df['Diverted'] = df['Diverted'].fillna(0)
df['CancellationCode'] = df['CancellationCode'].fillna('None')


# COMMAND ----------

# Format Date and Time Columns

# Use dayfirst=True since format is DD-MM-YYYY
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Convert DepTime, ArrTime, CRSArrTime from HHMM integer to time
def convert_hhmm(time):
    try:
        time = int(time)
        return pd.to_datetime(f'{time:04}', format='%H%M').time()
    except:
        return pd.NaT

for col in ['DepTime', 'ArrTime', 'CRSArrTime']:
    df[col] = df[col].apply(convert_hhmm)

# Quick check
print(df[['Date', 'DepTime', 'ArrTime', 'CRSArrTime']].head())


# COMMAND ----------

# Create Derived Features

# Ensure Date is datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')  # invalid parsing → NaT

# Drop rows where Date could not be parsed 
df = df.dropna(subset=['Date'])

# Now create derived features
df['Month'] = df['Date'].dt.month
df['DayOfWeekNum'] = df['Date'].dt.dayofweek  # Monday=0
df['DepHour'] = df['DepTime'].apply(lambda x: x.hour if pd.notnull(x) else np.nan)
df['Route'] = df['Origin'] + '-' + df['Dest']

# Quick check
print(df[['Date','Month','DayOfWeekNum','DepHour','Route']].head())

# COMMAND ----------

# Remove Duplicates

# Drop duplicate flights (same Date, FlightNum, TailNum, Origin, Dest)
df.drop_duplicates(subset=['Date', 'FlightNum', 'TailNum', 'Origin', 'Dest'], inplace=True)

# COMMAND ----------

# Ensure Numeric Columns Are Correct

numeric_cols = ['ActualElapsedTime','CRSElapsedTime','AirTime','TaxiIn','TaxiOut'] + delay_cols
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# COMMAND ----------

# Quick Exploration Examples

# Average arrival delay by airline
print(df.groupby('Airline')['ArrDelay'].mean().sort_values(ascending=False))

# Cancellation rate by month
print(df.groupby('Month')['Cancelled'].mean())

# Most popular routes
print(df['Route'].value_counts().head(10))

# COMMAND ----------

# Check for any missing values

# Check total missing values per column
print(df.isna().sum())

# COMMAND ----------

# Handle Missing DepTime / ArrTime / DepHour

# Fill missing times with NaT (already converted), or optionally drop these rows if very few
df = df.dropna(subset=['DepTime', 'ArrTime'])

# Update DepHour again after dropping rows
df['DepHour'] = df['DepTime'].apply(lambda x: x.hour if pd.notnull(x) else np.nan)

# COMMAND ----------

# Handle Missing Airport Names

df['Org_Airport'] = df['Org_Airport'].fillna('Unknown')
df['Dest_Airport'] = df['Dest_Airport'].fillna('Unknown')


# COMMAND ----------

# Handle Missing Cancelled

# Fill missing Cancelled values with 0 (not cancelled)
df['Cancelled'] = df['Cancelled'].fillna(0)


# COMMAND ----------

# Verify again

# Check total missing values
print(df.isna().sum())

# Quick overall check
print("Any missing values left?", df.isna().any().any())

# COMMAND ----------

# Save Preprocessed Dataset

# Save as Parquet for fast reuse
df.to_parquet('flights_cleaned.parquet', index=False)

# save as CSV
df.to_csv('flights_cleaned.csv', index=False)

# COMMAND ----------

# Save cleaned file to DBFS FileStore for download

cleaned_path = '/Volumes/airfly_workspace/default/airfly_insights/flights_cleaned.csv'
df.to_csv(cleaned_path, index=False)
print(f"Cleaned file saved to: {cleaned_path}")
