# Databricks notebook source
import pandas as pd
import numpy as np
df=pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay.csv")



# COMMAND ----------



# COMMAND ----------

print("Schema & Data Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

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


df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce')
print("\n--- After Formatting Date ---")
print(df[['Date']].head())


df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek  # Monday=0, Sunday=6
print("\n--- Derived Features: Month & DayOfWeek ---")
print(df[['Date', 'Month', 'DayOfWeek']].head())


if 'DepTime' in df.columns:
    df['DepTime'] = pd.to_numeric(df['DepTime'], errors='coerce')
    df['Hour'] = (df['DepTime'] // 100).astype('Int64')

print("\n--- Extracted Hour from DepTime ---")
print(df[['DepTime', 'Hour']].head())

# Create Route column (convert categorical to string first)
if 'Origin' in df.columns and 'Dest' in df.columns:
    df['Route'] = df['Origin'].astype(str) + "-" + df['Dest'].astype(str)

print("\n--- Created Route Column ---")
print(df[['Origin', 'Dest', 'Route']].head())