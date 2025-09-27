# Databricks notebook source
import pandas as pd
import numpy as np
df = pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay.csv") 
df.head()

# COMMAND ----------

print("Shape:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

df.tail(5)

# COMMAND ----------

df.size

# COMMAND ----------

df = df.drop_duplicates()

# COMMAND ----------

df['Date'] = pd.to_datetime(df['Date'])
display(df)

# COMMAND ----------

df['month'] = df['Date'].dt.month
print(df)

# COMMAND ----------

df.columns

# COMMAND ----------

df.describe()

# COMMAND ----------

print("Average Arrival Delay:", df['ArrDelay'].mean())
print("Average Departure Delay:", df['DepDelay'].mean())

# COMMAND ----------

print("Final Shape:", df.shape)
print("Nulls after cleaning:\n", df.isnull().sum())