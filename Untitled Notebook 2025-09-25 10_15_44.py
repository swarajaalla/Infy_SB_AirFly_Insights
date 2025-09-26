# Databricks notebook source
/Volumes/workspace/default/air_lines/Flight_delay.csv

# COMMAND ----------

import pandas as pd

df = pd.read_csv('/Volumes/workspace/default/air_lines/Flight_delay.csv')
display(df)

# COMMAND ----------

import pandas as pd

df = pd.read_csv('/Volumes/workspace/default/air_lines/Flight_delay.csv')
display(df)

# COMMAND ----------

sampled_df= df.sample(frac=0.1, random_state=42)
display(sampled_df)

# COMMAND ----------

import pandas as pd

df = pd.read_csv('/Volumes/workspace/default/air_lines/Flight_delay.csv')

delay_cols = [
    "ArrDelay", "DepDelay", "CarrierDelay", "WeatherDelay",
    "NASDelay", "SecurityDelay", "LateAircraftDelay"
]
cancellation_cols = ["Cancelled", "CancellationCode"]

# Fill nulls in delay columns with 0
df[delay_cols] = df[delay_cols].fillna(0)

# Fill nulls in Cancelled with 0 and CancellationCode with empty string
df["Cancelled"] = df["Cancelled"].fillna(0)
df["CancellationCode"] = df["CancellationCode"].fillna("")

display(df)

# COMMAND ----------



# COMMAND ----------

