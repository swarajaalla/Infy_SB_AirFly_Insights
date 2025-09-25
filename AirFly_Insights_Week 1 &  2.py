# Databricks notebook source
#Define project goals, KPIs, and workflow for the Flight delay dataset

#Goals
#1. Analyze flight inlay patters and identify hey factors contributing to delays.
#2. Build predictive models to estimate the likelihood and duration of flight delays.
#3. Provide actionable insights to reduce delays and improve operational efficiency

#KPIs:
#1.	-Average delay time per airline, airport, and route
#2.	-Percentage of delayed flights
#3.	-On-time performance rate
#4.	-Delay distribution by time of day, day of week, and season
#5.	-Feature importance in delay prediction models

#workflow
workflow_steps = [
"1. Dets Ingestion Load and inspect the light delay dataset.",
"2. data Cleaning: Mandle missing values, outliers, and optimize memory usage.",
"3. Exploratory Dets analysis: visualize delay patterns and compute KPIs.",
"4. Feature engineering: Create new features (e.g., delay categories, time bins).",
"5. Modeling: Train and evaluate predictive models fur flight delays."
"6. Insights & Recommendations: Summarize findings and suggest improvements."
] 

#Display KPIs and workflow
Kpis = [
"Average delay time per airline, airport, and route",
"Percentage of delayed flights",
"On-time performance rate",
"Delay distribution by time of day, day of week, and season",
"Feature Importance in delay prediction models"
]
display(pd.DataFrame({'KPI' : Kpis}))
display(pd.Dataframe({ 'Workflow step': workflow_steps}))

# COMMAND ----------

import pandas as pd

df_pd = pd.read_csv('/Volumes/workspace/default/airlines/Flight_delay.csv')
display(df_pd)

# COMMAND ----------

df_pd['Date'] = pd.to_datetime(df_pd['Date'])
display(df_pd)

# COMMAND ----------

df_pd['month'] = df_pd['Date'].dt.month
print(df_pd)

# COMMAND ----------

df.head(10) #top 10 rowdf

# COMMAND ----------

df.tail(10) #Last 10 row

# COMMAND ----------

df.columns

# COMMAND ----------

df.size

# COMMAND ----------

df.shape

# COMMAND ----------

df.dtypes

# COMMAND ----------

df.info()

# COMMAND ----------

df_pd.describe()

# COMMAND ----------

# DBTITLE 1,Null
#check nulls
df_pd.isnull().sum()

# COMMAND ----------

# DBTITLE 1,Null Handling
df_pd['Org_Airport'] = df_pd['Org_Airport'].fillna("Unknown")
df_pd['Dest_Airport'] = df_pd['Dest_Airport'].fillna("Unknown")

# COMMAND ----------

# DBTITLE 1,duplicates
print(df_pd.shape)
df_pd = df_pd.drop_duplicates([col for col in ['UniqueCarrier','FlightNum','AirNum']
if col in df_pd.columns])
#absolute duplicate
print(df_pd.shape)

# COMMAND ----------

#Sample 10% of the data 
df_sample = df_pd.sample(frac=0.1, random_state=42)
display(df_sample)


# COMMAND ----------

df.duplicated().sum()

# COMMAND ----------

df_pd.memory_usage(deep=True)

# COMMAND ----------

Org_Airport_column = df['Org_Airport']
print(Org_Airport_column.unique())

# COMMAND ----------

print("Minimum Distance:", df['Distance'].min())

# COMMAND ----------

print("Maximum Distance:", df['Distance'].max())

# COMMAND ----------

print("Average Distance:", df['Distance'].mean())

# COMMAND ----------

#Flight count by DayOfWeek
flights_by_day = df['DayOfWeek'].value_counts().sort_index()
print("\n=== Flights by Day of Week ===")
print(flights_by_day)

# COMMAND ----------

#Cancellation analysis
cancellations = df['Cancelled'].value_counts(normalize=True) * 100
print("\nCancellation Rate (%)")
print(cancellations)

# COMMAND ----------

#FLIGHT COUNTS & BASIC STATS
print(df['Airline'].value_counts())

# COMMAND ----------

print(df['DayOfWeek'].value_counts().sort_index())

# COMMAND ----------

df_pd.to_csv("/Volumes/workspace/default/airlines/Flight_delay_cleaned.csv")
