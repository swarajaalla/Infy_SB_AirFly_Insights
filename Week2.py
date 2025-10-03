# Databricks notebook source
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

import pandas as pd
df = spark.table("adf_databricks_nyc.default.flight_delay").toPandas()
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
print(df["Cancelled"].isnull().sum())
df['Cancelled'] = df['CancellationCode'].fillna('N').apply(lambda x: 0 if x == 'N' else 1)

# COMMAND ----------

# DBTITLE 1,show the nulls in Airports
#Display the rows that have Null Values in Org_Airport and Dest_Airport
print(df["Org_Airport"].isnull().sum())
print(df["Dest_Airport"].isnull().sum())

# COMMAND ----------


df.info()
     

# COMMAND ----------

# Downcast numeric types to save memory
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = pd.to_numeric(df[col], downcast="integer")

# COMMAND ----------

# Sample 10% of data (if dataset is very large)
df_sampled = df.sample(frac=0.1, random_state=42)

# COMMAND ----------

# Handle Nulls in Delay/Cancellation Columns

delay_cols = ["ArrDelay","DepDelay","CarrierDelay","WeatherDelay",
              "NASDelay","SecurityDelay","LateAircraftDelay"]
for col in delay_cols:
    if col in df_sampled.columns:
        df_sampled[col] = df_sampled[col].fillna(0)

cancel_cols = ["Cancelled", "Diverted"]
for col in cancel_cols:
    if col in df_sampled.columns:
        df_sampled[col] = df_sampled[col].fillna(0)


# COMMAND ----------

# Feature Engineering

# Convert Date + Time columns
df_sampled["Date"] = pd.to_datetime(df_sampled["Date"])
df_sampled["Month"] = df_sampled["Date"].dt.month
df_sampled["DayOfWeek"] = df_sampled["Date"].dt.dayofweek + 1  # 1=Mon
df_sampled["DepHour"] = (df_sampled["DepTime"] // 100).astype(int)

# Create Route
df_sampled["Route"] = df_sampled["Origin"] + "-" + df_sampled["Dest"]

# COMMAND ----------

plt.figure(figsize=(8, 5))
sns.countplot(
    x="Month",
    data=df_sampled,
    palette="Blues"
)
plt.title("Flights per Month")
plt.xlabel("Month")
plt.ylabel("Number of Flights")
plt.show()

# COMMAND ----------

# 2. Average Delay by Day of Week
avg_delay = df_sampled.groupby("DayOfWeek")["ArrDelay"].mean().reset_index()
plt.figure(figsize=(8,5))
sns.barplot(x="DayOfWeek", y="ArrDelay", data=avg_delay, palette="Oranges")
plt.title("Average Arrival Delay by Day of Week")
plt.xlabel("Day of Week (1=Mon, 7=Sun)")
plt.ylabel("Avg Arrival Delay (minutes)")
plt.show()

# COMMAND ----------

# 3. Delays by Hour of Day
plt.figure(figsize=(10,6))
sns.boxplot(x="DepHour", y="DepDelay", data=df_sampled, showfliers=False, palette="Greens")
plt.title("Departure Delay Distribution by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Departure Delay (minutes)")
plt.show()

# COMMAND ----------


# 4. Top 10 Routes
top_routes = df_sampled["Route"].value_counts().head(10).reset_index()
top_routes.columns = ["Route", "Count"]
plt.figure(figsize=(10,6))
sns.barplot(x="Count", y="Route", data=top_routes, palette="Purples")
plt.title("Top 10 Busiest Routes")
plt.xlabel("Number of Flights")
plt.ylabel("Route")
plt.show()

# COMMAND ----------


# 5. Delay Reasons
cancel_summary = df_sampled[delay_cols].sum().reset_index()
cancel_summary.columns = ["Reason", "Total Delay (minutes)"]
plt.figure(figsize=(8,5))
sns.barplot(x="Reason", y="Total Delay (minutes)", data=cancel_summary, palette="Reds")
plt.title("Total Delays by Reason")
plt.xticks(rotation=30)
plt.show()