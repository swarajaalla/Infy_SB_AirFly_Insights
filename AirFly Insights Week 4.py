# Databricks notebook source
import pandas as pd
import numpy as np

# Load your cleaned data
df = pd.read_csv("/Volumes/airfly_workspace/default/airfly_insights/flights_cleaned.csv")

# COMMAND ----------

# Group by airline to find avg delays

carrier_stats = df.groupby("UniqueCarrier").agg({
    "ArrDelay": ["mean", "median", "std", "max", "min", "count"],
    "DepDelay": ["mean", "median", "std", "max", "min"]
}).round(2)
display(carrier_stats)

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Delay-Cause Composition (Carrier vs Weather vs NAS vs LateAircraft vs Security)

delay_causes = ["CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay"]
df[delay_causes] = df[delay_causes].fillna(0)

cause_by_carrier = df.groupby("UniqueCarrier")[delay_causes].sum()
cause_by_carrier_pct = cause_by_carrier.div(cause_by_carrier.sum(axis=1), axis=0)

# Stacked absolute
cause_by_carrier.plot(kind="bar", stacked=True, figsize=(12,6))
plt.title("Total Delay Minutes by Cause per Carrier")
plt.ylabel("Total Minutes")
plt.tight_layout()
plt.show()

# Percent stacked
cause_by_carrier_pct.plot(kind="bar", stacked=True, figsize=(12,6), colormap="tab20c")
plt.title("Proportion of Delay Causes by Carrier (%)")
plt.ylabel("Proportion")
plt.tight_layout()
plt.show()

# COMMAND ----------

# Day-of-week mapping
dow_map = {1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat",7:"Sun"}
df["DayOfWeekName"] = df["DayOfWeek"].map(dow_map)

# Delays by day-of-week

dow_delay = df.groupby("DayOfWeekName")["ArrDelay"].mean().reindex(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])

sns.barplot(x=dow_delay.index, y=dow_delay.values)
plt.title("Average Arrival Delay by Day of Week")
plt.ylabel("Avg Delay (min)")
plt.show()

# Delay by hour of departure

def get_hour(x):
    try:
        return int(str(int(x)).zfill(4)[:2])
    except:
        return np.nan

df["DepHour"] = df["DepTime"].apply(get_hour)
hour_delay = df.groupby("DepHour")["ArrDelay"].mean().reset_index()

sns.lineplot(x="DepHour", y="ArrDelay", data=hour_delay, marker="o")
plt.title("Average Arrival Delay by Departure Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Avg Delay (min)")
plt.grid(True)
plt.show()

# COMMAND ----------

# Average delay by origin
origin_delay = df.groupby("Origin")["ArrDelay"].mean().sort_values(ascending=False).head(10)

sns.barplot(x=origin_delay.index, y=origin_delay.values)
plt.title("Top 10 Origins by Avg Arrival Delay")
plt.ylabel("Avg Delay (min)")
plt.xticks(rotation=45)
plt.show()

# Destination
dest_delay = df.groupby("Dest")["ArrDelay"].mean().sort_values(ascending=False).head(10)

sns.barplot(x=dest_delay.index, y=dest_delay.values)
plt.title("Top 10 Destinations by Avg Arrival Delay")
plt.ylabel("Avg Delay (min)")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# Route-Level Analysis

df["Route"] = df["Origin"] + "-" + df["Dest"]

route_delay = (
    df.groupby("Route")
    .agg(flights=("FlightNum","count"), avg_delay=("ArrDelay","mean"))
    .sort_values("flights", ascending=False)
    .head(10)
)

sns.barplot(x="Route", y="avg_delay", data=route_delay)
plt.title("Top 10 Routes by Average Delay")
plt.xticks(rotation=45)
plt.ylabel("Avg Delay (min)")
plt.show()

# COMMAND ----------

# Flight Distance vs AirTime Relationship

sns.scatterplot(x="Distance", y="AirTime", hue="ArrDelay", data=df, palette="coolwarm", alpha=0.6)
plt.title("Flight Distance vs AirTime Colored by Arrival Delay")
plt.xlabel("Distance (miles)")
plt.ylabel("AirTime (minutes)")
plt.show()

# Regression line
sns.lmplot(x="Distance", y="AirTime", data=df, line_kws={"color":"red"})
plt.title("Regression: Distance vs AirTime")
plt.show()

# COMMAND ----------

# Outlier Detection — Boxplots for Delay Distribution

top_carriers = df["UniqueCarrier"].value_counts().head(6).index

sns.boxplot(data=df[df["UniqueCarrier"].isin(top_carriers)], x="UniqueCarrier", y="ArrDelay")
plt.title("Arrival Delay Distribution — Top Carriers")
plt.ylabel("Arrival Delay (min)")
plt.show()

# COMMAND ----------

# Correlation & Heatmap

# Choose numeric columns
num_cols = [
    "ArrDelay","DepDelay","AirTime","Distance","TaxiIn","TaxiOut",
    "CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay"
]

corr = df[num_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap — Flight & Delay Features")
plt.show()

# COMMAND ----------

# Composite metric combining mean delay, cancellation %, and distance efficiency
df["Cancelled"] = df["Cancelled"].replace({"N":0,"Y":1}).astype(int)

carrier_score = df.groupby("UniqueCarrier").agg({
    "ArrDelay":"mean",
    "Cancelled":"mean",
    "Distance":"mean"
}).rename(columns={"ArrDelay":"MeanArrDelay","Cancelled":"CancelRate","Distance":"MeanDistance"})

# Normalize values
carrier_score = (carrier_score - carrier_score.min()) / (carrier_score.max() - carrier_score.min())
carrier_score["PerformanceScore"] = 1 - (0.6*carrier_score["MeanArrDelay"] + 0.3*carrier_score["CancelRate"] - 0.1*carrier_score["MeanDistance"])

carrier_score = carrier_score.sort_values("PerformanceScore", ascending=False)
display(carrier_score.head(10))