# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

df=pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay_processed.csv")
display(df)

# COMMAND ----------

df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# COMMAND ----------

# MAGIC %md
# MAGIC #Univariate analysis

# COMMAND ----------

# --- Top Airlines ---
plt.figure(figsize=(10,5))
top_airlines = df['Airline'].value_counts().head(10)
sns.barplot(x=top_airlines.index, y=top_airlines.values)
plt.title("Top 10 Airlines by Flight Count")
plt.xlabel("Airline")
plt.ylabel("Number of Flights")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# --- Top Routes (Directionless) ---
plt.figure(figsize=(12,5))
# Create a directionless route by sorting airport codes in each route
df['Route_Undirected'] = df['Route'].apply(lambda x: '-'.join(sorted(x.split('-'))))
top_routes = df['Route_Undirected'].value_counts().head(10)
sns.barplot(x=top_routes.index, y=top_routes.values)
plt.title("Top 10 Routes by Flight Count (Directionless)")
plt.xlabel("Route")
plt.ylabel("Number of Flights")
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

# --- Busiest Months ---
plt.figure(figsize=(10,5))
month_counts = df['Month'].value_counts().sort_index()
sns.lineplot(x=month_counts.index, y=month_counts.values, marker="o")
plt.title("Busiest Months by Number of Flights")
plt.xlabel("Month")
plt.ylabel("Number of Flights")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #Flight Distribution

# COMMAND ----------

# --- Distribution by Day of Week ---
plt.figure(figsize=(8,5))
sns.countplot(x='DayOfWeek', data=df, order=sorted(df['DayOfWeek'].unique()))
plt.title("Flight Distribution by Day of Week (1=Mon, 7=Sun)")
plt.xlabel("Day of Week")
plt.ylabel("Number of Flights")
plt.show()

# COMMAND ----------

# --- Distribution by Departure Hour ---
plt.figure(figsize=(10,5))
sns.histplot(df['DepHour'], bins=24, kde=True)
plt.title("Flight Distribution by Departure Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Flights")
plt.show()

# COMMAND ----------

# --- Distribution by Origin Airport ---
plt.figure(figsize=(12,5))
top_airports = df['Org_Airport'].value_counts().head(10)
sns.barplot(x=top_airports.index, y=top_airports.values)
plt.title("Top 10 Origin Airports")
plt.xlabel("Airport")
plt.ylabel("Number of Departures")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #Bivariate Analysis

# COMMAND ----------

# --- Airline vs Average Arrival Delay ---
plt.figure(figsize=(12,6))
avg_delay = df.groupby('Airline')['ArrDelay'].mean().sort_values(ascending=False)
sns.barplot(x=avg_delay.index, y=avg_delay.values)
plt.title("Average Arrival Delay by Airline")
plt.xlabel("Airline")
plt.ylabel("Average Delay (min)")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# --- Month vs Average Total Delay ---
plt.figure(figsize=(10,5))
sns.boxplot(x='Month', y='TotalDelay', data=df)
plt.title("Total Delay Distribution by Month")
plt.xlabel("Month")
plt.ylabel("Total Delay (min)")
plt.show()

# COMMAND ----------

# --- Distance vs AirTime ---
plt.figure(figsize=(8,5))
sns.scatterplot(x='Distance', y='AirTime', data=df, alpha=0.5)
plt.title("Relationship between Distance and Air Time")
plt.xlabel("Distance (miles)")
plt.ylabel("AirTime (minutes)")
plt.show()

# COMMAND ----------

# --- Departure Hour vs Average Delay ---
plt.figure(figsize=(10,5))
hourly_delay = df.groupby('DepHour')['TotalDelay'].mean()
sns.lineplot(x=hourly_delay.index, y=hourly_delay.values, marker='o')
plt.title("Average Total Delay by Departure Hour")
plt.xlabel("Departure Hour")
plt.ylabel("Average Delay (min)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #Extra Univariate and Bivariate analysis

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Flight Duration, Distance, and Time Distributions

# COMMAND ----------

# --- Flight Duration Distribution ---
plt.figure(figsize=(8,5))
sns.histplot(df['ActualElapsedTime'], bins=40, kde=True)
plt.title("Distribution of Flight Duration (Actual Elapsed Time)")
plt.xlabel("Elapsed Time (minutes)")
plt.ylabel("Number of Flights")
plt.show()

# COMMAND ----------

# --- Distance Distribution ---
plt.figure(figsize=(8,5))
sns.histplot(df['Distance'], bins=40, kde=True)
plt.title("Distribution of Flight Distance")
plt.xlabel("Distance (miles)")
plt.ylabel("Number of Flights")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Day vs Month Patterns (Bivariate Time Relationships)

# COMMAND ----------

# --- Flights by Day of Week and Month ---
plt.figure(figsize=(12,6))
sns.countplot(x='DayOfWeek', hue='Month', data=df)
plt.title("Flights by Day of Week and Month")
plt.xlabel("Day of Week (1=Mon ... 7=Sun)")
plt.ylabel("Number of Flights")
plt.legend(title="Month", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Top Destination Airports

# COMMAND ----------

# --- Top Destination Airports ---
plt.figure(figsize=(12,5))
top_dest = df['Dest_Airport'].value_counts().head(10)
sns.barplot(x=top_dest.index, y=top_dest.values)
plt.title("Top 10 Destination Airports")
plt.xlabel("Destination Airport")
plt.ylabel("Number of Arrivals")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Airline Market Share (Pie Chart)

# COMMAND ----------

plt.figure(figsize=(8,8))
airline_share = df['Airline'].value_counts().head(8)
plt.pie(airline_share.values, labels=airline_share.index, autopct='%1.1f%%', startangle=140)
plt.title("Market Share of Top 8 Airlines")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 5. Average Flight Duration by Airline

# COMMAND ----------

plt.figure(figsize=(10,5))
avg_duration = df.groupby('Airline')['ActualElapsedTime'].mean().sort_values(ascending=False)
sns.barplot(x=avg_duration.index, y=avg_duration.values)
plt.title("Average Flight Duration by Airline")
plt.xlabel("Airline")
plt.ylabel("Average Duration (minutes)")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 6. Average Distance by Route (Optional for deeper insight)

# COMMAND ----------

plt.figure(figsize=(12,5))
top_routes = df.groupby('Route')['Distance'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=top_routes.index, y=top_routes.values)
plt.title("Top 10 Longest Average Routes")
plt.xlabel("Route")
plt.ylabel("Average Distance (miles)")
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 7. Correlation Heatmap (Numerical Relationships)

# COMMAND ----------

plt.figure(figsize=(10,6))
numeric_cols = ['ActualElapsedTime', 'AirTime', 'Distance', 'TaxiIn', 'TaxiOut', 'Month', 'DepHour']
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='Blues', fmt='.2f')
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 8. Boxplot of AirTime by Day of Week

# COMMAND ----------

plt.figure(figsize=(10,6))
sns.boxplot(x='DayOfWeek', y='AirTime', data=df)
plt.title("Flight Duration by Day of Week")
plt.xlabel("Day of Week (1=Mon ... 7=Sun)")
plt.ylabel("AirTime (minutes)")
plt.show()

# COMMAND ----------

# Top 5 airlines by average distance
print(df.groupby('Airline')['Distance'].mean().sort_values(ascending=False).head())

# Flights per month summary
print(df['Month'].value_counts().sort_index())

# COMMAND ----------

plt.savefig("plot_name.png", bbox_inches='tight')