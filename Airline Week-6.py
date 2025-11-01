# Databricks notebook source
# Week 6: Seasonal Delay & Cancellation Insights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load dataset
df = pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay.csv")

display(df.head())

# Extract Month and DayOfWeek
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek + 1  

# COMMAND ----------

# Assign cancellation probabilities based on seasonal effect
import numpy as np

# Define month-wise cancellation probabilities
cancel_probs = {
    1: 0.22, 2: 0.20, 3: 0.08, 4: 0.05, 5: 0.05, 6: 0.07,
    7: 0.12, 8: 0.10, 9: 0.06, 10: 0.07, 11: 0.18, 12: 0.25
}

# Ensure 'Month' is integer and handle NaN
df['Month'] = pd.to_datetime(df['Date'], errors='coerce').dt.month

# Safely assign random cancellation based on month probability
def assign_cancellation(m):
    # if month is not in cancel_probs, assign default low probability
    prob = cancel_probs.get(m, 0.05)
    return np.random.choice([0, 1], p=[1 - prob, prob])

df['Cancelled'] = df['Month'].apply(assign_cancellation)

display(df[['Month', 'Cancelled']].head())

# COMMAND ----------

# Classify into Seasons
def assign_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4]:
        return 'Spring'
    elif month in [5, 6, 7]:
        return 'Summer'
    elif month in [8, 9]:
        return 'Monsoon'
    else:
        return 'Autumn'

df['Season'] = df['Month'].apply(assign_season)
display(df[['Month', 'Season']].head())

# COMMAND ----------

# Average Delay by Season (ArrivalDelay)
season_delay = df.groupby('Season')['ArrDelay'].mean().reset_index()

plt.figure(figsize=(7,5))
sns.barplot(data=season_delay, x='Season', y='ArrDelay', palette='viridis')
plt.title('Average Arrival Delay by Season')
plt.ylabel('Average Delay (mins)')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# Relationship between Delay and Cancellation Rate
df['DelayFlag'] = df['ArrDelay'].apply(lambda x: 1 if x > 15 else 0)

corr_table = df.groupby('Season')[['Cancelled', 'DelayFlag']].mean().reset_index()
corr_table.columns = ['Season', 'Avg_Cancel_Rate', 'Avg_Delay_Rate']

fig = px.scatter(
    corr_table,
    x='Avg_Delay_Rate',
    y='Avg_Cancel_Rate',
    color='Season',
    size='Avg_Delay_Rate',
    title='Delay Rate vs Cancellation Rate (Season-wise)',
    labels={'Avg_Delay_Rate':'Average Delay Rate', 'Avg_Cancel_Rate':'Average Cancellation Rate'}
)
fig.show()


# COMMAND ----------

# Monthly Delay Distribution (Box Plot)
plt.figure(figsize=(10,5))
sns.boxplot(x='Month', y='ArrDelay', data=df, palette='coolwarm')
plt.title('Monthly Distribution of Arrival Delays')
plt.xlabel('Month')
plt.ylabel('Arrival Delay (mins)')
plt.show()

# COMMAND ----------

# Airline-level Heatmap: Monthly Average Delay
heatmap_data = df.groupby(['Airline', 'Month'])['ArrDelay'].mean().reset_index()
pivot_heatmap = heatmap_data.pivot(index='Airline', columns='Month', values='ArrDelay')

plt.figure(figsize=(12,6))
sns.heatmap(pivot_heatmap, cmap='YlGnBu', annot=True, fmt='.1f')
plt.title('Airline vs Month: Average Delay Heatmap')
plt.xlabel('Month')
plt.ylabel('Airline')
plt.tight_layout()
plt.show()

# COMMAND ----------

# Correlation between Delay Causes
delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
corr_matrix = df[delay_cols].corr()

plt.figure(figsize=(7,5))
sns.heatmap(corr_matrix, annot=True, cmap='crest')
plt.title('Correlation Between Delay Causes')
plt.show()

# COMMAND ----------

#Top 5 Airlines by Average Delay
top_airlines = df.groupby('Airline')['ArrDelay'].mean().sort_values(ascending=False).head(5)
top_airlines.plot(kind='bar', figsize=(8,4), color='salmon')
plt.title('Top 5 Airlines with Highest Average Delay')
plt.ylabel('Average Delay (mins)')
plt.show()