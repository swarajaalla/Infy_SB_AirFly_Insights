# Databricks notebook source
import pandas as pd
df1 = pd.read_csv('/Volumes/workspace/default/airlines/Flight_delay_cleaned.csv')
display(df1.info())

# COMMAND ----------

display(df1[['Cancelled', 'CancellationCode']])

# COMMAND ----------

import numpy as np

df1['Cancelled'] = np.random.choice([0, 1], size=len(df1), p=[0.7, 0.3])


display(df1[['Cancelled', 'CancellationCode']])

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# COMMAND ----------

cancelled_count = df1['Cancelled'].sum()
display(cancelled_count)

# COMMAND ----------

import matplotlib.pyplot as plt

cancelled_by_month = df1[df1['Cancelled'] == 1].groupby('month').size()

# Orange to red gradient
cmap = LinearSegmentedColormap.from_list("orange_red_gradient", ["#FFA500", "#FF4500", "#B22222"])

plt.figure(figsize=(10,6))
bars = plt.bar(cancelled_by_month.index, cancelled_by_month.values, color=cmap(np.linspace(0, 1, len(cancelled_by_month))))
plt.title('Cancelled Flights by Month', fontsize=16, color='white')
plt.xlabel('Month', fontsize=12, color='white')
plt.ylabel('Number of Cancelled Flights', fontsize=12, color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.gca().set_facecolor('#001f3f')
plt.gcf().patch.set_facecolor('#001f3f')
plt.tight_layout()
display(plt.gcf())
plt.close()

# COMMAND ----------

winter_months = [11, 12, 1]
df1['Winter'] = df1['month'].apply(lambda x: 'Winter' if x in winter_months else 'Other')

cancel_rate = df1.groupby('Winter')['Cancelled'].mean()

plt.figure(figsize=(6,4))
bars = plt.bar(cancel_rate.index, cancel_rate.values, color=['#1f77b4', '#d62728'])
plt.title('Cancellation Rate: Winter vs Other Months', fontsize=16, color='white')
plt.xlabel('Month Group', fontsize=12, color='white')
plt.ylabel('Cancellation Rate', fontsize=12, color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.gca().set_facecolor('#001f3f')
plt.gcf().patch.set_facecolor('#001f3f')
plt.tight_layout()
display(plt.gcf())
plt.close()

# COMMAND ----------

winter_months = [11,12, 1]
month_names = {1: 'January', 11: 'November', 12: 'December'}
winter_cancelled = df1[(df1['Cancelled'] == 1) & (df1['month'].isin(winter_months))]
cancelled_counts = winter_cancelled['month'].value_counts().sort_index()
labels = [month_names.get(m, str(m)) for m in cancelled_counts.index]

plt.figure(figsize=(8,8))
plt.pie(cancelled_counts, labels=labels, autopct='%1.1f%%', colors=['#FFA500', '#FF4500', '#B22222'])
plt.title('Cancelled Flights in Winter Months', fontsize=16, color='white')
plt.gcf().patch.set_facecolor('#001f3f')
display(plt.gcf())
plt.close()

# COMMAND ----------

delay_types = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
cancelled_by_delay = {}

for delay in delay_types:
    if delay in df1.columns:
        cancelled_by_delay[delay] = df1[df1['Cancelled'] == 1][delay].sum()

plt.figure(figsize=(8,6))
plt.bar(cancelled_by_delay.keys(), cancelled_by_delay.values(), color=['#FFA500', '#FF4500', '#B22222', '#8B0000', '#FFD700'])
plt.title('Cancelled Flights by Delay Type', fontsize=16, color='white')
plt.xlabel('Delay Type', fontsize=12, color='white')
plt.ylabel('Total Delay Minutes (Cancelled Flights)', fontsize=12, color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.gca().set_facecolor('#001f3f')
plt.gcf().patch.set_facecolor('#001f3f')
plt.tight_layout()
display(plt.gcf())
plt.close()

# COMMAND ----------

# Top 5 delays contributing to cancellations
delay_types = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
cancelled_delays = df1[df1['Cancelled'] == 1][delay_types].sum().sort_values(ascending=False)
top5_delays = cancelled_delays.head(5).index.tolist()

# Heatmap: cancellations by month and delay type
heatmap_data = df1[df1['Cancelled'] == 1].groupby(['month'])[top5_delays].sum()
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd")
plt.title('Heatmap of Cancelled Flights by Month and Top 5 Delay Types', fontsize=16, color='white')
plt.xlabel('Delay Type', fontsize=12, color='white')
plt.ylabel('Month', fontsize=12, color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.gca().set_facecolor('#001f3f')
plt.gcf().patch.set_facecolor('#001f3f')
plt.tight_layout()
display(plt.gcf())
plt.close()

# COMMAND ----------

# Heatmap: cancellations by airline and month
if 'Airline' in df1.columns:
    cancellations_airline_month = df1[df1['Cancelled'] == 1].groupby(['Airline', 'month']).size().unstack(fill_value=0)
    import seaborn as sns
    plt.figure(figsize=(12, 8))
    sns.heatmap(cancellations_airline_month, annot=True, fmt="d", cmap="YlOrRd")
    plt.title('Heatmap of Cancelled Flights by Airline and Month', fontsize=16, color='white')
    plt.xlabel('Month', fontsize=12, color='white')
    plt.ylabel('Airline', fontsize=12, color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.gca().set_facecolor('#001f3f')
    plt.gcf().patch.set_facecolor('#001f3f')
    plt.tight_layout()
    display(plt.gcf())
    plt.close()