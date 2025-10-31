# Databricks notebook source
# MAGIC %md
# MAGIC Week 6 : Seasonal and Cancellation Analysis

# COMMAND ----------

'''
1. modify the cancelled column to change the values as 0 or 1
then use the random for assigning the values 
then for cancellation code use mapping technique to mapp the cancellation code 0 for not Cancelled(N) & 1 for cancelled

2. plot a bar chart for the cancellation by months 

3. plot a bar chart to show Cancellation Cause Type

4. Plot a graph for average Cancellation Rate: Winter vs Non-Winter

5 Use heatmap for Month vs Airline cancellation rates.

6. Airline Level- seasonal Cancellation rate analysis
'''

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import plotly as px
import seaborn as sns

# COMMAND ----------

df=pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay_processed.csv")
display(df)

# COMMAND ----------

display(df.columns)

# COMMAND ----------

# Extract 'DayOfWeek' and 'Month' from 'Date'
df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek + 1  # 1=Monday, 7=Sunday
df['Month'] = pd.to_datetime(df['Date']).dt.month

display(df[['Date', 'DayOfWeek', 'Month']])

# COMMAND ----------

import numpy as np

# Assign different cancellation probabilities for each month
month_cancel_probs = {
    1: 0.25,  # January - high cancellations (e.g., winter)
    2: 0.20,  # February - high
    3: 0.10,  # March - moderate
    4: 0.05,  # April - low
    5: 0.05,  # May - low
    6: 0.08,  # June - moderate
    7: 0.12,  # July - moderate
    8: 0.07,  # August - low
    9: 0.05,  # September - low
    10: 0.06, # October - low
    11: 0.18, # November - high (e.g., weather/holidays)
    12: 0.22  # December - high (e.g., winter/holidays)
}

# Generate cancellations based on month-specific probabilities
df['Cancelled'] = df['Month'].apply(lambda m: np.random.choice([0, 1], p=[1 - month_cancel_probs[m], month_cancel_probs[m]]))
display(df)

# COMMAND ----------

def map_cancellation_code(row):
    if row['Cancelled'] == 0:
        return 'N'
    if row['CarrierDelay'] > 0:
        return 'A'
    if row['WeatherDelay'] > 0:
        return 'B'
    if row['NASDelay'] > 0:
        return 'C'
    if row['SecurityDelay'] > 0:
        return 'D'
    if row['LateAircraftDelay'] > 0:
        return 'E'
    return 'N'

df['CancellationCode'] = df.apply(map_cancellation_code, axis=1)
display(df[['Cancelled', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'CancellationCode']])

# COMMAND ----------

import plotly.express as px

# Group by Month and sum cancellations
monthly_cancellations = df.groupby('Month')['Cancelled'].sum().reset_index()

# Map month numbers to names for better readability
month_map = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
}
monthly_cancellations['MonthName'] = monthly_cancellations['Month'].map(month_map)

# Sort by number of cancellations (height of bar)
monthly_cancellations = monthly_cancellations.sort_values('Cancelled', ascending=False)

fig = px.bar(
    monthly_cancellations,
    x='MonthName',
    y='Cancelled',
    labels={'MonthName': 'Month', 'Cancelled': 'Number of Cancellations'},
    title='Interactive Bar Plot: Cancellation Trend Over Months',
    color='MonthName'
)
fig.update_layout(
    xaxis_title='Month',
    yaxis_title='Cancellations',
    template='plotly_dark',
    showlegend=False
)
fig.show()

# COMMAND ----------

import plotly.express as px

# Map cancellation codes to reasons
reason_map = {
    'A': 'Carrier Delay',
    'B': 'Weather Delay',
    'C': 'NAS Delay',
    'D': 'Security Delay',
    'E': 'Late Aircraft Delay'
}

# Count cancellations by reason (excluding 'N' for not cancelled)
cause_counts = df[df['CancellationCode'] != 'N']['CancellationCode'].map(reason_map).value_counts().reset_index()
cause_counts.columns = ['CancellationReason', 'Count']

fig = px.bar(
    cause_counts,
    x='CancellationReason',
    y='Count',
    labels={'CancellationReason': 'Cancellation Reason', 'Count': 'Number of Cancellations'},
    title='Bar Chart: Cancellation Reason',
    color='CancellationReason'
)
fig.update_layout(
    xaxis_title='Cancellation Reason',
    yaxis_title='Number of Cancellations',
    template='plotly_dark',
    showlegend=False
)
fig.show()

# COMMAND ----------

import matplotlib.pyplot as plt

# Define winter and non-winter months
winter_months = [12, 1, 2]
df['Season'] = df['Month'].apply(lambda m: 'Winter' if m in winter_months else 'Non-Winter')

# Calculate average cancellation rate for each season
seasonal_cancellation = df.groupby('Season')['Cancelled'].mean().reset_index()

# Plot
plt.figure(figsize=(6,4))
plt.bar(seasonal_cancellation['Season'], seasonal_cancellation['Cancelled'], color=['skyblue', 'salmon'])
plt.ylabel('Average Cancellation Rate')
plt.title('Average Cancellation Rate: Winter vs Non-Winter')
plt.ylim(0, 1)
plt.show()

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate cancellation rate per Airline per Month
pivot = df.groupby(['Month', 'Airline'])['Cancelled'].mean().reset_index()
heatmap_data = pivot.pivot(index='Airline', columns='Month', values='Cancelled')

plt.figure(figsize=(12, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    cbar_kws={'label': 'Cancellation Rate'}
)
plt.title('Heatmap of Cancellation Rates: Month vs Airline')
plt.xlabel('Month')
plt.ylabel('Airline')
plt.tight_layout()
plt.show()

# COMMAND ----------

import numpy as np

# Define custom seasons
def assign_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4]:
        return 'Spring'
    elif month in [5, 6, 7]:
        return 'Summer'
    elif month in [8, 9]:
        return 'Rainy'
    elif month in [10, 11]:
        return 'Autumn'
    else:
        return 'Unknown'

df['Season'] = df['Month'].apply(assign_season)

# Calculate average cancellation rate per Airline per Season
airline_seasonal = df.groupby(['Airline', 'Season'])['Cancelled'].mean().reset_index()

# Pivot for better visualization
airline_seasonal_pivot = airline_seasonal.pivot(index='Airline', columns='Season', values='Cancelled')

yticks = np.arange(0, 0.18, 0.02)
plt.figure(figsize=(10, 12))  # Increased height
ax = airline_seasonal_pivot.plot(kind='bar', figsize=(12,12), color=['skyblue', 'salmon', 'gold', 'limegreen', 'orange'])
plt.ylabel('Average Cancellation Rate')
plt.title('Airline Level: Seasonal Cancellation Rate Analysis')
plt.ylim(0, 0.16)
plt.yticks(yticks, [f"{x:.2f}" for x in yticks])
plt.xticks(rotation=45)
plt.legend(title='Season')
plt.tight_layout()
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

winter_months = ['December', 'January', 'February']

# Ensure 'Date' is datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Extract month name
df['Month'] = df['Date'].dt.month_name()

# Assign season
df['Season'] = df['Month'].apply(
    lambda x: 'Winter' if x in winter_months else 'Non_Winter'
)

# Group and pivot
airline_season_cancel = df.groupby(
    ['Airline', 'Season']
)['Cancelled'].mean().reset_index()

pivot_airline_season = airline_season_cancel.pivot(
    index='Airline',
    columns='Season',
    values='Cancelled'
)

# Plot
pivot_airline_season.plot(
    kind='bar',
    figsize=(12, 6),
    title='Airline Cancellation Rate: Winter vs Non_Winter',
    ylabel='Cancellation Rate'
)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()