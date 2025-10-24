# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# COMMAND ----------

df=pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay_processed.csv")
display(df)

# COMMAND ----------

df = df.drop(
    df.columns[0],
    axis=1,
    errors='ignore'
)
display(df)

# COMMAND ----------

#Average Delay by Cause and Airline
delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
airline_delays = df.groupby('Airline')[delay_cols].mean().reset_index()

# Plot
airline_delays_melted = airline_delays.melt(id_vars='Airline', var_name='DelayType', value_name='AverageDelay')

plt.figure(figsize=(12,6))
sns.barplot(data=airline_delays_melted, x='Airline', y='AverageDelay', hue='DelayType')
plt.title('Average Delay by Cause and Airline')
plt.ylabel('Average Delay (minutes)')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

#Exploring Carrier, Weather, and NAS Delays
plt.figure(figsize=(10,6))
sns.boxplot(data=df[['CarrierDelay', 'WeatherDelay', 'NASDelay','SecurityDelay', 'LateAircraftDelay']])
plt.title('Distribution of Major Delay Types')
plt.ylabel('Delay (minutes)')
plt.show()

# COMMAND ----------

#Distribution of all the Delay types
delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

plt.figure(figsize=(15, 8))
for i, col in enumerate(delay_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel('Delay (minutes)')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# COMMAND ----------

#Outliers for Each Delay Type
plt.figure(figsize=(15, 8))
for i, col in enumerate(delay_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')
    plt.ylabel('Delay (minutes)')
plt.tight_layout()
plt.show()

# COMMAND ----------

#Mean Delay of Each Delay Type
delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
mean_delays = df[delay_cols].mean().reset_index()
mean_delays.columns = ['DelayType', 'MeanDelay']
plt.figure(figsize=(8,6))
sns.barplot(data=mean_delays, x='DelayType', y='MeanDelay')
plt.title('Mean Delay per Type')
plt.ylabel('Mean Delay (minutes)')
plt.xlabel('Delay Type')
plt.show()

# COMMAND ----------

#Visualize Delays by Time of Day
hourly_delays = df.groupby('DepHour')['TotalDelay'].mean().reset_index()

plt.figure(figsize=(10,6))
sns.lineplot(data=hourly_delays, x='DepHour', y='TotalDelay', marker='o')
plt.title('Average Total Delay by Time of Day')
plt.xlabel('Departure Hour')
plt.ylabel('Average Delay (minutes)')
plt.show()

# COMMAND ----------

#Average Delay by type and time of the day
delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

hourly_means = df.groupby('DepHour')[delay_cols].mean().reset_index()

plt.figure(figsize=(12, 7))
for col in delay_cols:
    sns.lineplot(data=hourly_means, x='DepHour', y=col, marker='o', label=col)
plt.title('Average Delay by Type and Time of Day')
plt.xlabel('Departure Hour')
plt.ylabel('Average Delay (minutes)')
plt.legend(title='Delay Type')
plt.xticks(hourly_means['DepHour'])
plt.show()

# COMMAND ----------

#Average Total Delay by Hour for Top Airlines
top5_airlines = df.groupby('Airline')['TotalDelay'].mean().nlargest(5).index
df_top5 = df[df['Airline'].isin(top5_airlines)]

hourly_airline_delays = df_top5.groupby(['DepHour', 'Airline'])['TotalDelay'].mean().reset_index()

plt.figure(figsize=(12,7))
sns.lineplot(data=hourly_airline_delays, x='DepHour', y='TotalDelay', hue='Airline', marker='o')
plt.title('Average Total Delay by Hour for Top 5 Airlines')
plt.xlabel('Departure Hour')
plt.ylabel('Average Delay (minutes)')
plt.legend(title='Airline')
plt.xticks(hourly_airline_delays['DepHour'].unique())
plt.show()

# COMMAND ----------

#Average Departure delayby hour for Top 5 Airlines
if all(col in df.columns for col in['DepTime','Org_Airport','DepDelay']):
    df['DEP_HOUR']=df['DepTime']//100
    top_airports=df['Org_Airport'].value_counts().head(5).index
    df.top=df[df['Org_Airport'].isin(top_airports)]
    plt.figure(figsize=(14,6))
    sns.lineplot(
        data=df.top,
        x='DEP_HOUR',
        y='DepDelay',
        hue='Org_Airport',
        estimator='mean',
        ci=None,
        marker='o'
    )
    plt.title('Average Departure Delay by Hour for Top 5 Airports')
    plt.xlabel('Departure Hour')
    plt.ylabel('Average Departure Delay (minutes)')
    plt.legend(title='Origin Airport')
    plt.xticks(range(0,24))
    plt.show()

# COMMAND ----------

# Average Arrival Delay by Hour for Top 5 Airports
if all(col in df.columns for col in ['ArrDelay', 'DepTime', 'Org_Airport']):
    df['ARR_HOUR'] = df['DepTime'] // 100
    top_airports = df['Org_Airport'].value_counts().head(5).index
    df_top = df[df['Org_Airport'].isin(top_airports)]
    plt.figure(figsize=(14,6))
    sns.lineplot(
        data=df_top,
        x='ARR_HOUR',
        y='ArrDelay',
        hue='Org_Airport',
        estimator='mean',
        ci=None,
        marker='o'
    )
    plt.title('Average Arrival Delay by Hour for Top 5 Airports')
    plt.xlabel('Departure Hour')
    plt.ylabel('Average Arrival Delay (minutes)')
    plt.legend(title='Origin Airport')
    plt.xticks(range(0,24))
    plt.show()

# COMMAND ----------

#Visualize Delays by Airport
origin_delays = df.groupby('Origin')['TotalDelay'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=origin_delays.index, y=origin_delays.values)
plt.title('Top 10 Origin Airports by Average Delay')
plt.xlabel('Origin Airport')
plt.ylabel('Average Delay (minutes)')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

#Correlation between Delay Causes
plt.figure(figsize=(8,6))
sns.heatmap(df[delay_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Delay Causes')
plt.show()

# COMMAND ----------

# Correlation of delay by hour for top 5 Airports using heatmap
if all(col in df.columns for col in ['DepTime', 'Org_Airport', 'TotalDelay']):
    df['DEP_HOUR'] = df['DepTime'] // 100
    top_airports = df['Org_Airport'].value_counts().head(5).index
    df_top = df[df['Org_Airport'].isin(top_airports)]
    pivot_table = df_top.groupby(['Org_Airport', 'DEP_HOUR'])['TotalDelay'].mean().round(2).unstack(fill_value=0)
    plt.figure(figsize=(24, 6))
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Average Total Delay by Hour for Top 5 Airports')
    plt.xlabel('Departure Hour')
    plt.ylabel('Origin Airport')
    plt.show()

# COMMAND ----------

#An Interactive Bar chart for Airline Delay Comparision
fig = px.bar(airline_delays_melted, x='Airline', y='AverageDelay', color='DelayType', 
             title='Airline Delay Comparison')
fig.show()

# COMMAND ----------

