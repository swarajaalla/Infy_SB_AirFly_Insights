# Databricks notebook source
import pandas as pd

df = pd.read_csv('/Volumes/workspace/default/airlines/Flight_delay.csv')
df=df.drop_duplicates()
display(df)

# COMMAND ----------

display(df.info())

# COMMAND ----------

null_counts = df.isnull().sum()
display(null_counts)

# COMMAND ----------

df['Date']=pd.to_datetime(df['Date'])


# COMMAND ----------

from datetime import datetime

def convert_to_time_format(time_int):
    try:
        if pd.isnull(time_int):
            return None
        time_str = str(int(time_int)).zfill(4)
        time_str = time_str[-4:]
        return datetime.strptime(time_str, "%H%M").strftime("%H:%M")
    except Exception:
        return None

columns_to_convert = ['DepTime', 'ArrTime', 'CRSArrTime']
df[columns_to_convert] = df[columns_to_convert].applymap(convert_to_time_format)
display(df)

# COMMAND ----------

columns_to_fill = [
    'DayOfWeek', 'ActualElapsedTime', 'CRSElapsedTime', 'ArrDelay', 'DepDelay',
    'Distance', 'AirTime', 'CarrierDelay', 'WeatherDelay', 'NASDelay',
    'SecurityDelay', 'LateAircraftDelay', 'Diverted', 'TaxiIn', 'TaxiOut'
]
means = df[columns_to_fill].mean()
df[columns_to_fill] = df[columns_to_fill].fillna(means)


# COMMAND ----------

null_counts = df.isnull().sum()
display(null_counts)

# COMMAND ----------

columns_to_fill = ['DepTime', 'ArrTime','Org_Airport','Dest_Airport']
for col in columns_to_fill:
    df[col] = df.groupby('FlightNum')[col].transform(
        lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x
    )

# COMMAND ----------

df= df.dropna()

# COMMAND ----------

null_counts = df.isnull().sum()
display(null_counts)

# COMMAND ----------

df['month']=df['Date'].dt.month
df['Hour'] = df['Date'].dt.hour

# COMMAND ----------

df['Route'] = df['Origin'] + '->' + df['Dest']
display(df)

# COMMAND ----------

day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
flight_count_by_day = df['DayOfWeek'].value_counts().sort_index()
flight_count_by_day = flight_count_by_day.rename(index=day_names)
df['DayName'] = df['DayOfWeek'].map(day_names)
display(flight_count_by_day)

# COMMAND ----------

min_distance = df['Distance'].min()
max_distance = df['Distance'].max()
avg_distance = df['Distance'].mean()
print(min_distance, max_distance, avg_distance)

# COMMAND ----------

df['TotalDelay'] = df['WeatherDelay'] + df['CarrierDelay'] + df['NASDelay']+df['SecurityDelay']+df['LateAircraftDelay']
route_delay = df.groupby('Route')['TotalDelay'].sum()
max_delay_route = route_delay.idxmax()
min_delay_route = route_delay.idxmin()
display(route_delay)
print(f"Route with most delay: {max_delay_route}")
print(f"Route with least delay: {min_delay_route}")

# COMMAND ----------

display(df.info())

# COMMAND ----------

avg_delay_by_day = df.groupby('DayName')['TotalDelay'].mean().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
display(avg_delay_by_day)

# COMMAND ----------

df.to_csv('/Volumes/workspace/default/airlines/Flight_delay_cleaned.csv')

# COMMAND ----------

