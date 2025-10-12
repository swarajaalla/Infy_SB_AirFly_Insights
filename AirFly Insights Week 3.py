# Databricks notebook source
import pandas as pd

# Load your cleaned dataset
df = pd.read_csv('/Volumes/airfly_workspace/default/airfly_insights/flights_cleaned.csv')

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# 1. Month (numeric month from Date)
df['Month'] = df['Date'].dt.month

# 2. DayOfWeekNum (use existing DayOfWeek column)
# If DayOfWeek already exists as numeric (1=Mon etc.), just copy or standardize
df['DayOfWeekNum'] = df['DayOfWeek'] - 1  # optional normalization (0=Mon)

# 3. DepHour (extract hour from DepTime)
def extract_hour(x):
    try:
        x = int(x)
        hour = x // 100
        if 0 <= hour < 24:
            return hour
        else:
            return None
    except:
        return None

df['DepHour'] = df['DepTime'].apply(extract_hour)

# 4. Route (Origin–Destination)
df['Route'] = df['Origin'] + '-' + df['Dest']

# Verify new columns
print(df[['Month', 'DayOfWeekNum', 'DepHour', 'Route']].head())


# COMMAND ----------

# Save the Updated File

df.to_csv('/Volumes/airfly_workspace/default/airfly_insights/flights_cleaned.csv', index=False)
print("Updated cleaned dataset saved successfully.")

# COMMAND ----------

# Top Airlines by Number of Flights

plt.figure(figsize=(10,6))
df['Airline'].value_counts().head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Airlines by Number of Flights')
plt.xlabel('Airline')
plt.ylabel('Number of Flights')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# Top Routes by Flight Volume

plt.figure(figsize=(10,6))
df['Route'].value_counts().head(10).plot(kind='bar', color='lightgreen')
plt.title('Top 10 Routes by Flight Count')
plt.xlabel('Route (Origin–Destination)')
plt.ylabel('Number of Flights')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# Busiest Months

plt.figure(figsize=(10,6))
df['Month'].value_counts().sort_index().plot(kind='bar', color='orange')
plt.title('Flights per Month')
plt.xlabel('Month')
plt.ylabel('Number of Flights')
plt.show()

# COMMAND ----------

# Flights by Day of Week

plt.figure(figsize=(8,5))
sns.countplot(x='DayOfWeekNum', data=df, palette='viridis')
plt.title('Flights by Day of the Week')
plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
plt.ylabel('Flight Count')
plt.show()

# COMMAND ----------

# Departure Hour Distribution

plt.figure(figsize=(10,5))
sns.histplot(df['DepHour'], bins=24, kde=True)
plt.title('Distribution of Flights by Departure Hour')
plt.xlabel('Departure Hour')
plt.ylabel('Flight Count')
plt.show()

# COMMAND ----------

# Flights by Origin Airport

plt.figure(figsize=(12,6))
top_airports = df['Origin'].value_counts().head(10)
sns.barplot(x=top_airports.index, y=top_airports.values, palette='coolwarm')
plt.title('Top 10 Origin Airports by Flight Count')
plt.xlabel('Origin Airport')
plt.ylabel('Number of Flights')
plt.show()

# COMMAND ----------

# Distribution of Arrival Delay

plt.figure(figsize=(10,5))
sns.histplot(df['ArrDelay'], bins=50, kde=True)
plt.title('Distribution of Arrival Delays')
plt.xlabel('Arrival Delay (minutes)')
plt.ylabel('Number of Flights')
plt.show()

# COMMAND ----------

# Boxplot of Arrival Delay by Airline

plt.figure(figsize=(12,6))
sns.boxplot(x='Airline', y='ArrDelay', data=df)
plt.title('Arrival Delay by Airline')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# Average Delay by Month (Line Plot)

monthly_delay = df.groupby('Month')['ArrDelay'].mean().reset_index()

plt.figure(figsize=(8,5))
sns.lineplot(x='Month', y='ArrDelay', data=monthly_delay, marker='o')
plt.title('Average Arrival Delay by Month')
plt.xlabel('Month')
plt.ylabel('Average Delay (minutes)')
plt.show()