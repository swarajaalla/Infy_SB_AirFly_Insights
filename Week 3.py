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
# MAGIC Univariate analysis

# COMMAND ----------

#Top 10 Airlines by Flight Count
plt.figure(figsize=(10,5))
sns.countplot(data=df, y='Airline', order=df['Airline'].value_counts().index[:10])
plt.title('Top 10 Airlines by Flight Count')
plt.xlabel('Number of Flights')
plt.ylabel('Airline')
plt.show()

# COMMAND ----------

#Top 10 Busiest Routes (treating ORD-LGA and LGA-ORD as the same route)
df['Route_Undirected'] = df['Route'].apply(lambda x: '-'.join(sorted(x.split('-'))))
plt.figure(figsize=(12,6))
sns.countplot(data=df, y='Route_Undirected', order=df['Route_Undirected'].value_counts().index[:10])
plt.title('Top 10 Busiest Routes (Undirected)')
plt.xlabel('Number of Flights')
plt.ylabel('Route')
plt.show()

# COMMAND ----------

# Cancellation Codes
plt.figure(figsize=(7,4))
cancel_counts = df['CancellationCode'].value_counts(dropna=False)
plt.pie(cancel_counts, labels=cancel_counts.index, autopct='%1.1f%%', startangle=90)
#plt.pie(cancel_counts, labels=cancel_counts.index, autopct='%1.12f%%', startangle=90)
plt.title('Distribution of Cancellation Reasons')
plt.ylabel('')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Numerical Variables

# COMMAND ----------

num_vars = ['DepDelay', 'ArrDelay', 'TotalDelay', 'Distance', 'AirTime', 'TaxiOut', 'TaxiIn']

for var in num_vars:
    plt.figure(figsize=(8,4))
    sns.histplot(df[var], kde=True, bins=40)
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()

# COMMAND ----------

# Boxplots for Outliers
for var in ['TotalDelay']:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df[var])
    plt.title(f'Boxplot of {var}')
    plt.xlabel(var)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Temporal Variables

# COMMAND ----------

# Flights by Month (with Month Names)
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
df['MonthName'] = df['Month'].map(month_names)
plt.figure(figsize=(8,4))
sns.countplot(x='MonthName', data=df, order=[month_names[m] for m in sorted(df['Month'].unique())])
plt.title('Flights by Month')
plt.xlabel('Month')
plt.ylabel('Number of Flights')
plt.show()

# COMMAND ----------

# Flights by Day of Week
plt.figure(figsize=(8,4))
sns.countplot(x='DayOfWeek', data=df, order=sorted(df['DayOfWeek'].unique()))
plt.title('Flights by Day of Week (1=Mon, 7=Sun)')
plt.xlabel('Day of Week')
plt.ylabel('Number of Flights')
plt.show()

# COMMAND ----------

# Flights by Hour
plt.figure(figsize=(8,4))
sns.histplot(df['DepHour'], bins=24)
plt.title('Flight Distribution by Departure Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Flights')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Bivariate Analysis

# COMMAND ----------

    #Airline vs. Average Delay
    plt.figure(figsize=(10,5))
    sns.barplot(x='Airline', y='TotalDelay', data=df, estimator='mean', ci=None)
    plt.title('Average Total Delay per Airline')
    plt.xlabel('Airline')
    plt.ylabel('Average Total Delay (minutes)')
    plt.xticks(rotation=45)
    plt.show()


# COMMAND ----------

# Flight Distribution by Airport (Origin)
plt.figure(figsize=(12,6))
top_airports = df['Origin'].value_counts().head(15).index
sns.countplot(x='Origin', data=df[df['Origin'].isin(top_airports)], order=top_airports, palette='viridis')
plt.title('Flight Distribution by Origin Airport')
plt.xlabel('Origin Airport')
plt.ylabel('Number of Flights')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='DayOfWeek', y='TotalDelay', palette='Set2')
plt.title("Total Delay Distribution by Day of Week", fontsize=14)
plt.xlabel("Day of Week (1=Mon, 7=Sun)", fontsize=12)
plt.ylabel("Total Delay (minutes)", fontsize=12)
plt.tight_layout()
plt.show()


# COMMAND ----------

#Origin Airport vs Delay
plt.figure(figsize=(10,5))
top_origins = df['Origin'].value_counts().head(10).index
sns.barplot(x='Origin', y='TotalDelay', data=df[df['Origin'].isin(top_origins)], estimator='mean', ci=None)
plt.title('Average Delay by Origin Airport')
plt.xlabel('Origin Airport')
plt.ylabel('Average Delay (minutes)')
plt.show()


# COMMAND ----------

#Month vs. Average Delay
plt.figure(figsize=(8,4))
sns.lineplot(data=df, x='Month', y='TotalDelay', estimator='mean', marker='o', label='Avg Delay')
plt.title('Average Total Delay by Month')
plt.xlabel('Month')
plt.ylabel('Average Delay (minutes)')
plt.legend(title='Metric')
plt.show()


# COMMAND ----------

plt.figure(figsize=(10,5))
sns.lineplot(data=df, x='DepHour', y='TotalDelay', estimator='mean', ci=None, marker='o', color='coral')
plt.title("Average Delay by Departure Hour", fontsize=14)
plt.xlabel("Departure Hour (0-23)", fontsize=12)
plt.ylabel("Average Delay (minutes)", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


# COMMAND ----------

#Distance vs Delay
plt.figure(figsize=(8,5))
sns.scatterplot(x='Distance', y='TotalDelay', data=df, alpha=0.5, label='Flights')
plt.title('Distance vs Total Delay')
plt.xlabel('Distance (miles)')
plt.ylabel('Total Delay (minutes)')
plt.legend()
plt.show()


# COMMAND ----------

plt.figure(figsize=(10,6))
sns.barplot(data=df, x='Airline', y='ArrDelay', estimator='mean', ci=None, palette='coolwarm')
plt.title("Average Arrival Delay by Airline", fontsize=14)
plt.xlabel("Airline", fontsize=12)
plt.ylabel("Average Arrival Delay (minutes)", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# COMMAND ----------

#Departure vs Arrival Delay
plt.figure(figsize=(8,5))
sns.scatterplot(x='DepDelay', y='ArrDelay', data=df, alpha=0.5)
plt.title('Departure Delay vs Arrival Delay')
plt.xlabel('Departure Delay (minutes)')
plt.ylabel('Arrival Delay (minutes)')
plt.legend(['Flights'])
plt.show()


# COMMAND ----------

#Hour vs. Average Delay
plt.figure(figsize=(8,4))
sns.lineplot(x='Hour', y='TotalDelay', data=df, estimator='mean', marker='o', label='Avg Delay')
plt.title('Average Delay by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Delay (minutes)')
plt.legend()
plt.show()


# COMMAND ----------

top_routes = df['Route'].value_counts().head(10).index
plt.figure(figsize=(10,6))
sns.barplot(data=df[df['Route'].isin(top_routes)], x='Route', y='TotalDelay', estimator='mean', ci=None, palette='crest')
plt.title("Average Delay by Top 10 Routes", fontsize=14)
plt.xlabel("Route", fontsize=12)
plt.ylabel("Average Total Delay (minutes)", fontsize=12)
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Multivariate Analysis

# COMMAND ----------

#Correlation Heatmap
plt.figure(figsize=(8,6))
corr = df[['DepDelay', 'ArrDelay', 'AirTime', 'Distance', 'TaxiOut', 'TaxiIn', 'TotalDelay']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Key Numeric Features')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()


# COMMAND ----------

sns.pairplot(df[['DepDelay', 'ArrDelay', 'AirTime', 'Distance', 'TotalDelay']])
plt.suptitle('Pairwise Relationships Between Variables', y=1.02)
plt.show()


# COMMAND ----------

g = sns.FacetGrid(df, col="Airline", col_wrap=3, height=3)
g.map_dataframe(sns.boxplot, x='DayOfWeek', y='TotalDelay')
g.set_titles(col_template="{col_name}")
g.set_axis_labels('Day of Week', 'Total Delay (min)')
plt.suptitle('Delay Distribution by Day for Each Airline', y=1.05)
plt.show()

# COMMAND ----------

delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
delay_summary = df.groupby('Airline')[delay_cols].mean().reset_index()

delay_summary.plot(
    x='Airline', kind='bar', stacked=True, figsize=(10,6),
    ylabel='Average Delay (minutes)', title='Average Delay Causes per Airline'
)
plt.legend(title='Delay Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Airline')
plt.ylabel('Average Delay (minutes)')
plt.show()


# COMMAND ----------

plt.figure(figsize=(10,5))
sns.countplot(x='Airline', hue='Cancelled', data=df)
plt.title('Flight Cancellations by Airline')
plt.xlabel('Airline')
plt.ylabel('Number of Flights')
plt.legend(title='Cancelled', labels=['No', 'Yes'])
plt.xticks(rotation=45)
plt.show()
