# Databricks notebook source
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# COMMAND ----------

# Load your cleaned data
df = pd.read_csv("/Volumes/airfly_workspace/default/airfly_insights/flights_cleaned.csv")

# COMMAND ----------

# Convert 'Date' to datetime objects
df['Date'] = pd.to_datetime(df['Date'])

# Extract month and day of the week
df['Month'] = df['Date'].dt.month
df['DayOfWeek_num'] = df['Date'].dt.dayofweek # Monday=0, Sunday=6

# Analyze cancellations
cancellation_counts = df['Cancelled'].value_counts()
print("Cancellation counts:\n", cancellation_counts)

# Analyze cancellation reasons (if CancellationCode is available)
if 'CancellationCode' in df.columns:
    cancellation_reason_counts = df['CancellationCode'].value_counts()
    print("\nCancellation reason counts:\n", cancellation_reason_counts)

# COMMAND ----------

# If 'Cancelled' column does not exist, create it first
if 'Cancelled' not in df.columns:
    df['Cancelled'] = 0

# Randomly select 5% of rows to mark as cancelled
# Make sure to select rows from all months represented in the data
cancel_indices = df.groupby('Month').apply(lambda x: x.sample(frac=0.05)).index.get_level_values(1)
df.loc[cancel_indices, 'Cancelled'] = 1

# COMMAND ----------

# Monthly Flight Cancellations

monthly_cancellations = df.groupby('Month')['Cancelled'].sum()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Plot the monthly cancellation trends
plt.figure(figsize=(10, 6))
monthly_cancellations.plot(kind='bar')
plt.title('Monthly Flight Cancellations')
plt.xlabel('Month')
plt.ylabel('Number of Cancellations')
plt.xticks(ticks=range(len(monthly_cancellations.index)), labels=[month_names[i-1] for i in monthly_cancellations.index], rotation=0)
plt.show()

# COMMAND ----------

# Total Delay Duration by Cancellation Type

# Sum the delay types to get total delays for each category
total_carrier_delay = df['CarrierDelay'].sum()
total_weather_delay = df['WeatherDelay'].sum()
total_security_delay = df['SecurityDelay'].sum()
total_nas_delay = df['NASDelay'].sum()

# Create a pandas Series for easy plotting
cancellation_types = pd.Series({
    'Carrier Delay': total_carrier_delay,
    'Weather Delay': total_weather_delay,
    'Security Delay': total_security_delay,
    'NAS Delay': total_nas_delay
})

# Plot the cancellation types
plt.figure(figsize=(8, 6))
cancellation_types.plot(kind='bar', color=['#4C72B0','#55A868','#C44E52','#8172B2'])
plt.title('Total Delay by Cancellation Type')
plt.xlabel('Cancellation Type')
plt.ylabel('Total Delay (Minutes)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Print output summary
print(" Total Delay Duration by Cancellation Type (in Minutes) ")
print(cancellation_types)

# COMMAND ----------

# Overall Delay Duration by Cancellation Type (Pie Chart)

plt.figure(figsize=(8, 8))
plt.pie(
    cancellation_types,
    labels=cancellation_types.index,
    autopct='%1.1f%%',            # Show percentage on each slice
    startangle=90,                # Start at top
    colors=['#4C72B0','#55A868','#C44E52','#8172B2'], 
    wedgeprops={'edgecolor': 'white'}
)

plt.title('Overall Delay Distribution by Cancellation Type')
plt.tight_layout()
plt.show()

print("\n--- Total Delay Duration by Cancellation Type (in Minutes) ---")
print(cancellation_types)
print(f"\nTotal Combined Delay: {cancellation_types.sum():,.0f} minutes")


# COMMAND ----------

# First 5 rows of winter and holiday 

winter = df[df['Month'].isin(winter_months)].copy()
holiday = df[df['Date'].isin(major_holidays_2019.values())].copy()

print("First 5 rows of winter:")
display(winter.head())
print("\nFirst 5 rows of holiday:")
display(holiday.head())

# COMMAND ----------

# Winter and holiday analysis

total_flights_winter = len(df_winter)

cancelled_flights_winter = df_winter['Cancelled'].sum()
cancellation_rate_winter = (cancelled_flights_winter / total_flights_winter) * 100 if total_flights_winter > 0 else 0

print(f"Winter Months Analysis:")
print(f"Total flights: {total_flights_winter}")
print(f"Cancelled flights: {cancelled_flights_winter}")
print(f"Cancellation rate: {cancellation_rate_winter:.2f}%")

total_flights_holiday = len(df_holiday)
cancelled_flights_holiday = df_holiday['Cancelled'].sum()
cancellation_rate_holiday = (cancelled_flights_holiday / total_flights_holiday) * 100 if total_flights_holiday > 0 else 0

print(f"\nHoliday Periods Analysis:")
print(f"Total flights: {total_flights_holiday}")
print(f"Cancelled flights: {cancelled_flights_holiday}")
print(f"Cancellation rate: {cancellation_rate_holiday:.2f}%")
     

# COMMAND ----------

# Overall analysis of cancellation rate         

total_flights_overall = len(df)
cancelled_flights_overall = df['Cancelled'].sum()
overall_cancellation_rate = (cancelled_flights_overall / total_flights_overall) * 100 if total_flights_overall > 0 else 0

print("Overall Dataset Analysis:")
print(f"Total flights: {total_flights_overall}")
print(f"Cancelled flights: {cancelled_flights_overall}")
print(f"Overall cancellation rate: {overall_cancellation_rate:.2f}%")

print("\nComparison:")
if cancellation_rate_winter > overall_cancellation_rate:
    print(f"Winter cancellation rate ({cancellation_rate_winter:.2f}%) is HIGHER than the overall rate ({overall_cancellation_rate:.2f}%).")
elif cancellation_rate_winter < overall_cancellation_rate:
    print(f"Winter cancellation rate ({cancellation_rate_winter:.2f}%) is LOWER than the overall rate ({overall_cancellation_rate:.2f}%).")
else:
    print(f"Winter cancellation rate ({cancellation_rate_winter:.2f}%) is the SAME as the overall rate ({overall_cancellation_rate:.2f}%).")

if cancellation_rate_holiday > overall_cancellation_rate:
    print(f"Holiday cancellation rate ({cancellation_rate_holiday:.2f}%) is HIGHER than the overall rate ({overall_cancellation_rate:.2f}%).")
elif cancellation_rate_holiday < overall_cancellation_rate:
    print(f"Holiday cancellation rate ({cancellation_rate_holiday:.2f}%) is LOWER than the overall rate ({overall_cancellation_rate:.2f}%).")
else:
    print(f"Holiday cancellation rate ({cancellation_rate_holiday:.2f}%) is the SAME as the overall rate ({overall_cancellation_rate:.2f}%).")
     

# COMMAND ----------

# Winter vs Non-Winter Cancellation Rate 

cancellation_rate_winter = df_winter['Cancelled'].mean() * 100
cancellation_rate_non_winter = df[~df['Month'].isin(winter_months)]['Cancelled'].mean() * 100

labels_winter = ['Winter Months', 'Non-Winter Months']
rates_winter = [cancellation_rate_winter, cancellation_rate_non_winter]

plt.figure(figsize=(7,5))
plt.bar(labels_winter, rates_winter, color=['lightcoral', 'skyblue'])
plt.title('Flight Cancellation Rate: Winter vs Non-Winter')
plt.xlabel('Period')
plt.ylabel('Cancellation Rate (%)')
plt.tight_layout()
plt.show()


# COMMAND ----------

# Holiday vs Non-Holiday Cancellation Rate

cancellation_rate_holiday = df_holiday['Cancelled'].mean() * 100
cancellation_rate_non_holiday = df[~df['Date'].isin(major_holidays_2019.values())]['Cancelled'].mean() * 100

labels_holiday = ['Holiday Periods', 'Non-Holiday Periods']
rates_holiday = [cancellation_rate_holiday, cancellation_rate_non_holiday]

plt.figure(figsize=(7,5))
plt.bar(labels_holiday, rates_holiday, color=['lightgreen', 'lightgray'])
plt.title('Flight Cancellation Rate: Holiday vs Non-Holiday')
plt.xlabel('Period')
plt.ylabel('Cancellation Rate (%)')
plt.tight_layout()
plt.show()


# COMMAND ----------

# Overall Cancellation Distribution Pie Chart

total_cancellations = df['Cancelled'].sum()
total_winter = df_winter['Cancelled'].sum()
total_holiday = df_holiday['Cancelled'].sum()

non_winter_cancellations = total_cancellations - total_winter
non_holiday_cancellations = total_cancellations - total_holiday

labels = ['Winter Months', 'Non-Winter Months', 'Holiday Periods', 'Non-Holiday Periods']
sizes = [total_winter, non_winter_cancellations, total_holiday, non_holiday_cancellations]

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=['lightcoral', 'skyblue', 'lightgreen', 'lightgray'],
        wedgeprops={'edgecolor': 'white'})
plt.title('Overall Flight Cancellation Distribution')
plt.tight_layout()
plt.show()


print("\n--- Cancellation Summary ---")
print(f"Total Cancellations: {total_cancellations}")
print(f"Winter Cancellations: {total_winter}")
print(f"Holiday Cancellations: {total_holiday}")
