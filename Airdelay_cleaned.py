# Databricks notebook source

import pandas as pd

# Replace 'your_file.csv' with the path to your dataset
df = pd.read_csv("/Volumes/workspace/default/airdelay/Flight_delay.csv")

# COMMAND ----------

# displays first 5 rows of the DataFrame 
df.head()

# COMMAND ----------

# displays last 10 rows of the DataFrame
df.tail(10)

# COMMAND ----------

 # returns total number of elements (rows × columns) in the DataFrame
 df.size

# COMMAND ----------

 # shows data type of each column in the DataFrame
 df.dtypes

# COMMAND ----------

# returns the list of column labels in the DataFrame
df.columns


# COMMAND ----------

# returns the number of rows and columns in the DataFrame
df.shape

# COMMAND ----------

 # generates summary statistics (count, mean, std, min, quartiles, max) for numerical columns
 df.describe()

# COMMAND ----------

# displays summary of DataFrame (index, columns, non-null counts, and data types)
df.info()

# COMMAND ----------

# remove duplicate rows
df_no_duplicates = df.drop_duplicates()

# check shape before and after removing duplicates
print("Before removing duplicates:", df.shape)  
print("After removing duplicates :", df_no_duplicates.shape)


# COMMAND ----------

# take 1% random sample of the data
sample_df = df.sample(frac=0.01, random_state=42)

# show first 5 rows of the sample
print(sample_df.head())


# COMMAND ----------

# random sampling
sample_df = pd.read_csv("/Volumes/workspace/default/airdelay/Flight_delay.csv", nrows=100000)  # first 100k rows


# COMMAND ----------

#Memory Optimization
def optimize_airfly_memory(path):
    """
    Load AirFly dataset with optimized dtypes for memory usage.
    """
    # Define numeric downcast mapping
    dtype_map = {
        "DayOfWeek": "int8",
        "DepTime": "int32",
        "ArrTime": "int32",
        "CRSArrTime": "int32",
        "FlightNum": "int32",
        "ActualElapsedTime": "float32",   # may contain nulls after cleaning
        "CRSElapsedTime": "float32",
        "AirTime": "float32",
        "ArrDelay": "float32",
        "DepDelay": "float32",
        "Distance": "int32",
        "TaxiIn": "float32",
        "TaxiOut": "float32",
        "Cancelled": "int8",
        "Diverted": "int8",
        "CarrierDelay": "float32",
        "WeatherDelay": "float32",
        "NASDelay": "float32",
        "SecurityDelay": "float32",
        "LateAircraftDelay": "float32"
    }

    # Read CSV with dtype mapping (for numeric columns)
    data = pd.read_csv("/Volumes/workspace/default/airdelay/Flight_delay.csv", dtype=dtype_map, low_memory=True)

    # Convert object columns to category
    categorical_cols = [
        "Date", "UniqueCarrier", "Airline", "TailNum",
        "Origin", "Org_Airport", "Dest", "Dest_Airport",
        "CancellationCode"
    ]
    for col in categorical_cols:
        data[col] = data[col].astype("category")

    return data


def report_memory(data):
    """
    Prints memory usage of dataframe in MB by column.
    """
    mem = data.memory_usage(deep=True) / 1024**2
    mem_data = mem.reset_index()
    mem_data.columns = ["Column", "Memory_MB"]
    print("Total Memory: {:.2f} MB".format(mem.sum()))
    return mem_data


# COMMAND ----------

path = "/Volumes/workspace/default/airdelay/Flight_delay.csv"

print("🔹 Loading original dataset...")
df_raw = pd.read_csv(path)
print("Original memory usage:")
print(report_memory(df_raw))

# COMMAND ----------


path = "/Volumes/workspace/default/Airdelay/Flight_delay.csv"
print("\n🔹 Loading optimized dataset...")
df_opt = optimize_airfly_memory(path)
print("Optimized memory usage:")
print(report_memory(df_opt))

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# Handling Missing Values




#1 Fill categorical text fields with "Unknown" for missing values
categorical_cols = [
    "Date", "UniqueCarrier", "Airline", "TailNum",
    "Origin", "Org_Airport", "Dest", "Dest_Airport",
    "CancellationCode"
]

for col in categorical_cols:
    if df[col].isnull().any():
        df[col] = df[col].fillna("Unknown")

# COMMAND ----------

#2 Replace NaN with 0 for numeric delay fields
delay_cols = [
    "ArrDelay", "DepDelay", "CarrierDelay", "WeatherDelay",
    "NASDelay", "SecurityDelay", "LateAircraftDelay"
]
df[delay_cols] = df[delay_cols].fillna(0)

# COMMAND ----------

#3 Fill missing values in elapsed time or airtime columns with median
elapsed_cols = ["ActualElapsedTime", "CRSElapsedTime", "AirTime"]
for col in elapsed_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# COMMAND ----------

#4 If 'CancellationCode' is null, it means the flight was NOT cancelled
df['CancellationCode'] = df['CancellationCode'].fillna("NotCancelled")

# COMMAND ----------

# Check results 
print(df[delay_cols + ["Cancelled", "CancellationCode"]].isnull().sum())

# COMMAND ----------

 #5 If still missing values remain → drop those rows (safe clean-up)
df = df.dropna()

# COMMAND ----------

#Missing value after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# COMMAND ----------

# First, convert the 'Date' column to datetime type
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

# COMMAND ----------

#Convert 'DepTime' (HHMM integer) to a 4-digit string with leading zeros
df["DepTime"] = df["DepTime"].apply(lambda x: f"{int(x):04d}" if pd.notnull(x) else "0000")

# COMMAND ----------

 #Extract hour and minute from 'DepTime' string
 df["DepHour"] = df["DepTime"].str[:2].astype(int)
df["DepMinute"] = df["DepTime"].str[2:].astype(int)

# COMMAND ----------

# Combine 'Date', 'DepHour', and 'DepMinute' into a single datetime column 'DepDatetime'

df["DepDatetime"] = df.apply(
    lambda row: pd.Timestamp(
        year=row["Date"].year,
        month=row["Date"].month,
        day=row["Date"].day,
        hour=min(max(row["DepHour"], 0), 23),
        minute=min(max(row["DepMinute"], 0), 59)
    ),
    axis=1
)

# COMMAND ----------

# Extract the month number from departure datetime (1–12)
df["Month"] = df["DepDatetime"].dt.month

# Extract the day name of the week (Monday, Tuesday, etc.)
df["DayOfWeek"] = df["DepDatetime"].dt.day_name()

# Extract the hour of departure
df["Hour"] = df["DepDatetime"].dt.hour

# COMMAND ----------

# Create a route string by combining Origin and Destination (e.g., ATL-LAX)
df["Route"] = df["Origin"] + "-" + df["Dest"]

# COMMAND ----------


# Display the first few rows of the relevant columns to verify the derived features
display(df[["Date", "DepTime", "DepDatetime", "Month", "DayOfWeek", "Hour", "Route"]])

# COMMAND ----------

# Print the minimum value from the 'Distance' column
print("Minimum Distance:", df['Distance'].min())

# COMMAND ----------

# Print the maximum value from the 'Distance' column
print("Maximum Distance:", df['Distance'].max())

# COMMAND ----------

# Print the average (mean) value of the 'Distance' column
print("Average Distance:", df['Distance'].mean())

# COMMAND ----------

display(df.loc[df['Distance'] > 1000, ['FlightNum', 'Origin', 'Dest', 'Distance']])

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------


df.to_csv("/Volumes/workspace/default/airdelay/Flight_delay_cleaned.csv", index=False)