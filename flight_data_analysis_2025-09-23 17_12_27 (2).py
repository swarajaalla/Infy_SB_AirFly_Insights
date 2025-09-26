# Databricks notebook source
# MAGIC %md
# MAGIC DayOfWeek → Day of the week (1 = Monday, …, 7 = Sunday).
# MAGIC
# MAGIC Date → Flight date.
# MAGIC
# MAGIC DepTime → Actual departure time (local, HHMM format, e.g., 1345 = 1:45 PM).
# MAGIC
# MAGIC ArrTime → Actual arrival time (local, HHMM format).
# MAGIC
# MAGIC CRSArrTime → Scheduled arrival time (as per airline’s published schedule).
# MAGIC
# MAGIC UniqueCarrier → Carrier code (e.g., "AA" = American Airlines, "DL" = Delta).
# MAGIC
# MAGIC Airline → Airline name (full name of UniqueCarrier).
# MAGIC
# MAGIC FlightNum → Flight number assigned by the airline.
# MAGIC
# MAGIC TailNum → Aircraft registration number (plane’s unique ID).
# MAGIC
# MAGIC ActualElapsedTime → Actual elapsed flight time in minutes (ArrTime − DepTime).
# MAGIC
# MAGIC CRSElapsedTime → Scheduled elapsed flight time in minutes.
# MAGIC
# MAGIC AirTime → Actual time spent flying in minutes (excluding taxiing).
# MAGIC
# MAGIC ArrDelay → Arrival delay in minutes (early arrivals = negative values).
# MAGIC
# MAGIC DepDelay → Departure delay in minutes.
# MAGIC
# MAGIC Origin → Origin airport code (e.g., "ATL" = Atlanta).
# MAGIC
# MAGIC Org_Airport → Origin airport full name (if available).
# MAGIC
# MAGIC Dest → Destination airport code (e.g., "LAX" = Los Angeles).
# MAGIC
# MAGIC Dest_Airport → Destination airport full name (if available).
# MAGIC
# MAGIC Distance → Distance between origin and destination airports (miles).
# MAGIC
# MAGIC TaxiIn → Taxi-in time in minutes (arrival gate from runway).
# MAGIC
# MAGIC TaxiOut → Taxi-out time in minutes (departure runway from gate).
# MAGIC
# MAGIC Cancelled → 1 = Flight cancelled, 0 = Not cancelled.
# MAGIC
# MAGIC CancellationCode → Reason for cancellation:
# MAGIC
# MAGIC "A" = Carrier
# MAGIC
# MAGIC "B" = Weather
# MAGIC
# MAGIC "C" = NAS (National Airspace System)
# MAGIC
# MAGIC "D" = Security
# MAGIC
# MAGIC Diverted → 1 = Flight diverted, 0 = Not diverted.
# MAGIC
# MAGIC CarrierDelay → Delay minutes due to airline (maintenance, crew, etc.).
# MAGIC
# MAGIC WeatherDelay → Delay minutes due to weather.
# MAGIC
# MAGIC NASDelay → Delay minutes due to National Airspace System (air traffic control, heavy traffic, etc.).
# MAGIC
# MAGIC SecurityDelay → Delay minutes due to security (e.g., evacuation, screening).
# MAGIC
# MAGIC LateAircraftDelay → Delay minutes because the aircraft arrived late from a previous flight.

# COMMAND ----------

import pandas as pd

# Replace 'your_file.csv' with the path to your dataset
data = pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay.csv")

# Display first 5 rows
print(data.head())


# COMMAND ----------





# Get all column names
columns = data.columns.tolist()

print("Columns in the dataset:")
print(columns)


# COMMAND ----------

print(data.head())

# COMMAND ----------

# Set Pandas option to display ALL columns
pd.set_option('display.max_columns', None)

# Now info() and describe() will show everything
print("\n--- Dataset Info ---")
data.info()




# COMMAND ----------

print("\n--- Dataset Description ---")
print(data.describe(include='all'))  # include='all' adds categorical columns too

# COMMAND ----------

# Remove duplicate rows
data_no_duplicates = data.drop_duplicates()
print("Before removing duplicates:", data.shape)
print("After removing duplicates:", data_no_duplicates.shape)

# COMMAND ----------

sample_df = data.sample(frac=0.01, random_state=42)  # 1% random sample
print(sample_df.head())

# COMMAND ----------

#random sampling

sample_df = pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay.csv", nrows=100000)  # first 100k rows

# COMMAND ----------

sample_df = data.groupby("Airline", group_keys=False).apply(lambda x: x.sample(frac=0.01))

# COMMAND ----------

# MAGIC %md
# MAGIC Key Optimizations
# MAGIC
# MAGIC Integers downcasted (int64 → int8/16/32)
# MAGIC Example: DayOfWeek, Cancelled, Diverted can fit into int8.
# MAGIC
# MAGIC Floats used for delay columns since they can contain NaN.
# MAGIC
# MAGIC Objects → Categories (Airline, Origin, Dest, etc.), since they repeat values.
# MAGIC
# MAGIC Memory report before & after shows how much RAM was saved.

# COMMAND ----------

#Memory Management


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
    data = pd.read_csv("/Volumes/workspace/default/airlines/Flight_delay.csv", dtype=dtype_map, low_memory=True)

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

# Example usage
path = "/Volumes/workspace/default/airlines/Flight_delay.csv"

print("🔹 Loading original dataset...")
df_raw = pd.read_csv(path)
print("Original memory usage:")
print(report_memory(df_raw))


# COMMAND ----------

path = "/Volumes/workspace/default/airlines/Flight_delay.csv"
print("\n🔹 Loading optimized dataset...")
df_opt = optimize_airfly_memory(path)
print("Optimized memory usage:")
print(report_memory(df_opt))

# COMMAND ----------

# ---- General Missing Value Check ----
print("Missing values per column:")
print(data.isnull().sum())


# COMMAND ----------

# ---- Handling Missing Values ----

# 1. For categorical text fields → fill with "Unknown" or most frequent value
categorical_cols = ["Org_Airport", "Dest_Airport", "CancellationCode", "TailNum"]
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].fillna("Unknown")   # or use df[col].mode()[0]

# COMMAND ----------

# 2. For numeric fields related to delays → replace NaN with 0 (means no delay recorded)
delay_cols = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
for col in delay_cols:
    if col in data.columns:
        data[col] = data[col].fillna(0)

# COMMAND ----------

# 3. For elapsed time or airtime → fill with median (less skewed than mean)
time_cols = ["ActualElapsedTime", "AirTime"]
for col in time_cols:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].median())

# COMMAND ----------

# ---- Cancellation column ----
# If 'CancellationCode' is null, it means the flight was NOT cancelled
if "CancellationCode" in data.columns:
    data["CancellationCode"] = data["CancellationCode"].fillna("Not Cancelled")

# COMMAND ----------

# ---- Check results ----
print(data[delay_cols + ["Cancelled", "CancellationCode"]].isnull().sum())

# COMMAND ----------

# 4. If still missing values remain → drop those rows (safe clean-up)
df = data.dropna()

# COMMAND ----------

# ---- Verify again ----
print("\nMissing values after cleaning:")
print(data.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC Delay columns → Null means no delay logged → safely replaced with 0.
# MAGIC
# MAGIC CancellationCode → Null means flight wasn’t cancelled → replaced with "Not Cancelled".
# MAGIC
# MAGIC Cancelled column (0/1) → already numeric, but you can also double-check consistency (if Cancelled=0, then CancellationCode should be "Not Cancelled").

# COMMAND ----------

# ---- Convert 'Date' + 'DepTime' into a proper datetime ----
# Step 1: Ensure Date is datetime
data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")



# COMMAND ----------

# Step 2: Convert DepTime (HHMM int) into string with leading zeros
data["DepTime"] = data["DepTime"].apply(lambda x: f"{int(x):04d}" if pd.notnull(x) else "0000")


# COMMAND ----------

# Step 3: Extract hour + minute
data["DepHour"] = data["DepTime"].str[:2].astype(int)
data["DepMinute"] = data["DepTime"].str[2:].astype(int)

# COMMAND ----------

# Step 4: Combine Date + Time into full datetime
data["DepDatetime"] = data["Date"] + pd.to_timedelta(data["DepHour"], unit="h") + pd.to_timedelta(data["DepMinute"], unit="m")


# COMMAND ----------

# ---- Derived Features ----
data["Month"] = data["DepDatetime"].dt.month          # Month number (1–12)
data["DayOfWeek"] = data["DepDatetime"].dt.day_name() # Monday, Tuesday...
data["Hour"] = data["DepDatetime"].dt.hour            # Hour of departure
data["Route"] = data["Origin"] + "-" + df["Dest"]     # Route string (e.g., ATL-LAX)

# COMMAND ----------

# ---- Check results ----
print(data[["Date", "DepTime", "DepDatetime", "Month", "DayOfWeek", "Hour", "Route"]].head())

# COMMAND ----------

# DBTITLE 1,formatting date and time columns
# STEP 1: Convert Date column
data["Date"] = pd.to_datetime(data["Date"], errors="coerce")  

# COMMAND ----------

# STEP 2: Define helper function
# ===============================
def convert_hhmm_to_time(series):
    """
    Convert HHMM (e.g., 930 = 09:30) integers into timedelta objects.
    - Handles missing values by replacing with "0000".
    - Ensures proper zero-padding (e.g., 45 -> "0045").
    """
    # Fill missing values with 0, convert to string, pad to 4 digits
    series = series.fillna(0).astype(int).astype(str).str.zfill(4)
    
    # Extract hours and minutes separately
    hours = series.str[:2].astype(int)
    minutes = series.str[2:].astype(int)
    
    # Convert to timedelta (hours + minutes)
    return pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")

# COMMAND ----------

# STEP 3: Apply conversion to time fields
# ===============================
data["DepTime_obj"] = convert_hhmm_to_time(data["DepTime"])
data["ArrTime_obj"] = convert_hhmm_to_time(data["ArrTime"])
data["CRSArrTime_obj"] = convert_hhmm_to_time(data["CRSArrTime"])

# COMMAND ----------

# STEP 4: Combine Date + Time
# Creates full datetime columns for departure and arrival
data["DepDateTime"] = data["Date"] + data["DepTime_obj"]
data["ArrDateTime"] = data["Date"] + data["ArrTime_obj"]
data["CRSArrDateTime"] = data["Date"] + data["CRSArrTime_obj"]

# COMMAND ----------

# STEP 5: Handle overnight flights (optional but recommended)
# If arrival time is earlier than departure time, it means arrival is on the next day
data.loc[data["ArrDateTime"] < data["DepDateTime"], "ArrDateTime"] += pd.Timedelta(days=1)
data.loc[data["CRSArrDateTime"] < data["DepDateTime"], "CRSArrDateTime"] += pd.Timedelta(days=1)

# COMMAND ----------

# STEP 6: Clean up temporary columns
data.drop(columns=["DepTime_obj", "ArrTime_obj", "CRSArrTime_obj"], inplace=True)


# COMMAND ----------

# Final check
print(data[["Date", "DepTime", "ArrTime", "CRSArrTime", 
          "DepDateTime", "ArrDateTime", "CRSArrDateTime"]].head())

# COMMAND ----------

# ---- 5. Save the preprocessed dataset ----
data.to_csv("flights_preprocessed.csv", index=False)
print("✅ Preprocessed dataset saved as flights_preprocessed.csv")

# COMMAND ----------

