# Load CSV into a pandas DataFrame

import pandas as pd
file_path = "/Volumes/airfly_workspace/default/airfly_insights/airfly _raw_data.csv"
df = pd.read_csv(file_path)
print(df.head())

# COMMAND ----------

# View column names, data types, and non-null counts

print("Dataset Info:")
print(df.info())

# COMMAND ----------

# Get only column names and types

print("\nColumn Names and Data Types:")
print(df.dtypes)


# COMMAND ----------

# Check shape/ size

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


# COMMAND ----------

# Check null values

print("Null values per column:\n", df.isnull().sum())


# COMMAND ----------

# Random Sampling by Fraction (Sample 10% of the data randomly)

sample_frac = df.sample(frac=0.1, random_state=42)
print("Random sample (10%):")
print(sample_frac.head())


# COMMAND ----------

# Random Sampling by Number of Rows (Sample 100 rows randomly)

sample_n = df.sample(n=100, random_state=42)
print("Random sample (100 rows):")
print(sample_n.head())


# COMMAND ----------

# Head Sampling (First 10 rows)

head_sample = df.head(10)
print("First 10 rows:")
print(head_sample)

# Tail Sampling (Last 10 rows)

tail_sample = df.tail(10)
print("Last 10 rows:")
print(tail_sample)


# COMMAND ----------

# Reduce Numeric Memory (Downcast integer and float columns)

for col in df.select_dtypes(include=['int', 'float']).columns:
    if df[col].dtype == 'int64':
        df[col] = pd.to_numeric(df[col], downcast='integer')
    elif df[col].dtype == 'float64':
        df[col] = pd.to_numeric(df[col], downcast='float')


# COMMAND ----------

# Convert Object Columns to Category (Convert object columns with few unique values to 'category')

for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() / len(df) < 0.5:
        df[col] = df[col].astype('category')


# COMMAND ----------

# Memory usage before optimization

print("Memory usage BEFORE optimization:")
print(df.memory_usage(deep=True))

# Memory usage after optimization

print("\nMemory usage AFTER optimization:")
print(df.memory_usage(deep=True))


# COMMAND ----------

# Displaying the dataframe
display(df)
