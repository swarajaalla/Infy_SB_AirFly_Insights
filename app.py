import streamlit as st
import pandas as pd
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Flight Dashboard", layout="wide")
st.title("Flight Dashboard (Week 7 & 8)")

ZIP_FILE = "Flight_delay_final.zip"
EXTRACT_FOLDER = "data"

# --- Step 1: Unzip the dataset ---
if not os.path.exists(EXTRACT_FOLDER):
    os.makedirs(EXTRACT_FOLDER)

try:
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)
    st.success(f"{ZIP_FILE} unzipped successfully into {EXTRACT_FOLDER}")
except FileNotFoundError:
    st.error(f"{ZIP_FILE} not found in repository!")
except zipfile.BadZipFile:
    st.error(f"{ZIP_FILE} is not a valid zip file!")

# --- Step 2: Load CSV automatically ---
csv_files = [f for f in os.listdir(EXTRACT_FOLDER) if f.endswith('.csv')]
if len(csv_files) == 0:
    st.warning("No CSV file found in the unzipped folder!")
else:
    csv_path = os.path.join(EXTRACT_FOLDER, csv_files[0])
    df = pd.read_csv(csv_path)
    st.success(f"Loaded CSV: {csv_files[0]}")
    st.write("Data shape:", df.shape)

# --- Step 3: Preview Data ---
st.subheader("Dataset Preview")
st.dataframe(df.head())

# --- Step 4: Example Visuals ---
if 'df' in locals():
    # Create Route column if missing
    if 'Route' not in df.columns:
        df['Route'] = df['Origin'] + '-' + df['Dest']
    
    # Top 10 Routes by Avg Arrival Delay
    if 'ArrDelay' in df.columns:
        st.subheader("Top 10 Routes by Avg Arrival Delay")
        top_routes = df.groupby('Route')['ArrDelay'].mean().sort_values(ascending=False).head(10)
        st.bar_chart(top_routes)
    
    # Cancellations by Airline
    if 'Cancelled' in df.columns and 'Airline' in df.columns:
        st.subheader("Total Cancellations by Airline")
        airline_cancel = df.groupby('Airline')['Cancelled'].sum().sort_values(ascending=False)
        st.bar_chart(airline_cancel)
    
    # Heatmap of Arrival Delays (Top 15 Routes)
    st.subheader("Arrival Delay Heatmap (Top 15 Routes)")
    top15_routes = df['Route'].value_counts().head(15).index
    heat_df = df[df['Route'].isin(top15_routes)]
    pivot_table = heat_df.pivot_table(values='ArrDelay', index='Route', columns='Airline', aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)
