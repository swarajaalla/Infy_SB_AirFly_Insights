# Infy_SB_AirFly_Insights
# AirFly Insights ✈️

#Data Visualization & Analysis of Airline Operations

## 📌 Project Overview

AirFly Insights is a data analysis and visualization project focused on large-scale airline flight data. The goal is to uncover **operational trends, delay patterns, cancellation insights, and route-level performance**. This project was completed as part of the Infosys SpringBoard internship program under the mentorship of **Mr. Swaraj Aalla**.

## 📂 Dataset

* Source: Kaggle Airlines Flights Dataset (sample of 100,000 records used)
* Coverage: 2019–2023
* Size: 100,000 rows
* Features: Flight dates, carriers, origin & destination airports, delays, cancellations, etc.

## 🔧 Tech Stack

* Languages: Python
* Libraries: Pandas, NumPy, Matplotlib, Seaborn, Plotly, Folium
* Environment: Jupyter Notebook

## 🛠 Data Cleaning Steps

1. Loaded CSV with Pandas (`read_csv`) using `low_memory=True`
2. Schema review with `info()`, `describe()`, and null-value analysis
3. Converted date fields with `pd.to_datetime` and derived features (Year, Month, DayOfWeek)
4. Coerced delay columns to numeric (`pd.to_numeric`)
5. Handled cancellations via `CANCELLED` flag (0/1)
6. Managed missing values (dropped critical IDs, filled others as required)
7. Removed duplicate rows
8. Engineered new features: route (ORIGIN → DEST), delay >15min flags
9. Optimized column types (converted categorical fields to `category`)
10. Exported a cleaned dataset snapshot for reproducibility

## 📊 Key Metrics

* Total flights analyzed: 100,000
* Unique carriers: 18
* Unique origin airports: 372
* Unique destination airports: 376
* Average departure delay: 10.18 minutes
* Average arrival delay: 4.35 minutes
* Percentage of flights delayed >15 min: ~17%
* Cancellation rate: 2.63%

## 📈 Insights & Recommendations

* Carrier Delays: Delta & American showed higher average delays; Southwest managed better performance at scale.
* Routes: High-frequency routes (e.g., LGA–ORD, JFK–LAX) showed concentrated delays.
* Airports: Major hubs like ATL, DFW, ORD contributed most to congestion.
* Recommendations:

  * Optimize turnaround processes
  * Implement seasonal readiness (especially for winter disruptions)
  * Smooth schedules at congested hubs
  * Deploy dashboards to monitor KPIs in real time

## 📑 Conclusion

Data-driven insights from airline operations can improve efficiency, reduce delays, and enhance passenger satisfaction. This project demonstrates how structured data cleaning, visualization, and analysis can support evidence-based decision-making for airlines and airports.
