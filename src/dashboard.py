import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

st.title("Recovery Risk Dashboard")

st.write("Enter workout data for multiple days:")

num_days = st.number_input("Number of days", min_value=1, max_value=30, value=5)

workouts = []

for i in range(num_days):
    st.subheader(f"Day {i+1}")
    rolling_volume_7d = st.number_input(f"Rolling Volume 7d (Day {i+1})", value=150000.0, key=f"v7_{i}")
    rolling_volume_14d = st.number_input(f"Rolling Volume 14d (Day {i+1})", value=300000.0, key=f"v14_{i}")
    sleep_avg_7d = st.number_input(f"Sleep Avg 7d (Day {i+1})", value=7.0, key=f"savg_{i}")
    sleep_std_7d = st.number_input(f"Sleep Std 7d (Day {i+1})", value=0.3, key=f"sstd_{i}")
    days_since_rest = st.number_input(f"Days Since Rest (Day {i+1})", value=3, key=f"dsr_{i}")
    acwr = st.number_input(f"ACWR (Day {i+1})", value=1.2, key=f"acwr_{i}")
    
    workouts.append({
        "rolling_volume_7d": rolling_volume_7d,
        "rolling_volume_14d": rolling_volume_14d,
        "sleep_avg_7d": sleep_avg_7d,
        "sleep_std_7d": sleep_std_7d,
        "days_since_rest": days_since_rest,
        "acwr": acwr
    })

if st.button("Predict Recovery Risk"):
    response = requests.post("http://127.0.0.1:8000/predict_batch", json={"data": workouts})
    results = response.json()["results"]

    df_results = pd.DataFrame(results)
    df_results["day"] = range(1, len(df_results)+1)

    st.write(df_results)

    plt.figure(figsize=(10,4))
    plt.plot(df_results["day"], df_results["risk_probability"], marker='o')
    plt.title("Recovery Risk Probability Over Time")
    plt.xlabel("Day")
    plt.ylabel("Risk Probability")
    plt.ylim(0,1)
    plt.grid(True)
    st.pyplot(plt)
