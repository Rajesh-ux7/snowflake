import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Bike Rental Prediction",
    page_icon="🚲",
    layout="wide"
)

# Snowflake connection
conn = st.connection("snowflake")
session = conn.session()

# Load data
evaluation = session.table(
    "BIKE_SHARING_DB.ML_SCHEMA.MODEL_EVALUATION"
).to_pandas()

predictions = session.table(
    "BIKE_SHARING_DB.ML_SCHEMA.BIKE_DASHBOARD_DATA"
).to_pandas()

# TITLE
st.title("🚲 Bike Rental Prediction Dashboard")
st.write("Machine Learning Regression Project using Snowflake")

# Random Forest metrics
rf = evaluation[
    evaluation["MODEL"].str.upper() == "RANDOM FOREST"
].iloc[0]

# KPI CARDS
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Actual Rentals",
    f"{predictions['ACTUAL_CNT'].sum():,.0f}"
)

c2.metric(
    "Predicted Rentals",
    f"{predictions['PREDICTED_CNT'].sum():,.0f}"
)

c3.metric(
    "Random Forest R²",
    f"{rf['R2']:.4f}"
)

c4.metric(
    "RMSE",
    f"{rf['RMSE']:.2f}"
)

st.divider()

# MODEL COMPARISON
st.subheader("🤖 Model Performance")

st.dataframe(
    evaluation,
    use_container_width=True,
    hide_index=True
)

# R2 chart
chart_data = evaluation.set_index("MODEL")[["R2"]]
st.bar_chart(chart_data)

st.divider()

# ACTUAL VS PREDICTED
st.subheader("📈 Actual vs Predicted Rentals")

actual_predicted = predictions[
    ["ACTUAL_CNT", "PREDICTED_CNT"]
].reset_index(drop=True)

st.line_chart(actual_predicted)

st.divider()

# WEATHER
st.subheader("🌦️ Average Rentals by Weather")

weather = (
    predictions
    .groupby("WEATHERSIT")["ACTUAL_CNT"]
    .mean()
)

st.bar_chart(weather)

st.divider()

# MONTHLY RENTALS
st.subheader("📅 Average Rentals by Month")

monthly = (
    predictions
    .groupby("MNTH")["ACTUAL_CNT"]
    .mean()
)

st.line_chart(monthly)

st.divider()

# ERROR
st.subheader("📉 Prediction Errors")

error_data = predictions[
    ["ERROR"]
].reset_index(drop=True)

st.line_chart(error_data)

st.divider()

# DATA
st.subheader("🔍 Prediction Details")

st.dataframe(
    predictions,
    use_container_width=True,
    hide_index=True
)

st.success("Bike Rental ML Dashboard running successfully!")