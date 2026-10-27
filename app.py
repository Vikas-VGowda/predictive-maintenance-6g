import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Predictive Maintenance | Smart Manufacturing",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Predictive Maintenance & Anomaly Detection")
st.caption("6G-Integrated Smart Manufacturing Systems")

@st.cache_data
def load_data():
    data = pd.read_csv("processed_manufacturing_data.csv")
    data["DateTime"] = pd.to_datetime(data["DateTime"], errors="coerce")
    data["Machine_ID"] = data["Machine_ID"].astype(str)
    return data.dropna(subset=["DateTime"])

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Dataset not found. Place processed_manufacturing_data.csv "
        "in the same folder as app.py."
    )
    st.stop()

# Sidebar filters
st.sidebar.header("Dashboard Filters")

machine_options = sorted(df["Machine_ID"].unique().tolist())
selected_machines = st.sidebar.multiselect(
    "Select machines",
    machine_options,
    default=machine_options[:min(5, len(machine_options))]
)

risk_options = ["Low", "Medium", "High"]
selected_risks = st.sidebar.multiselect(
    "Maintenance risk",
    risk_options,
    default=risk_options
)

mode_options = sorted(df["Operation_Mode"].dropna().unique().tolist())
selected_modes = st.sidebar.multiselect(
    "Operation mode",
    mode_options,
    default=mode_options
)

start_date = df["DateTime"].min().date()
end_date = df["DateTime"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(start_date, end_date),
    min_value=start_date,
    max_value=end_date
)

threshold = st.sidebar.slider(
    "Anomaly score alert threshold",
    min_value=float(df["Anomaly_Score"].min()),
    max_value=float(df["Anomaly_Score"].max()),
    value=float(df["Anomaly_Score"].quantile(0.95)),
    step=max(
        float((df["Anomaly_Score"].max() -
               df["Anomaly_Score"].min()) / 1000),
        0.000001
    )
)

# Apply filters
filtered = df.copy()

if selected_machines:
    filtered = filtered[filtered["Machine_ID"].isin(selected_machines)]
else:
    filtered = filtered.iloc[0:0]

if selected_risks:
    filtered = filtered[
        filtered["Maintenance_Risk_Level"].isin(selected_risks)
    ]
else:
    filtered = filtered.iloc[0:0]

if selected_modes:
    filtered = filtered[
        filtered["Operation_Mode"].isin(selected_modes)
    ]
else:
    filtered = filtered.iloc[0:0]

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    filtered = filtered[
        (filtered["DateTime"].dt.date >= date_range[0]) &
        (filtered["DateTime"].dt.date <= date_range[1])
    ]

if filtered.empty:
    st.warning("No records match these filters. Change your selections.")
    st.stop()

# Overview metrics
st.header("1. Predictive Maintenance Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Records", f"{len(filtered):,}")
c2.metric("Machines", filtered["Machine_ID"].nunique())
c3.metric(
    "High-risk records",
    f"{(filtered['Maintenance_Risk_Level'] == 'High').sum():,}"
)
c4.metric(
    "Potential anomalies",
    f"{(filtered['Anomaly_Flag'] == 'Potential Anomaly').sum():,}"
)

left, right = st.columns(2)

risk_counts = (
    filtered["Maintenance_Risk_Level"]
    .value_counts()
    .reindex(["Low", "Medium", "High"], fill_value=0)
    .rename_axis("Risk")
    .reset_index(name="Records")
)

with left:
    fig = px.bar(
        risk_counts,
        x="Risk",
        y="Records",
        color="Risk",
        title="Risk Distribution",
        category_orders={"Risk": ["Low", "Medium", "High"]}
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    mode_counts = (
        filtered["Operation_Mode"]
        .value_counts()
        .rename_axis("Operation Mode")
        .reset_index(name="Records")
    )
    fig = px.pie(
        mode_counts,
        names="Operation Mode",
        values="Records",
        title="Operating Mode Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

# Machine anomaly analysis
st.header("2. Machine Anomaly Dashboard")

machine = st.selectbox(
    "Choose a machine for detailed analysis",
    sorted(filtered["Machine_ID"].unique())
)

machine_df = (
    filtered[filtered["Machine_ID"] == machine]
    .sort_values("DateTime")
)

sensor_options = [
    "Temperature_C",
    "Vibration_Hz",
    "Power_Consumption_kW",
    "Network_Latency_ms",
    "Packet_Loss_%",
    "Error_Rate_%"
]

sensor = st.selectbox("Select sensor", sensor_options)

fig = px.line(
    machine_df,
    x="DateTime",
    y=sensor,
    title=f"{sensor} over time — Machine {machine}",
    markers=False
)
st.plotly_chart(fig, use_container_width=True)

fig = px.scatter(
    machine_df,
    x="DateTime",
    y="Anomaly_Score",
    color="Anomaly_Flag",
    hover_data=["Temperature_C", "Vibration_Hz", "Operation_Mode"],
    title="Anomaly Score over Time"
)
fig.add_hline(
    y=threshold,
    line_dash="dash",
    annotation_text="Selected alert threshold"
)
st.plotly_chart(fig, use_container_width=True)

# Maintenance alert panel
st.header("3. Maintenance Alert Panel")

alerts = filtered[
    (filtered["Anomaly_Score"] >= threshold) |
    (filtered["Maintenance_Risk_Level"] == "High")
].copy()

alert_columns = [
    "DateTime", "Machine_ID", "Operation_Mode",
    "Temperature_C", "Vibration_Hz",
    "Power_Consumption_kW", "Error_Rate_%",
    "Anomaly_Score", "Maintenance_Risk_Level"
]

alerts = alerts.sort_values("Anomaly_Score", ascending=False)

st.write(f"Records requiring investigation: {len(alerts):,}")
st.dataframe(
    alerts[alert_columns].head(500),
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download alert records as CSV",
    data=alerts[alert_columns].to_csv(index=False).encode("utf-8"),
    file_name="maintenance_alerts.csv",
    mime="text/csv"
)

# Historical analysis
st.header("4. Historical Risk Analysis")

history = (
    filtered.set_index("DateTime")
    .groupby("Maintenance_Risk_Level")
    .resample("D")
    .size()
    .rename("Records")
    .reset_index()
)

fig = px.line(
    history,
    x="DateTime",
    y="Records",
    color="Maintenance_Risk_Level",
    title="Daily Records by Risk Level"
)
st.plotly_chart(fig, use_container_width=True)

daily_sensor = (
    filtered.set_index("DateTime")
    .resample("D")[["Temperature_C", "Vibration_Hz"]]
    .mean()
    .reset_index()
)

fig = px.line(
    daily_sensor,
    x="DateTime",
    y=["Temperature_C", "Vibration_Hz"],
    title="Daily Average Sensor Trends"
)
st.plotly_chart(fig, use_container_width=True)

# Data download
st.header("Dataset Export")

st.download_button(
    "Download filtered dataset",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_manufacturing_data.csv",
    mime="text/csv"
)

st.caption(
    "Risk categories are relative anomaly indicators, not calibrated "
    "failure probabilities. Verify alerts against machine specifications "
    "and maintenance records before taking action."
)
