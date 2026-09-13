import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import APP_TITLE, APP_ICON
from utils.database import init_db, fetch_all

st.set_page_config(page_title=f"Analytics | {APP_TITLE}", page_icon=APP_ICON, layout="wide")
init_db()
st.title("📊 Analytics Dashboard")

records = fetch_all()
if not records:
    st.info("No data yet. Run some detections first (Image or Video Detection pages).")
    st.stop()

df = pd.DataFrame(records)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date

col1, col2 = st.columns(2)

with col1:
    st.subheader("Daily Detection Count")
    daily = df.groupby("date").size().reset_index(name="count")
    fig = px.bar(daily, x="date", y="count")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Compliance Status Breakdown")
    status_counts = df["compliance_status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    fig = px.pie(status_counts, names="status", values="count", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Compliance Over Time")
    trend = df.groupby("date").agg(
        compliant=("compliant_count", "sum"),
        non_compliant=("non_compliant_count", "sum"),
    ).reset_index()
    trend["compliance_pct"] = (
        trend["compliant"] / (trend["compliant"] + trend["non_compliant"]).replace(0, 1) * 100
    )
    fig = px.line(trend, x="date", y="compliance_pct", markers=True,
                   labels={"compliance_pct": "Compliance %"})
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Confidence Score Distribution")
    fig = px.histogram(df[df["avg_confidence"] > 0], x="avg_confidence", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Missing PPE Frequency")
missing_counter = Counter()
for row in df["missing_ppe"].dropna():
    for item in str(row).split(","):
        item = item.strip()
        if item:
            missing_counter[item] += 1

if missing_counter:
    missing_df = pd.DataFrame(missing_counter.items(), columns=["ppe_item", "count"]).sort_values("count", ascending=False)
    fig = px.bar(missing_df, x="ppe_item", y="count")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("No non-compliance records yet.")

st.subheader("Monthly Compliance Trend")
df["month"] = df["timestamp"].dt.to_period("M").astype(str)
monthly = df.groupby("month").agg(
    compliant=("compliant_count", "sum"),
    non_compliant=("non_compliant_count", "sum"),
).reset_index()
monthly["compliance_pct"] = (
    monthly["compliant"] / (monthly["compliant"] + monthly["non_compliant"]).replace(0, 1) * 100
)
fig = px.bar(monthly, x="month", y="compliance_pct", labels={"compliance_pct": "Compliance %"})
st.plotly_chart(fig, use_container_width=True)
