"""
PPE Detection System — main entry point (Home dashboard).
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px

from config import APP_TITLE, APP_ICON, CUSTOM_WEIGHTS_PATH
from utils.database import init_db, get_summary_stats, fetch_all

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
init_db()

st.title(f"{APP_ICON} {APP_TITLE}")

if not CUSTOM_WEIGHTS_PATH.exists():
    st.warning(
        "**Running in DEMO mode** — no custom-trained weights found at "
        f"`{CUSTOM_WEIGHTS_PATH}`. Detections use a generic person-detector "
        "and simulated PPE compliance so you can explore the full app. "
        "Run `train.py` on a labeled PPE dataset (e.g. from Roboflow) to "
        "switch to real detections.",
        icon="⚠️",
    )

stats = get_summary_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Detections", stats["total_detections"])
col2.metric("Total Workers Detected", stats["total_workers"])
col3.metric("PPE Compliance", f"{stats['compliance_pct']}%")
col4.metric("Today's Detections", stats["today_detections"])

col5, col6 = st.columns(2)
col5.metric("Compliant Workers", stats["compliant_workers"])
col6.metric("Non-Compliant Workers", stats["non_compliant_workers"])

st.divider()

st.subheader("Recent Activity")
records = fetch_all(limit=10)
if records:
    df = pd.DataFrame(records)[
        ["timestamp", "source_type", "filename", "worker_count",
         "compliance_status", "avg_confidence"]
    ]
    st.dataframe(df, use_container_width=True, hide_index=True)

    fig = px.pie(
        names=["Compliant", "Non-Compliant"],
        values=[stats["compliant_workers"], stats["non_compliant_workers"]],
        color=["Compliant", "Non-Compliant"],
        color_discrete_map={"Compliant": "#16a34a", "Non-Compliant": "#dc2626"},
        title="Overall Compliance Split",
        hole=0.45,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No detections yet — head to **Image Detection** or **Video Detection** in the sidebar to get started.")

st.sidebar.success("Select a page above to begin.")
