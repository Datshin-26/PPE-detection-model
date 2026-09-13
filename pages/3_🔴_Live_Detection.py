import sys
import time
from pathlib import Path

import cv2
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import APP_TITLE, APP_ICON
from utils.detector import PPEDetector

st.set_page_config(page_title=f"Live Detection | {APP_TITLE}", page_icon=APP_ICON, layout="wide")
st.title("🔴 Live Camera Detection")

st.info(
    "Live webcam access requires running Streamlit **locally** (browser-based "
    "webcam capture doesn't work the same way in hosted/cloud environments "
    "without `streamlit-webrtc`). This page uses OpenCV's local camera capture."
)

@st.cache_resource
def load_detector():
    return PPEDetector()

detector = load_detector()
if detector.mode == "demo":
    st.caption("⚠️ Demo mode: PPE compliance is simulated.")

run = st.toggle("Start Camera")
frame_placeholder = st.empty()
metrics_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    stop_button = st.button("Stop")

    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Could not access webcam. Make sure it's connected and not in use elsewhere.")
            break

        annotated, worker_results, summary = detector.detect(frame)

        now = time.time()
        fps = 1 / (now - prev_time) if now != prev_time else 0
        prev_time = now

        frame_placeholder.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True
        )
        with metrics_placeholder.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("FPS", f"{fps:.1f}")
            c2.metric("Persons", summary["worker_count"])
            c3.metric("Compliant", summary["compliant_count"])
            c4.metric("Non-Compliant", summary["non_compliant_count"])

        if stop_button:
            break

    cap.release()
else:
    st.info("Toggle **Start Camera** above to begin live detection.")
