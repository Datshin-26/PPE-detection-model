import io
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import APP_TITLE, APP_ICON
from utils.detector import PPEDetector
from utils.database import init_db, insert_detection

st.set_page_config(page_title=f"Image Detection | {APP_TITLE}", page_icon=APP_ICON, layout="wide")
init_db()
st.title("📷 Image Detection")

@st.cache_resource
def load_detector():
    return PPEDetector()

detector = load_detector()
if detector.mode == "demo":
    st.caption("⚠️ Demo mode: PPE compliance is simulated. Train a custom model for real detections.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with st.spinner("Running detection..."):
        annotated, worker_results, summary = detector.detect(image_bgr)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detected")
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Workers Found", summary["worker_count"])
    m2.metric("Compliant", summary["compliant_count"])
    m3.metric("Non-Compliant", summary["non_compliant_count"])
    m4.metric("Avg Confidence", f"{summary['avg_confidence']*100:.1f}%")

    if summary["compliance_status"] == "Compliant":
        st.success("✅ All detected workers are PPE-compliant.")
    elif summary["compliance_status"] == "No Workers Detected":
        st.info("No workers detected in this image.")
    else:
        st.error(f"⚠️ Non-compliant workers detected. Missing: {summary['missing_ppe']}")

    # Download annotated image
    result_img = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    st.download_button(
        "⬇️ Download Annotated Image",
        data=buf.getvalue(),
        file_name=f"annotated_{uploaded.name}",
        mime="image/png",
    )

    # Persist to DB
    if st.button("💾 Save to History"):
        insert_detection(
            source_type="image",
            filename=uploaded.name,
            worker_count=summary["worker_count"],
            compliant_count=summary["compliant_count"],
            non_compliant_count=summary["non_compliant_count"],
            missing_ppe=summary["missing_ppe"],
            avg_confidence=summary["avg_confidence"],
            compliance_status=summary["compliance_status"],
        )
        st.success("Saved to Detection History.")
else:
    st.info("Upload a JPG/PNG image to run PPE detection.")
