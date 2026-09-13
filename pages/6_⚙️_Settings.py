import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    APP_TITLE, APP_ICON, CUSTOM_WEIGHTS_PATH, CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD, MANDATORY_PPE, PPE_CLASS_NAMES,
)

st.set_page_config(page_title=f"Settings | {APP_TITLE}", page_icon=APP_ICON, layout="wide")
st.title("⚙️ Settings")

st.subheader("Model Status")
if CUSTOM_WEIGHTS_PATH.exists():
    st.success(f"Custom model loaded from `{CUSTOM_WEIGHTS_PATH}`")
else:
    st.warning(
        f"No custom model found at `{CUSTOM_WEIGHTS_PATH}`. Running in DEMO mode "
        "with simulated PPE compliance. Run `python train.py` after preparing a "
        "labeled dataset to enable real detections."
    )

st.subheader("Current Configuration")
st.json({
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "iou_threshold": IOU_THRESHOLD,
    "mandatory_ppe": MANDATORY_PPE,
    "class_names": PPE_CLASS_NAMES,
})

st.caption(
    "To change thresholds or class names, edit `config.py` directly — "
    "keeping configuration in one file avoids values drifting out of sync "
    "across pages."
)
