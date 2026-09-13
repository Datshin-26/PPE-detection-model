import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import APP_TITLE, APP_ICON
from utils.detector import PPEDetector
from utils.database import init_db, insert_detection

st.set_page_config(page_title=f"Video Detection | {APP_TITLE}", page_icon=APP_ICON, layout="wide")
init_db()
st.title("🎥 Video Detection")

@st.cache_resource
def load_detector():
    return PPEDetector()

detector = load_detector()
if detector.mode == "demo":
    st.caption("⚠️ Demo mode: PPE compliance is simulated. Train a custom model for real detections.")

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
frame_skip = st.slider("Process every Nth frame (higher = faster)", 1, 10, 3)

if uploaded is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
    tfile.write(uploaded.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    progress = st.progress(0, text="Processing video...")
    frame_placeholder = st.empty()

    frame_idx = 0
    all_worker_counts, all_compliant, all_non_compliant, all_conf = [], [], [], []
    last_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            annotated, worker_results, summary = detector.detect(frame)
            last_annotated = annotated
            all_worker_counts.append(summary["worker_count"])
            all_compliant.append(summary["compliant_count"])
            all_non_compliant.append(summary["non_compliant_count"])
            if summary["worker_count"]:
                all_conf.append(summary["avg_confidence"])
            if frame_idx % (frame_skip * 5) == 0:
                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Frame {frame_idx}/{total_frames}",
                    use_container_width=True,
                )
        frame_to_write = last_annotated if last_annotated is not None else frame
        writer.write(frame_to_write)
        frame_idx += 1
        if total_frames:
            progress.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    writer.release()
    progress.progress(1.0, text="Done!")

    total_workers = sum(all_worker_counts)
    total_compliant = sum(all_compliant)
    total_non_compliant = sum(all_non_compliant)
    avg_conf = sum(all_conf) / len(all_conf) if all_conf else 0.0
    status = (
        "No Workers Detected" if total_workers == 0 else
        "Compliant" if total_non_compliant == 0 else
        "Non-Compliant" if total_compliant == 0 else "Mixed"
    )

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Worker-Frames", total_workers)
    c2.metric("Compliant Frames", total_compliant)
    c3.metric("Non-Compliant Frames", total_non_compliant)

    with open(out_path, "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes)
    st.download_button("⬇️ Download Processed Video", data=video_bytes,
                        file_name=f"annotated_{uploaded.name}", mime="video/mp4")

    if st.button("💾 Save to History"):
        insert_detection(
            source_type="video",
            filename=uploaded.name,
            worker_count=total_workers,
            compliant_count=total_compliant,
            non_compliant_count=total_non_compliant,
            missing_ppe="",
            avg_confidence=avg_conf,
            compliance_status=status,
        )
        st.success("Saved to Detection History.")
else:
    st.info("Upload an MP4/AVI/MOV video to run PPE detection frame-by-frame.")
