import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import APP_TITLE, APP_ICON
from utils.database import init_db, fetch_filtered, delete_detection
from utils.reports import to_csv_bytes, to_pdf_bytes

st.set_page_config(page_title=f"History | {APP_TITLE}", page_icon=APP_ICON, layout="wide")
init_db()
st.title("📜 Detection History")

col1, col2, col3 = st.columns(3)
status_filter = col1.selectbox("Status", ["All", "Compliant", "Non-Compliant", "Mixed", "No Workers Detected"])
start_date = col2.date_input("From", value=None)
end_date = col3.date_input("To", value=None)

records = fetch_filtered(
    status=status_filter,
    start_date=str(start_date) if start_date else None,
    end_date=str(end_date) if end_date else None,
)

st.caption(f"{len(records)} record(s) found.")

if records:
    df = pd.DataFrame(records)
    st.dataframe(
        df[["id", "timestamp", "source_type", "filename", "worker_count",
            "compliant_count", "non_compliant_count", "missing_ppe",
            "compliance_status", "avg_confidence"]],
        use_container_width=True, hide_index=True,
    )

    dc1, dc2, dc3 = st.columns(3)
    dc1.download_button("⬇️ Export CSV", data=to_csv_bytes(records),
                         file_name="ppe_detection_history.csv", mime="text/csv")
    try:
        pdf_bytes = to_pdf_bytes(records)
        dc2.download_button("⬇️ Export PDF", data=pdf_bytes,
                             file_name="ppe_detection_report.pdf", mime="application/pdf")
    except ImportError:
        dc2.caption("Install `reportlab` to enable PDF export.")

    st.divider()
    st.subheader("Delete a Record")
    id_to_delete = st.number_input("Record ID", min_value=1, step=1)
    if st.button("🗑️ Delete", type="primary"):
        delete_detection(int(id_to_delete))
        st.success(f"Deleted record #{id_to_delete}. Refresh to update the table.")
        st.rerun()
else:
    st.info("No records match your filters.")
