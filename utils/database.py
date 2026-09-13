"""
SQLite persistence layer for detection records.
All database access in the app should go through this module — no raw
SQL scattered across pages.
"""
import sqlite3
import datetime
from contextlib import contextmanager
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DB_PATH


SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source_type TEXT NOT NULL,          -- 'image', 'video', 'webcam'
    filename TEXT,
    worker_count INTEGER DEFAULT 0,
    compliant_count INTEGER DEFAULT 0,
    non_compliant_count INTEGER DEFAULT 0,
    missing_ppe TEXT,                   -- comma-separated list
    avg_confidence REAL,
    compliance_status TEXT,             -- 'Compliant', 'Non-Compliant', 'Mixed'
    annotated_image_path TEXT
);
"""


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.execute(SCHEMA)


def insert_detection(source_type, filename, worker_count, compliant_count,
                      non_compliant_count, missing_ppe, avg_confidence,
                      compliance_status, annotated_image_path=None):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO detections
               (timestamp, source_type, filename, worker_count, compliant_count,
                non_compliant_count, missing_ppe, avg_confidence,
                compliance_status, annotated_image_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.datetime.now().isoformat(timespec="seconds"),
                source_type,
                filename,
                worker_count,
                compliant_count,
                non_compliant_count,
                missing_ppe,
                avg_confidence,
                compliance_status,
                annotated_image_path,
            ),
        )


def fetch_all(limit=None):
    query = "SELECT * FROM detections ORDER BY id DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    with get_connection() as conn:
        rows = conn.execute(query).fetchall()
    return [dict(r) for r in rows]


def fetch_filtered(status=None, start_date=None, end_date=None):
    query = "SELECT * FROM detections WHERE 1=1"
    params = []
    if status and status != "All":
        query += " AND compliance_status = ?"
        params.append(status)
    if start_date:
        query += " AND date(timestamp) >= date(?)"
        params.append(start_date)
    if end_date:
        query += " AND date(timestamp) <= date(?)"
        params.append(end_date)
    query += " ORDER BY id DESC"
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def delete_detection(detection_id):
    with get_connection() as conn:
        conn.execute("DELETE FROM detections WHERE id = ?", (detection_id,))


def get_summary_stats():
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM detections").fetchone()["c"]
        workers = conn.execute(
            "SELECT COALESCE(SUM(worker_count),0) c FROM detections"
        ).fetchone()["c"]
        compliant = conn.execute(
            "SELECT COALESCE(SUM(compliant_count),0) c FROM detections"
        ).fetchone()["c"]
        non_compliant = conn.execute(
            "SELECT COALESCE(SUM(non_compliant_count),0) c FROM detections"
        ).fetchone()["c"]
        today = conn.execute(
            "SELECT COUNT(*) c FROM detections WHERE date(timestamp) = date('now')"
        ).fetchone()["c"]
    compliance_pct = (compliant / workers * 100) if workers else 0.0
    return {
        "total_detections": total,
        "total_workers": workers,
        "compliant_workers": compliant,
        "non_compliant_workers": non_compliant,
        "compliance_pct": round(compliance_pct, 1),
        "today_detections": today,
    }
