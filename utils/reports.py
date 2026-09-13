"""
Export helpers for the Detection History page — CSV always available,
PDF built with reportlab (falls back gracefully if not installed).
"""
import io
import csv
import datetime


def to_csv_bytes(records: list[dict]) -> bytes:
    if not records:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue().encode("utf-8")


def to_pdf_bytes(records: list[dict], title: str = "PPE Detection Report") -> bytes:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    elements = [
        Paragraph(title, styles["Title"]),
        Paragraph(f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}", styles["Normal"]),
        Spacer(1, 12),
    ]

    if records:
        headers = list(records[0].keys())
        data = [headers] + [[str(r.get(h, "")) for h in headers] for r in records]
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
        ]))
        elements.append(table)
    else:
        elements.append(Paragraph("No records found.", styles["Normal"]))

    doc.build(elements)
    return buf.getvalue()
