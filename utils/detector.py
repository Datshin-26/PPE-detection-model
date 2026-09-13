"""
Detection engine: wraps Ultralytics YOLOv8 and turns raw model output into
per-worker compliance decisions.

Two modes:
  - CUSTOM mode: loads models/ppe_best.pt (produced by train.py after you
    fine-tune on a labeled PPE dataset). Detects real helmet/vest/mask classes.
  - DEMO mode: falls back to stock COCO yolov8n.pt. Only "person" is
    meaningful; PPE presence is simulated so the full pipeline (compliance
    logic, DB writes, analytics) is demoable end-to-end without a trained
    model. Every demo result is clearly labeled as such in the UI.

Swap DEMO for real detections the moment you have CUSTOM_WEIGHTS_PATH —
no other code changes needed.
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    CUSTOM_WEIGHTS_PATH, DEMO_WEIGHTS_NAME, MANDATORY_PPE,
    CONFIDENCE_THRESHOLD, IOU_THRESHOLD, COMPLIANT_COLOR, NON_COMPLIANT_COLOR,
)


class PPEDetector:
    def __init__(self):
        if CUSTOM_WEIGHTS_PATH.exists():
            self.model = YOLO(str(CUSTOM_WEIGHTS_PATH))
            self.mode = "custom"
        else:
            self.model = YOLO(DEMO_WEIGHTS_NAME)
            self.mode = "demo"

    # ------------------------------------------------------------------
    def detect(self, image_bgr: np.ndarray):
        """
        Run detection on a single BGR image (as read by cv2.imread).
        Returns (annotated_image, list_of_worker_results, summary_dict).
        """
        results = self.model.predict(
            image_bgr, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False
        )[0]

        names = results.names
        boxes = results.boxes
        persons = []
        ppe_items = []

        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            if label == "person":
                persons.append({"box": xyxy, "conf": conf})
            else:
                ppe_items.append({"label": label, "box": xyxy, "conf": conf})

        annotated = image_bgr.copy()
        worker_results = []

        if self.mode == "custom":
            worker_results = self._match_ppe_to_persons(persons, ppe_items)
        else:
            # DEMO MODE: no real PPE classes exist in COCO, so we simulate
            # compliance per detected person. This keeps the rest of the
            # pipeline (DB, analytics, history) fully functional for a demo.
            for p in persons:
                is_compliant = random.random() > 0.35
                missing = [] if is_compliant else random.sample(
                    MANDATORY_PPE, k=random.randint(1, len(MANDATORY_PPE))
                )
                worker_results.append({
                    "box": p["box"],
                    "conf": p["conf"],
                    "compliant": is_compliant,
                    "missing": missing,
                })

        for w in worker_results:
            x1, y1, x2, y2 = map(int, w["box"])
            color = COMPLIANT_COLOR if w["compliant"] else NON_COMPLIANT_COLOR
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_text = "Compliant" if w["compliant"] else f"Missing: {', '.join(w['missing'])}"
            cv2.putText(annotated, label_text, (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        compliant_count = sum(1 for w in worker_results if w["compliant"])
        non_compliant_count = len(worker_results) - compliant_count
        avg_conf = (
            sum(w["conf"] for w in worker_results) / len(worker_results)
            if worker_results else 0.0
        )
        all_missing = sorted({m for w in worker_results for m in w.get("missing", [])})

        if not worker_results:
            status = "No Workers Detected"
        elif non_compliant_count == 0:
            status = "Compliant"
        elif compliant_count == 0:
            status = "Non-Compliant"
        else:
            status = "Mixed"

        summary = {
            "mode": self.mode,
            "worker_count": len(worker_results),
            "compliant_count": compliant_count,
            "non_compliant_count": non_compliant_count,
            "avg_confidence": round(avg_conf, 3),
            "missing_ppe": ", ".join(all_missing),
            "compliance_status": status,
        }
        return annotated, worker_results, summary

    # ------------------------------------------------------------------
    @staticmethod
    def _match_ppe_to_persons(persons, ppe_items, iou_thresh=0.1):
        """
        For CUSTOM mode: associate detected PPE boxes with the nearest
        person box (simple IoU / containment heuristic) and decide
        compliance based on MANDATORY_PPE presence.
        """
        def iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            union = area_a + area_b - inter
            return inter / union if union else 0

        results = []
        for p in persons:
            found = set()
            confs = [p["conf"]]
            for item in ppe_items:
                if iou(p["box"], item["box"]) > iou_thresh:
                    found.add(item["label"])
                    confs.append(item["conf"])
            missing = [m for m in MANDATORY_PPE if m not in found
                       and f"no_{m}" not in found]
            # explicit "no_x" negative classes also count as missing
            missing += [m for m in MANDATORY_PPE if f"no_{m}" in found and m not in missing]
            missing = sorted(set(missing))
            results.append({
                "box": p["box"],
                "conf": p["conf"],
                "compliant": len(missing) == 0,
                "missing": missing,
            })
        return results
