# 🦺 PPE Detection & Safety Compliance System

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat-square)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue.svg?style=flat-square)](LICENSE)

An end-to-end computer vision and artificial intelligence solution designed to monitor workplace safety and enforce **Personal Protective Equipment (PPE)** compliance. Built with **YOLOv8**, **OpenCV**, **Streamlit**, and **Plotly**, this application enables real-time detection of mandatory PPE gear (helmets, safety vests, masks) across static images, video files, and live camera streams.

---

## 🌟 Key Features & Capabilities

- 📷 **Image Detection**: Perform instant object detection on uploaded JPG/PNG images. View original vs. annotated side-by-side comparison, inspect detailed metrics (worker count, compliant count, non-compliant count, average confidence score), and export annotated images.
- 🎥 **Video Detection**: Frame-by-frame processing of MP4/AVI/MOV video clips with configurable frame-skipping and smooth frame retention. Preview progress in real-time and download the processed video with bounding box overlays.
- 🔴 **Live Camera Feed**: Real-time webcam inference powered by OpenCV. Displays live frame-per-second (FPS) counters and immediate worker compliance statistics.
- 📜 **Detection History & Audit Trail**: Log detection events directly into a localized SQLite database (`ppe_detections.db`). Filter records by compliance status or date ranges, and export official safety compliance reports in **CSV** or **PDF** format.
- 📊 **Analytics Dashboard**: Interactive data visualizations powered by Plotly, including daily detection volume, overall compliance status split, historical compliance trends over time, confidence score histograms, and missing PPE item frequency analysis.
- ⚙️ **Configurable System Settings**: Centralized configuration management for model confidence thresholds, IoU thresholds, mandatory PPE item lists, and model operating modes.
- 🔄 **Dual Engine Architecture**: Automatically toggles between **DEMO Mode** (stock COCO YOLOv8n person detector with simulated PPE logic) and **CUSTOM Mode** (fine-tuned YOLOv8 model trained on custom PPE datasets).

---

## 🏗️ System Architecture

```
PPE-Detection/
├── app.py                      # Main entry point (Home Dashboard)
├── train.py                     # Fine-tuning pipeline for custom YOLOv8 PPE model
├── config.py                     # Central configuration & tunable thresholds
├── requirements.txt            # Python dependencies
├── README.md                   # Detailed documentation
├── pages/                      # Streamlit Multi-Page UI Modules
│   ├── 1_📷_Image_Detection.py  # Image upload & detection page
│   ├── 2_🎥_Video_Detection.py  # Video processing & export page
│   ├── 3_🔴_Live_Detection.py   # Real-time webcam feed page
│   ├── 4_📜_Detection_History.py# Historical logs, filters & CSV/PDF export
│   ├── 5_📊_Analytics_Dashboard.py# Data visualizations & analytics
│   └── 6_⚙️_Settings.py         # System configuration & model status
├── utils/                      # Core Logic & Utilities
│   ├── detector.py             # YOLOv8 inference wrapper & compliance evaluation engine
│   ├── database.py             # SQLite persistence layer & summary querying
│   └── reports.py              # CSV & ReportLab PDF document generators
├── models/                     # Trained weights directory (ppe_best.pt)
├── dataset/                    # Roboflow / YOLOv8 dataset storage directory
├── database/                   # SQLite database file storage (ppe_detections.db)
└── reports/                    # Output directory for exported reports
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites
- Python 3.10 or higher installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/Datshin-26/PPE-detection-model.git
cd PPE-detection-model
```

### 3. Create a Virtual Environment & Install Dependencies
```bash
# On Linux / macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 4. Launch the Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`. On initial launch, the system automatically downloads `yolov8n.pt` and initializes the SQLite database.

---

## ⚙️ Operating Modes: Demo vs. Custom Model

Out of the box, standard pre-trained models (such as COCO weights) only detect generic classes like `person`. To support full PPE detection:

### 1. DEMO Mode (Default)
If no custom weight file (`models/ppe_best.pt`) is detected, the system runs in **DEMO Mode**. It uses `yolov8n.pt` to detect persons and simulates PPE compliance logic. This allows you to explore the full dashboard, database, analytics, and reporting capabilities immediately.

### 2. CUSTOM Mode (Real Detection)
To perform real PPE item detection (helmets, vests, masks):
1. **Download/Prepare Dataset**: Export a labeled PPE dataset in **YOLOv8** format (e.g. from [Roboflow Universe](https://universe.roboflow.com/)). Place it inside the `dataset/` directory so that `dataset/data.yaml` exists.
2. **Verify Classes**: Ensure `PPE_CLASS_NAMES` in `config.py` matches the class ordering specified in `dataset/data.yaml`.
3. **Train Model**:
   ```bash
   python train.py --data dataset/data.yaml --epochs 100 --imgsz 640 --batch 16
   ```
4. **Automatic Upgrade**: `train.py` automatically copies the fine-tuned checkpoint to `models/ppe_best.pt`. The application detects `ppe_best.pt` on launch and seamlessly switches to **CUSTOM Mode**.

---

## 🗄️ Database & Schema

The application uses an SQLite database (`database/ppe_detections.db`) managed through `utils/database.py`.

### Schema (`detections` table)
| Column | Type | Description |
|---|---|---|
| `id` | `INTEGER` | Auto-incrementing primary key |
| `timestamp` | `TEXT` | ISO-8601 formatted timestamp |
| `source_type` | `TEXT` | Source mode: `'image'`, `'video'`, or `'webcam'` |
| `filename` | `TEXT` | Original filename or stream label |
| `worker_count` | `INTEGER` | Total number of detected workers |
| `compliant_count` | `INTEGER` | Count of workers adhering to mandatory PPE |
| `non_compliant_count` | `INTEGER` | Count of workers missing mandatory PPE |
| `missing_ppe` | `TEXT` | Comma-separated list of missing gear items |
| `avg_confidence` | `REAL` | Average model confidence score |
| `compliance_status` | `TEXT` | Overall status (`'Compliant'`, `'Non-Compliant'`, `'Mixed'`) |

---

## 🐳 Docker Deployment

To package and run the application inside a Docker container:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

Build and execute the container:
```bash
docker build -t ppe-detection-app .
docker run -p 8501:8501 ppe-detection-app
```

---

## 🛠️ Technology Stack

- **Computer Vision & AI**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [OpenCV](https://opencv.org/), [PyTorch](https://pytorch.org/)
- **Frontend & Web Framework**: [Streamlit](https://streamlit.io/)
- **Data Analysis & Visualization**: [Pandas](https://pandas.pydata.org/), [Plotly Express](https://plotly.com/python/)
- **Storage & PDF Generation**: [SQLite3](https://sqlite.org/), [ReportLab](https://www.reportlab.com/)

---

## 📝 License

Distributed under the open-source AGPL-3.0 License (inherited from Ultralytics YOLOv8). Refer to `LICENSE` for more details.
