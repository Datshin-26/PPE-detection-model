"""
Fine-tune YOLOv8 on a custom PPE dataset.

1. Export a labeled PPE dataset from Roboflow (or another source) in
   "YOLOv8" format. You'll get a folder with:
       data.yaml
       train/images, train/labels
       valid/images, valid/labels
       test/images,  test/labels
2. Place that folder inside ./dataset/ (or point --data at its data.yaml).
3. Make sure config.PPE_CLASS_NAMES matches the class order in data.yaml.
4. Run:
       python train.py --data dataset/data.yaml --epochs 100

The best weights are copied to models/ppe_best.pt, which detector.py picks
up automatically (switching the whole app from DEMO mode to CUSTOM mode).
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from config import MODELS_DIR, CUSTOM_WEIGHTS_PATH


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a PPE dataset")
    parser.add_argument("--data", type=str, default="dataset/data.yaml",
                         help="Path to data.yaml exported from Roboflow")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="Base checkpoint to fine-tune (n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Export a PPE dataset from Roboflow "
            "in YOLOv8 format and place it under ./dataset/ first."
        )

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs/train",
        name="ppe_detection",
    )

    # Locate best.pt from the run and copy it to the path detector.py expects
    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if best_weights.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights, CUSTOM_WEIGHTS_PATH)
        print(f"\n✅ Training complete. Best weights copied to {CUSTOM_WEIGHTS_PATH}")
        print("The Streamlit app will now use your custom model automatically.")
    else:
        print(f"\n⚠️ Training finished but best.pt not found at {best_weights}")


if __name__ == "__main__":
    main()
