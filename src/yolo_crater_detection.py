from pathlib import Path
import pandas as pd
from ultralytics import YOLO


def train_crater_detector(data_yaml, output_dir="crater_run", epochs=50, imgsz=640, **kwargs):
    """Train a YOLOv8 crater detector using a prepared YOLO-format dataset."""
    model = YOLO("yolov8n.pt")
    return model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=output_dir,
        name="yolov8n_craters",
        **kwargs,
    )


def summarize_training(results_csv):
    """Return the final Precision, Recall, and mAP@0.5 values from a YOLO run."""
    df = pd.read_csv(results_csv)
    df.columns = [column.strip() for column in df.columns]
    metrics = {
        "precision": float(df["metrics/precision(B)"].iloc[-1]),
        "recall": float(df["metrics/recall(B)"].iloc[-1]),
        "mAP50": float(df["metrics/mAP50(B)"].iloc[-1]),
    }
    return metrics


def compare_runs(default_csv, tuned_csv):
    default = summarize_training(default_csv)
    tuned = summarize_training(tuned_csv)
    comparison = pd.DataFrame([default, tuned], index=["Default", "Tuned"])
    print(comparison)
    return comparison


if __name__ == "__main__":
    default_results = Path("crater_run/yolov8n_craters/results.csv")
    tuned_results = Path("crater_run/yolov8n_craters_Tuned_Hyp/results.csv")

    if default_results.exists() and tuned_results.exists():
        compare_runs(default_results, tuned_results)
    else:
        print("Train the detector first or provide existing YOLO results.csv files.")
