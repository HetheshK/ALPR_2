# detector.py
"""
Step 1 — YOLO license-plate detection.

Exports:
    load_detector(weights_path) -> YOLO model
    detect_plates(model, image, ...) -> list[dict]
"""

import numpy as np
from pathlib import Path


def load_detector(weights_path: str = "yolo_plates.pt"):
    """
    Load a YOLOv8 model.

    Tries the fine-tuned plate weights first; falls back to the base
    yolov8n.pt (general-purpose) if the file is not found.
    """
    from ultralytics import YOLO

    weights = Path(weights_path)
    if weights.exists():
        model = YOLO(str(weights))
        print(f"[Step 1] Loaded fine-tuned weights: {weights}")
    else:
        model = YOLO("yolov8n.pt")
        print(f"[Step 1] Weights not found at '{weights_path}' — using base yolov8n.pt")

    return model


def detect_plates(
    model,
    image: np.ndarray,
    conf: float = 0.5,
    iou: float = 0.45,
    imgsz: int = 640,
    max_det: int = 5,
) -> list[dict]:
    """
    Run YOLO inference and return a list of plate detections.

    Each detection is a dict with keys:
      x1, y1, x2, y2  — bounding box in pixel coords
      confidence       — YOLO detection score
    """
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
        max_det=max_det,
    )

    detections = []
    h, w = image.shape[:2]

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            score = float(box.conf[0].cpu().numpy())

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Skip degenerate / tiny boxes
            if (x2 - x1) < 20 or (y2 - y1) < 5:
                continue

            detections.append({
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "confidence": score,
            })

    # Highest confidence first
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections
