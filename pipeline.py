# pipeline.py
"""
Core pipeline runner and frame annotation.

Exports:
    annotate_frame(frame, detections, ocr_results) -> np.ndarray
    run_pipeline(model, ocr, image, conf_threshold, llm_cfg) -> tuple
    process_image(model, ocr, llm_cfg, image_path, conf, save)
    process_webcam(model, ocr, llm_cfg, webcam_id, conf, every_n)
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

from detector import detect_plates
from preprocessor import crop_plate, preprocess_crop
from ocr import read_plate


def annotate_frame(
    frame: np.ndarray,
    detections: list[dict],
    ocr_results: list[tuple[str, float]],
) -> np.ndarray:
    """Draw bounding boxes and OCR text onto the frame."""
    out = frame.copy()
    for det, (text, conf) in zip(detections, ocr_results):
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        det_conf = det["confidence"]

        # Box colour: green → high det confidence, orange → medium, red → low
        if det_conf >= 0.80:
            color = (0, 220, 0)
        elif det_conf >= 0.55:
            color = (0, 165, 255)
        else:
            color = (0, 0, 220)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{text}  ({conf:.0%})" if text else f"det={det_conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        label_y = max(y1 - 8, th + 4)
        cv2.rectangle(out, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), color, -1)
        cv2.putText(out, label, (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return out


def run_pipeline(model, ocr, image: np.ndarray, conf_threshold: float,
                 llm_cfg: dict | None = None,
                 trocr=None,
                 crnn=None,
                 debug: bool = False,
                 llm_threshold: float = 0.85) -> tuple:
    """
    Run the full detection + OCR pipeline on a single frame.

    Returns:
        (annotated_frame, detections, ocr_results)
    """
    print("\n" + "─" * 50)

    # Step 1 — Detect
    t0 = time.time()
    detections = detect_plates(model, image, conf=conf_threshold)
    print(f"[Step 1] Detected {len(detections)} plate(s) in {(time.time()-t0)*1000:.0f}ms")

    if not detections:
        return image, [], []

    # Keep only the highest-confidence plate per image
    detections = detections[:1]

    ocr_results = []
    for i, det in enumerate(detections, 1):
        print(f"\n  — Plate #{i}  bbox=({det['x1']},{det['y1']},{det['x2']},{det['y2']})"
              f"  det_conf={det['confidence']:.3f}")

        # Step 2 — Crop (tight vertical trim strips state banners)
        t1 = time.time()
        crop = crop_plate(image, det)
        print(f"[Step 2] Cropped: {crop.shape[1]}×{crop.shape[0]}px  ({(time.time()-t1)*1000:.0f}ms)")
        if debug:
            cv2.imwrite(f"debug_plate{i}_raw.jpg", crop)

        # Step 3 — Preprocess
        t2 = time.time()
        preprocessed = preprocess_crop(crop)
        print(f"[Step 3] Preprocessed → {preprocessed.shape[1]}×{preprocessed.shape[0]}px  "
              f"({(time.time()-t2)*1000:.0f}ms)")
        if debug:
            cv2.imwrite(f"debug_plate{i}_preprocessed.jpg", preprocessed)

        # Step 4 — OCR (passes A/B via PaddleOCR, pass C via Qwen2-VL if needed)
        t3 = time.time()
        text, ocr_conf = read_plate(ocr, crop, preprocessed,
                                    trocr=trocr, crnn=crnn,
                                    llm_cfg=llm_cfg,
                                    llm_threshold=llm_threshold)
        print(f"[Step 4] OCR → '{text}'  conf={ocr_conf:.3f}  ({(time.time()-t3)*1000:.0f}ms)")

        ocr_results.append((text, ocr_conf))

    annotated = annotate_frame(image, detections, ocr_results)
    return annotated, detections, ocr_results


def process_image(model, ocr, llm_cfg, image_path: str,
                  conf: float, save: bool, debug: bool = False,
                  trocr=None, crnn=None, llm_threshold: float = 0.85) -> None:
    """Run the pipeline on a single image file."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        sys.exit(1)

    annotated, _, _ = run_pipeline(model, ocr, image, conf, llm_cfg=llm_cfg,
                                   trocr=trocr, crnn=crnn, debug=debug,
                                   llm_threshold=llm_threshold)

    if save:
        out_path = f"{Path(image_path).stem}_detected.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"\nAnnotated image saved → {out_path}")

    try:
        cv2.imshow("ALPR — Detected Plates", annotated)
        print("\nPress any key in the image window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("(No display available — use --save to write the result to disk)")


def process_webcam(model, ocr, llm_cfg, webcam_id: int,
                   conf: float, every_n: int,
                   trocr=None, crnn=None, llm_threshold: float = 0.85) -> None:
    """Run the pipeline on a live webcam feed."""
    cap = cv2.VideoCapture(webcam_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open webcam {webcam_id}")
        sys.exit(1)

    print(f"\nWebcam {webcam_id} opened — press Q to quit")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Run the full pipeline every N frames to keep the feed smooth
        if frame_count % every_n == 0:
            annotated, _, _ = run_pipeline(model, ocr, frame, conf, llm_cfg=llm_cfg,
                                           trocr=trocr, crnn=crnn,
                                           llm_threshold=llm_threshold)
        else:
            annotated = frame

        cv2.imshow("ALPR — Webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
