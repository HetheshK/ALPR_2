# prep_crops.py
"""
One-time data preparation: run YOLO on every labeled image and save the
plate crops to disk so train_crnn.py can load them directly.

Usage:
    python prep_crops.py
    python prep_crops.py --conf 0.3 --out-dir crops --out-csv crops.csv
    python prep_crops.py --limit 500   # quick test run
"""

import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import csv
import argparse
import contextlib
import io
from pathlib import Path

import cv2

from detector import load_detector, detect_plates
from preprocessor import crop_plate
from evaluate import load_ground_truth

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser(description="Pre-crop labeled plates for CRNN training")
    parser.add_argument("--lakh",     default="Lakh",
                        help="Folder of full car images (default: Lakh)")
    parser.add_argument("--labels",   default="TRAINDATA - cleaned_output.csv.csv",
                        help="Ground-truth CSV (default: TRAINDATA - cleaned_output.csv.csv)")
    parser.add_argument("--weights",  default="yolo_plates.pt",
                        help="YOLO weights (default: yolo_plates.pt)")
    parser.add_argument("--conf",     type=float, default=0.3,
                        help="YOLO confidence threshold (default: 0.3 — lower for more coverage)")
    parser.add_argument("--out-dir",  default="crops", dest="out_dir",
                        help="Directory to save crop images (default: crops/)")
    parser.add_argument("--out-csv",  default="crops.csv", dest="out_csv",
                        help="Output CSV listing crop paths + labels (default: crops.csv)")
    parser.add_argument("--limit",    type=int, default=0,
                        help="Stop after N images (0 = all, useful for testing)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # ── Load YOLO ─────────────────────────────────────────────────────────────
    print("Loading YOLO...")
    yolo = load_detector(args.weights)

    # ── Load ground truth ─────────────────────────────────────────────────────
    gt = load_ground_truth(args.labels)
    print(f"Loaded {len(gt)} labels from {args.labels}")

    # ── Collect labeled images ────────────────────────────────────────────────
    lakh_dir  = Path(args.lakh)
    all_imgs  = [p for p in lakh_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    labeled   = sorted([p for p in all_imgs if p.name in gt], key=lambda p: p.name)

    print(f"Images with label: {len(labeled)}")

    if args.limit:
        labeled = labeled[:args.limit]
        print(f"Limiting to first {args.limit} images")

    # ── Run detection + crop ─────────────────────────────────────────────────
    rows         = []
    no_detection = 0
    saved        = 0

    for idx, img_path in enumerate(labeled, 1):
        label = gt[img_path.name]

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[{idx}/{len(labeled)}] SKIP (unreadable): {img_path.name}")
            continue

        with contextlib.redirect_stdout(io.StringIO()):
            detections = detect_plates(yolo, image, conf=args.conf)

        if not detections:
            no_detection += 1
            if idx % 500 == 0:
                print(f"[{idx}/{len(labeled)}] no detection so far: {no_detection}")
            continue

        crop = crop_plate(image, detections[0])
        if crop.size == 0:
            no_detection += 1
            continue

        crop_name = img_path.stem + "_crop.jpg"
        crop_path = out_dir / crop_name
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        rows.append({"crop": str(crop_path), "label": label})
        saved += 1

        if idx % 1000 == 0:
            pct = saved / idx * 100
            print(f"[{idx}/{len(labeled)}]  saved={saved}  no_det={no_detection}  "
                  f"yield={pct:.1f}%")

    # ── Write CSV ────────────────────────────────────────────────────────────
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["crop", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'─'*50}")
    print(f"  Total labeled images  : {len(labeled)}")
    print(f"  Crops saved           : {saved}")
    print(f"  No detection          : {no_detection}")
    print(f"  Yield                 : {saved/len(labeled)*100:.1f}%")
    print(f"  Output dir            : {out_dir}/")
    print(f"  Output CSV            : {args.out_csv}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()
