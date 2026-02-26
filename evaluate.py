# evaluate.py
"""
Batch accuracy evaluation against the TRAINDATA CSV ground truth.

Usage:
    python evaluate.py                          # PaddleOCR + TrOCR, no LLM
    python evaluate.py --no-trocr               # PaddleOCR only (faster)
    python evaluate.py --trocr-only             # TrOCR only (benchmarks TrOCR in isolation)
    python evaluate.py --llm                    # enable LLM tiebreaker
    python evaluate.py --out results.csv        # save per-image results
    python evaluate.py --lakh  path/to/images   # custom image folder
    python evaluate.py --labels path/to/gt.csv  # custom CSV
    python evaluate.py --limit 100              # stop after 100 images
"""

# Must be set before any PaddleOCR / PaddleX imports
import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import csv
import io
import time
import argparse
import contextlib
from pathlib import Path

import cv2

from detector import load_detector, detect_plates
from ocr import load_ocr, load_llm_ocr, load_trocr, load_crnn, run_trocr_only, read_plate_trocr_llm
from preprocessor import crop_plate
from pipeline import run_pipeline


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_ground_truth(csv_path: str) -> dict[str, str]:
    """Return {filename: plate_label} from the CSV (one label per file assumed)."""
    gt = {}
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[0].strip() and row[2].strip():
                filename = row[0].strip()
                plate    = row[2].strip().upper()
                gt[filename] = plate   # last label wins if duplicates exist
    return gt


def normalize(text: str) -> str:
    """Strip dashes and spaces — compare alphanumeric content only."""
    return "".join(c for c in text.upper() if c.isalnum())


def char_accuracy(pred: str, gt: str) -> float:
    """1 - normalised edit distance (0.0 = completely wrong, 1.0 = perfect)."""
    if not gt:
        return 1.0 if not pred else 0.0
    m, n = len(pred), len(gt)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if pred[i - 1] == gt[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return max(0.0, 1.0 - dp[n] / max(m, n))


def print_summary(rows, exact_correct, no_detection, total_char_acc, t_start, out_path):
    """Print accuracy summary and save results CSV."""
    n = len(rows)
    if n == 0:
        print("No results.")
        return

    exact_acc = exact_correct / n
    avg_char  = total_char_acc / n
    elapsed   = time.time() - t_start

    print("\n" + "═" * 55)
    print(f"  Images evaluated : {n}")
    print(f"  No detection     : {no_detection}")
    print(f"  Exact match      : {exact_correct}/{n}  ({exact_acc:.1%})")
    print(f"  Avg char accuracy: {avg_char:.1%}")
    print(f"  Total time       : {elapsed/60:.1f} min  ({elapsed/n:.1f}s / image)")
    print("═" * 55)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-image results saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch accuracy evaluation")
    parser.add_argument("--lakh",    default="Lakh",
                        help="Folder of images (default: Lakh)")
    parser.add_argument("--labels",  default="TRAINDATA - cleaned_output.csv.csv",
                        help="Ground-truth CSV  (default: TRAINDATA - cleaned_output.csv.csv)")
    parser.add_argument("--weights", default="yolo_plates.pt",
                        help="YOLO weights (default: yolo_plates.pt)")
    parser.add_argument("--conf",    type=float, default=0.5,
                        help="YOLO detection threshold (default: 0.5)")
    parser.add_argument("--out",     default="eval_results.csv",
                        help="Output CSV for per-image results (default: eval_results.csv)")
    parser.add_argument("--no-trocr", action="store_true", dest="no_trocr",
                        help="Disable TrOCR (faster, less accurate)")
    parser.add_argument("--trocr-only", action="store_true", dest="trocr_only",
                        help="Skip PaddleOCR — benchmark TrOCR in isolation (YOLO still runs)")
    parser.add_argument("--llm",    action="store_true",
                        help="Enable Qwen2-VL LLM tiebreaker")
    parser.add_argument("--llm-model", default="qwen2.5vl:3b", dest="llm_model")
    parser.add_argument("--trocr-model", default="microsoft/trocr-base-printed",
                        dest="trocr_model")
    parser.add_argument("--no-crnn", action="store_true", dest="no_crnn",
                        help="Disable CRNN cross-validation")
    parser.add_argument("--crnn-model", default="crnn_plates.pt", dest="crnn_model",
                        help="Path to trained CRNN weights (default: crnn_plates.pt)")
    parser.add_argument("--limit",   type=int, default=0,
                        help="Stop after N images (0 = all)")
    args = parser.parse_args()

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading models...")
    yolo  = load_detector(args.weights)
    trocr = load_trocr(model_name=args.trocr_model) if (args.trocr_only or not args.no_trocr) else None
    crnn  = None if (args.no_crnn or args.trocr_only) else load_crnn(model_path=args.crnn_model)

    if args.trocr_only:
        ocr     = None
        llm_cfg = load_llm_ocr(model=args.llm_model) if args.llm else None
        mode_label = "TrOCR+LLM" if llm_cfg else "TrOCR-only"
        print(f"[Mode] {mode_label} — PaddleOCR skipped")
    else:
        ocr     = load_ocr()
        llm_cfg = load_llm_ocr(model=args.llm_model) if args.llm else None

    llm_threshold = 0.85

    # ── Load ground truth ────────────────────────────────────────────────────
    gt = load_ground_truth(args.labels)
    print(f"Loaded {len(gt)} ground-truth labels from {args.labels}")

    # ── Collect images that have a label ────────────────────────────────────
    lakh_dir = Path(args.lakh)
    all_images = [p for p in lakh_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    labeled    = [p for p in all_images if p.name in gt]
    labeled.sort(key=lambda p: p.name)

    print(f"Images in folder : {len(all_images)}")
    print(f"Images with label: {len(labeled)}")

    if args.limit:
        labeled = labeled[:args.limit]
        print(f"Evaluating first {args.limit} images")

    # ── Run evaluation ───────────────────────────────────────────────────────
    rows = []
    exact_correct  = 0
    total_char_acc = 0.0
    no_detection   = 0
    t_start        = time.time()

    try:
        for idx, img_path in enumerate(labeled, 1):
            gt_label = gt[img_path.name]
            gt_norm  = normalize(gt_label)

            image = cv2.imread(str(img_path))
            if image is None:
                print(f"[{idx}/{len(labeled)}] SKIP (unreadable): {img_path.name}")
                continue

            if args.trocr_only:
                # YOLO detects the plate; TrOCR reads the raw crop directly
                with contextlib.redirect_stdout(io.StringIO()):
                    detections = detect_plates(yolo, image, conf=args.conf)
                if detections:
                    raw_crop  = crop_plate(image, detections[0])
                    pred_text, pred_conf = read_plate_trocr_llm(
                        trocr, raw_crop, llm_cfg=llm_cfg, llm_threshold=llm_threshold)
                else:
                    pred_text, pred_conf = "", 0.0
                    no_detection += 1
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    _, _, ocr_results = run_pipeline(
                        yolo, ocr, image,
                        conf_threshold=args.conf,
                        llm_cfg=llm_cfg,
                        trocr=trocr,
                        crnn=crnn,
                        llm_threshold=llm_threshold,
                    )
                if ocr_results:
                    pred_text, pred_conf = ocr_results[0]
                else:
                    pred_text, pred_conf = "", 0.0
                    no_detection += 1

            pred_norm = normalize(pred_text)
            exact     = pred_norm == gt_norm
            c_acc     = char_accuracy(pred_norm, gt_norm)

            exact_correct  += int(exact)
            total_char_acc += c_acc

            elapsed   = time.time() - t_start
            per_img   = elapsed / idx
            remaining = per_img * (len(labeled) - idx)

            status = "✓" if exact else "✗"
            conf_str = f"{pred_conf:.2f}" if pred_conf else "n/a"
            print(
                f"[{idx}/{len(labeled)}] {status}  "
                f"gt='{gt_label}'  pred='{pred_text}'  conf={conf_str}  "
                f"char={c_acc:.0%}  "
                f"ETA {remaining/60:.1f}min"
            )

            rows.append({
                "filename":      img_path.name,
                "ground_truth":  gt_label,
                "predicted":     pred_text,
                "exact_match":   int(exact),
                "char_accuracy": f"{c_acc:.4f}",
                "confidence":    conf_str,
            })

    except KeyboardInterrupt:
        print("\n\n[Interrupted — showing partial results]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(rows, exact_correct, no_detection, total_char_acc, t_start, args.out)


if __name__ == "__main__":
    main()
