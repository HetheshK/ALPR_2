# detect_plate.py
"""
License Plate Detector + PaddleOCR Reader — CLI entry point.

Pipeline steps live in separate modules:
  detector.py     — Step 1:  YOLO detection
  preprocessor.py — Steps 2/3: crop + image preprocessing
  ocr.py          — Steps 4/4b: PaddleOCR + Qwen2-VL LLM fallback
  pipeline.py     — frame annotation, pipeline runner, image/webcam handlers

Usage:
  python detect_plate.py --image car.jpg
  python detect_plate.py --image car.jpg --save
  python detect_plate.py --webcam
  python detect_plate.py --image car.jpg --conf 0.4
  python detect_plate.py --image car.jpg --no-llm
"""

# Must be set before any PaddleOCR / PaddleX imports
import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# Note: HF_HUB_OFFLINE is intentionally NOT set here so TrOCR can download on first run.

import sys
import argparse
import cv2

from detector import load_detector
from ocr import load_ocr, load_llm_ocr, load_trocr, load_crnn, load_fast_ocr, run_trocr_only, read_plate_trocr_llm
from pipeline import process_image, process_webcam


def main():
    parser = argparse.ArgumentParser(
        description="License Plate Detector + PaddleOCR Reader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", "-i", help="Path to an image file")
    parser.add_argument("--webcam", "-w", action="store_true",
                        help="Use webcam instead of an image file")
    parser.add_argument("--webcam-id", type=int, default=0, dest="webcam_id",
                        help="Webcam device index (default: 0)")
    parser.add_argument("--weights", default="yolo_plates.pt",
                        help="Path to YOLO weights (default: yolo_plates.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="YOLO detection confidence threshold (default: 0.5)")
    parser.add_argument("--save", action="store_true",
                        help="Save the annotated image to disk")
    parser.add_argument("--every-n", type=int, default=5, dest="every_n",
                        help="Webcam: run full pipeline every N frames (default: 5)")
    parser.add_argument("--no-llm", action="store_true", dest="no_llm",
                        help="Disable Qwen2-VL LLM fallback even if Ollama is running")
    parser.add_argument("--debug", action="store_true",
                        help="Save raw and preprocessed crop images for each detected plate")
    parser.add_argument("--llm-model", default="qwen2.5vl:3b", dest="llm_model",
                        help="Ollama vision model to use for LLM tiebreaker (default: qwen2.5vl:3b)")
    parser.add_argument("--llm-threshold", type=float, default=0.85, dest="llm_threshold",
                        help="Confidence below which LLM runs when TrOCR is disabled (default: 0.85)")
    parser.add_argument("--no-trocr", action="store_true", dest="no_trocr",
                        help="Disable TrOCR cross-validation")
    parser.add_argument("--trocr-model", default="microsoft/trocr-base-printed",
                        dest="trocr_model",
                        help="HuggingFace TrOCR model (default: microsoft/trocr-base-printed)")
    parser.add_argument("--trocr-only", action="store_true", dest="trocr_only",
                        help="Skip YOLO+PaddleOCR and run TrOCR directly on the image")
    parser.add_argument("--no-crnn", action="store_true", dest="no_crnn",
                        help="Disable CRNN cross-validation")
    parser.add_argument("--crnn-model", default="crnn_plates.pt", dest="crnn_model",
                        help="Path to trained CRNN weights (default: crnn_plates.pt)")
    parser.add_argument("--no-fast-ocr", action="store_true", dest="no_fast_ocr",
                        help="Disable fast_plate_ocr engine")
    parser.add_argument("--fast-ocr-model", default="cct-s-v1-global-model", dest="fast_ocr_model",
                        help="fast_plate_ocr model name (default: cct-s-v1-global-model)")

    args = parser.parse_args()

    # ── TrOCR-only shortcut — no YOLO or PaddleOCR needed ────────────────────
    if args.trocr_only:
        if not args.image:
            print("ERROR: --trocr-only requires --image <path>")
            sys.exit(1)
        image = cv2.imread(args.image)
        if image is None:
            print(f"ERROR: Could not load image: {args.image}")
            sys.exit(1)
        trocr = load_trocr(model_name=args.trocr_model)
        if trocr is None:
            print("ERROR: TrOCR failed to load")
            sys.exit(1)
        llm_cfg = None if args.no_llm else load_llm_ocr(model=args.llm_model)
        result, conf = read_plate_trocr_llm(trocr, image, llm_cfg=llm_cfg,
                                            llm_threshold=args.llm_threshold)
        print(f"\nResult: '{result}'  conf={conf:.3f}")
        return

    # Load models once — shared across all frames
    model    = load_detector(args.weights)
    ocr      = load_ocr()
    llm_cfg  = None if args.no_llm      else load_llm_ocr(model=args.llm_model)
    trocr    = None if args.no_trocr    else load_trocr(model_name=args.trocr_model)
    crnn     = None if args.no_crnn     else load_crnn(model_path=args.crnn_model)
    fast_ocr = None if args.no_fast_ocr else load_fast_ocr(model_name=args.fast_ocr_model)

    if args.webcam:
        process_webcam(model, ocr, llm_cfg, args.webcam_id, args.conf, args.every_n,
                       trocr=trocr, crnn=crnn, fast_ocr=fast_ocr,
                       llm_threshold=args.llm_threshold)
    elif args.image:
        process_image(model, ocr, llm_cfg, args.image, args.conf, args.save, args.debug,
                      trocr=trocr, crnn=crnn, fast_ocr=fast_ocr,
                      llm_threshold=args.llm_threshold)
    else:
        parser.print_help()
        print("\nERROR: Provide --image <path> or --webcam")
        sys.exit(1)


if __name__ == "__main__":
    main()
