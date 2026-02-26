# ALPR_2 — Automatic License Plate Recognition

A multi-engine OCR pipeline for reading US license plates from images or a live webcam feed.

## Pipeline Overview

```
Image → YOLO detection → Crop + Preprocess → PaddleOCR (pass A + B)
                                           → TrOCR (cross-check)
                                           → Qwen2-VL LLM (tiebreak on mismatch)
                                           → Zone-corrected text output
```

1. **YOLO** detects the plate bounding box
2. **Preprocessor** crops, denoises, enhances contrast, sharpens, and pads the crop
3. **PaddleOCR** runs two passes (preprocessed + plain resize)
4. **TrOCR** runs in parallel as a second opinion
5. If PaddleOCR and TrOCR agree → return that result (boosted confidence)
6. If they disagree → **Qwen2-VL LLM** (via Ollama) makes the final call
7. Zone correction fixes common OCR confusions (e.g. `0`↔`O`, `1`↔`I`, `8`↔`B`)

---

## detect_plate.py — Single image or webcam

### Image mode

```bash
python detect_plate.py --image car.jpg
```

Runs the full pipeline on one image and opens a window showing the annotated result.

```bash
python detect_plate.py --image car.jpg --save
```

Same as above, but also writes `car_detected.jpg` to disk (useful when no display is available).

```bash
python detect_plate.py --image car.jpg --debug
```

Saves `debug_plate1_raw.jpg` and `debug_plate1_preprocessed.jpg` alongside the result so you can inspect exactly what is being fed into OCR.

### Webcam mode

```bash
python detect_plate.py --webcam
```

Opens webcam 0 and runs the pipeline continuously. Press **Q** to quit.

```bash
python detect_plate.py --webcam --webcam-id 1
```

Use a specific webcam index if you have multiple cameras.

```bash
python detect_plate.py --webcam --every-n 10
```

Run the full pipeline every 10 frames instead of the default 5. Higher = smoother feed, lower = more frequent reads.

### Detection tuning

```bash
python detect_plate.py --image car.jpg --conf 0.4
```

Lower the YOLO confidence threshold (default `0.5`). Useful if plates are being missed; may increase false positives.

```bash
python detect_plate.py --image car.jpg --weights custom_yolo.pt
```

Use a different set of YOLO weights. Defaults to `yolo_plates.pt`; falls back to `yolov8n.pt` if not found.

### OCR engine flags

```bash
python detect_plate.py --image car.jpg --no-trocr
```

Disable TrOCR. Faster, but loses the cross-check — the LLM will never fire without a mismatch to resolve.

```bash
python detect_plate.py --image car.jpg --trocr-model microsoft/trocr-large-printed
```

Use a different HuggingFace TrOCR model (default: `microsoft/trocr-base-printed`). The `large` variant is more accurate but slower.

### LLM flags

```bash
python detect_plate.py --image car.jpg --no-llm
```

Disable the Qwen2-VL LLM entirely. If TrOCR and PaddleOCR disagree, the pipeline falls back to TrOCR at 0.75 confidence.

```bash
python detect_plate.py --image car.jpg --llm-model qwen2.5vl:7b
```

Use a different Ollama vision model (default: `qwen2.5vl:3b`). Larger models are more accurate but slower. Requires Ollama to be running.

```bash
python detect_plate.py --image car.jpg --llm-threshold 0.75
```

When TrOCR is disabled, the LLM fires if PaddleOCR confidence drops below this threshold (default: `0.85`). Has no effect when TrOCR is active (LLM only fires on mismatch then).

---

## evaluate.py — Batch accuracy evaluation

Runs the pipeline against a folder of images and compares results to a ground-truth CSV.

### Basic usage

```bash
python evaluate.py
```

Evaluates all labeled images in the `Lakh/` folder against `TRAINDATA - cleaned_output.csv.csv`. Prints per-image results and a final accuracy summary. Saves per-image results to `eval_results.csv`.

Press **Ctrl+C** at any time to stop early — partial results and accuracy will still be printed and saved.

### Outputs

Per-image console line:
```
[42/1826] ✓  gt='ABC-1234'  pred='ABC-1234'  conf=0.95  char=100%  ETA 12.3min
```

Final summary:
```
═══════════════════════════════════════════════
  Images evaluated : 1826
  No detection     : 23
  Exact match      : 1541/1826  (84.4%)
  Avg char accuracy: 96.2%
  Total time       : 94.3 min  (3.1s / image)
═══════════════════════════════════════════════
```

### Paths

```bash
python evaluate.py --lakh path/to/images --labels path/to/ground_truth.csv
```

Use a custom image folder and/or CSV. CSV format expected: `filename, state, plate_number` (columns 0 and 2 are used).

```bash
python evaluate.py --out my_results.csv
```

Change the output CSV filename (default: `eval_results.csv`).

### Speed / accuracy tradeoffs

```bash
python evaluate.py --no-trocr
```

Skip TrOCR — roughly 2-3x faster per image, at some accuracy cost.

```bash
python evaluate.py --llm
```

Enable the Qwen2-VL LLM tiebreaker (disabled by default in evaluate mode). Requires Ollama running with the model loaded.

```bash
python evaluate.py --llm --llm-model qwen2.5vl:7b
```

Use a larger LLM model for the tiebreaker.

```bash
python evaluate.py --limit 100
```

Stop after 100 images. Useful for quick spot-checks without running the full dataset.

```bash
python evaluate.py --conf 0.4
```

Lower the YOLO detection threshold. May recover missed plates at the cost of some false positives.

---

## Module reference

| File | Role |
|------|------|
| [detect_plate.py](detect_plate.py) | CLI entry point for image / webcam |
| [evaluate.py](evaluate.py) | Batch evaluation against ground-truth CSV |
| [detector.py](detector.py) | YOLO model loading and plate detection |
| [preprocessor.py](preprocessor.py) | Plate cropping and image enhancement |
| [ocr.py](ocr.py) | PaddleOCR, TrOCR, LLM OCR, zone correction |
| [pipeline.py](pipeline.py) | Orchestrates all steps, frame annotation |

---

## Requirements

- Python 3.10+
- PaddlePaddle 3.x + PaddleOCR
- Ultralytics (YOLOv8)
- `transformers` + `torch` (for TrOCR)
- OpenCV (`cv2`)
- Ollama running locally with `qwen2.5vl:3b` pulled (only needed if using LLM)
