# ALPR_2 — Automatic License Plate Recognition

A multi-engine OCR pipeline for reading US license plates from images or a live webcam feed.

## Pipeline Overview

```
Image → YOLO detection → Crop + Preprocess → PaddleOCR  ─┐
                                           → CRNN+CTC    ─┼─ 2-of-3 majority vote
                                           → TrOCR       ─┘
                                           → Qwen2-VL LLM (tiebreak on 3-way split)
                                           → Zone-corrected text output
```

1. **YOLO** detects the plate bounding box
2. **Preprocessor** crops, denoises (median blur), enhances contrast (CLAHE), sharpens, inverts dark plates, Otsu-binarises, and pads
3. **PaddleOCR** reads the preprocessed crop (two passes: preprocessed + plain resize)
4. **CRNN+CTC** reads the same crop as a fast cross-check engine (custom-trained on Lakh dataset)
5. **TrOCR** provides a third opinion via ViT encoder (optional, slower)
6. **2-of-3 majority vote** — if any two engines agree, that result wins
7. If all three disagree → **Qwen2-VL LLM** (via Ollama) makes the final call
8. **Zone correction** fixes common OCR confusions (e.g. `0`↔`O`, `1`↔`I`, `8`↔`B`)

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

Use a different set of YOLO weights. Defaults to `yolo_plates.pt`.

### OCR engine flags

```bash
python detect_plate.py --image car.jpg --no-trocr
```

Disable TrOCR. Faster; voting falls back to PaddleOCR + CRNN 2-of-2.

```bash
python detect_plate.py --image car.jpg --trocr-model microsoft/trocr-large-printed
```

Use a different HuggingFace TrOCR model (default: `microsoft/trocr-base-printed`). The `large` variant is more accurate but slower.

```bash
python detect_plate.py --image car.jpg --no-crnn
```

Disable the CRNN cross-check engine.

```bash
python detect_plate.py --image car.jpg --crnn-model path/to/crnn_plates.pt
```

Use a custom CRNN checkpoint (default: `crnn_plates.pt`).

### LLM flags

```bash
python detect_plate.py --image car.jpg --no-llm
```

Disable the Qwen2-VL LLM entirely. If all engines disagree, the highest-confidence result wins.

```bash
python detect_plate.py --image car.jpg --llm-model qwen2.5vl:7b
```

Use a different Ollama vision model (default: `qwen2.5vl:3b`). Larger models are more accurate but slower. Requires Ollama to be running.

```bash
python detect_plate.py --image car.jpg --llm-threshold 0.75
```

LLM fires when PaddleOCR confidence drops below this threshold (default: `0.85`).

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
python evaluate.py --trocr-only
```

Skip PaddleOCR and CRNN — benchmark TrOCR in isolation. YOLO still runs to find the plate crop.

```bash
python evaluate.py --no-crnn
```

Disable the CRNN cross-check engine.

```bash
python evaluate.py --crnn-model path/to/crnn_plates.pt
```

Use a custom CRNN checkpoint (default: `crnn_plates.pt`).

```bash
python evaluate.py --llm
```

Enable the Qwen2-VL LLM tiebreaker (disabled by default in evaluate mode). Requires Ollama running with the model loaded.

```bash
python evaluate.py --limit 100
```

Stop after 100 images. Useful for quick spot-checks without running the full dataset.

---

## confusion.py — Character-level confusion matrix

Reads `eval_results.csv` and shows which characters the OCR engine commonly mistakes for others. Run this after an `evaluate.py` pass.

```bash
python confusion.py
```

Lists the top 20 character confusion pairs (e.g. `O → 0`, `I → 1`, `8 → B`).

```bash
python confusion.py --top 30
```

Show more pairs.

```bash
python confusion.py --matrix
```

Also print a full character grid (rows = actual, columns = predicted).

```bash
python confusion.py --input my_results.csv
```

Use a different results CSV.

---

## train_crnn.py — Train the CRNN+CTC engine

Trains the custom CRNN directly from the `Lakh/` folder (images are already plate crops).

```bash
python train_crnn.py --device cuda --epochs 40
```

Trains for 40 epochs on GPU. Saves the best checkpoint (by validation loss) to `crnn_plates.pt`.

```bash
python train_crnn.py --device cpu --epochs 40 --limit 5000
```

CPU training on a subset — useful for smoke-testing the pipeline.

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--lakh` | `Lakh` | Folder of plate crop images |
| `--labels` | `TRAINDATA - cleaned_output.csv.csv` | Ground-truth CSV |
| `--out` | `crnn_plates.pt` | Output checkpoint path |
| `--epochs` | `40` | Training epochs |
| `--batch` | `64` | Batch size |
| `--lr` | `1e-3` | Initial learning rate |
| `--val-split` | `0.1` | Fraction held out for validation |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--limit` | `0` (all) | Cap number of training images |

---

## Preprocessing steps

Each plate crop goes through these steps before OCR:

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | Resize to 400 px wide | Normalise input scale |
| 2 | Median blur (3×3) | Remove sensor / JPEG noise |
| 3 | CLAHE on L channel | Boost local contrast |
| 4 | Unsharp mask | Sharpen character edges for DBNet |
| 5 | Dark-plate inversion | Ensure dark text on light background |
| 6 | Otsu binarisation | Clean binary image — helps TrOCR ViT encoder |
| 7 | 10 px white border | Prevent DBNet from clipping edge characters |

---

## Module reference

| File | Role |
|------|------|
| [detect_plate.py](detect_plate.py) | CLI entry point for image / webcam |
| [evaluate.py](evaluate.py) | Batch evaluation against ground-truth CSV |
| [confusion.py](confusion.py) | Character confusion matrix from eval results |
| [train_crnn.py](train_crnn.py) | Train CRNN+CTC on Lakh dataset |
| [crnn_model.py](crnn_model.py) | CRNN architecture (CNN + BiLSTM + CTC) |
| [detector.py](detector.py) | YOLO model loading and plate detection |
| [preprocessor.py](preprocessor.py) | Plate cropping and image enhancement |
| [ocr.py](ocr.py) | PaddleOCR, CRNN, TrOCR, LLM OCR, zone correction |
| [pipeline.py](pipeline.py) | Orchestrates all steps, frame annotation |

---

## Requirements

- Python 3.10+
- PaddlePaddle 3.x + PaddleOCR
- PyTorch (for CRNN + TrOCR)
- `transformers` (for TrOCR — `microsoft/trocr-base-printed`)
- OpenCV (`cv2`)
- Ollama running locally with `qwen2.5vl:3b` pulled (only needed if using LLM)
