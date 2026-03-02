# api.py
"""
FastAPI wrapper for the ALPR pipeline.

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health            — liveness + which models are loaded
    POST /read-plate        — multipart image file upload
    POST /read-plate/base64 — JSON body with base64-encoded image
"""

# Must be set before PaddleOCR imports
import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import base64
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from detector import load_detector, detect_plates
from ocr import load_ocr, load_llm_ocr, load_trocr, load_crnn, load_fast_ocr, read_plate
from preprocessor import crop_plate, preprocess_crop


# ── Config (override via environment variables) ───────────────────────────────

YOLO_WEIGHTS   = os.getenv("ALPR_YOLO",       "yolo_plates.pt")
CRNN_WEIGHTS   = os.getenv("ALPR_CRNN",        "crnn_plates.pt")
TROCR_MODEL    = os.getenv("ALPR_TROCR",       "microsoft/trocr-base-printed")
LLM_MODEL      = os.getenv("ALPR_LLM_MODEL",   "qwen2.5vl:3b")
YOLO_CONF      = float(os.getenv("ALPR_CONF",  "0.5"))
LLM_THRESHOLD  = float(os.getenv("ALPR_LLM_THRESHOLD", "0.85"))
ENABLE_TROCR    = os.getenv("ALPR_NO_TROCR",    "0") != "1"
ENABLE_CRNN     = os.getenv("ALPR_NO_CRNN",    "0") != "1"
ENABLE_LLM      = os.getenv("ALPR_NO_LLM",     "0") != "1"
ENABLE_FAST_OCR = os.getenv("ALPR_NO_FAST_OCR","0") != "1"
FAST_OCR_MODEL  = os.getenv("ALPR_FAST_OCR_MODEL", "cct-s-v1-global-model")


# ── Model state (shared across requests) ─────────────────────────────────────

_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models once at startup; release on shutdown."""
    print("Loading ALPR models...")
    t0 = time.time()

    _models["yolo"]     = load_detector(YOLO_WEIGHTS)
    _models["ocr"]      = load_ocr()
    _models["trocr"]    = load_trocr(model_name=TROCR_MODEL)       if ENABLE_TROCR    else None
    _models["crnn"]     = load_crnn(model_path=CRNN_WEIGHTS)        if ENABLE_CRNN     else None
    _models["fast_ocr"] = load_fast_ocr(model_name=FAST_OCR_MODEL) if ENABLE_FAST_OCR else None
    _models["llm"]      = load_llm_ocr(model=LLM_MODEL)            if ENABLE_LLM      else None

    print(f"Models ready in {time.time() - t0:.1f}s")
    yield
    _models.clear()


app = FastAPI(
    title="ALPR API",
    description="Automatic License Plate Recognition — YOLO + PaddleOCR + CRNN + TrOCR",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Shared processing logic ───────────────────────────────────────────────────

def _run_on_image(image: np.ndarray,
                  conf: float = YOLO_CONF,
                  skip_detection: bool = False) -> dict:
    """
    Run the full pipeline on a decoded BGR image.
    Returns a dict ready to be returned as JSON.
    """
    t_start = time.time()

    if skip_detection:
        # Image is already a plate crop (e.g. Lakh/ dataset)
        raw_crop     = image
        preprocessed = preprocess_crop(raw_crop)
        plate_text, plate_conf = read_plate(
            _models["ocr"], raw_crop, preprocessed,
            trocr=_models["trocr"],
            crnn=_models["crnn"],
            fast_ocr=_models["fast_ocr"],
            llm_cfg=_models["llm"],
            llm_threshold=LLM_THRESHOLD,
        )
        return {
            "plate":      plate_text or None,
            "confidence": round(plate_conf, 3) if plate_text else None,
            "bbox":       None,
            "elapsed_ms": round((time.time() - t_start) * 1000),
        }

    # Normal mode: YOLO → crop → OCR
    detections = detect_plates(_models["yolo"], image, conf=conf)
    if not detections:
        return {
            "plate":      None,
            "confidence": None,
            "bbox":       None,
            "elapsed_ms": round((time.time() - t_start) * 1000),
            "detail":     "No plate detected",
        }

    # Take highest-confidence detection
    det        = detections[0]
    raw_crop   = crop_plate(image, det)
    preprocessed = preprocess_crop(raw_crop)

    plate_text, plate_conf = read_plate(
        _models["ocr"], raw_crop, preprocessed,
        trocr=_models["trocr"],
        crnn=_models["crnn"],
        fast_ocr=_models["fast_ocr"],
        llm_cfg=_models["llm"],
        llm_threshold=LLM_THRESHOLD,
    )

    return {
        "plate":      plate_text or None,
        "confidence": round(plate_conf, 3) if plate_text else None,
        "bbox": {
            "x1": det["x1"], "y1": det["y1"],
            "x2": det["x2"], "y2": det["y2"],
            "det_confidence": round(det["confidence"], 3),
        },
        "elapsed_ms": round((time.time() - t_start) * 1000),
    }


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — returns which engines are loaded."""
    return {
        "status": "ok",
        "engines": {
            "yolo":     _models.get("yolo")     is not None,
            "paddle":   _models.get("ocr")      is not None,
            "crnn":     _models.get("crnn")     is not None,
            "trocr":    _models.get("trocr")    is not None,
            "fast_ocr": _models.get("fast_ocr") is not None,
            "llm":      _models.get("llm")      is not None,
        },
    }


@app.post("/read-plate")
async def read_plate_upload(
    file: UploadFile = File(...),
    conf: float = Query(default=YOLO_CONF, description="YOLO detection threshold"),
    skip_detection: bool = Query(default=False, description="Treat image as pre-cropped plate"),
):
    """
    Upload an image file and read its license plate.

    - **file**: JPEG / PNG / BMP image
    - **conf**: YOLO confidence threshold (default 0.5)
    - **skip_detection**: set true if image is already a plate crop
    """
    data = await file.read()
    image = _decode_image(data)
    return JSONResponse(_run_on_image(image, conf=conf, skip_detection=skip_detection))


class Base64Request(BaseModel):
    image: str          # base64-encoded image bytes
    conf: float = YOLO_CONF
    skip_detection: bool = False


@app.post("/read-plate/base64")
def read_plate_base64(body: Base64Request):
    """
    Submit a base64-encoded image and read its license plate.
    Useful for programmatic clients that don't do multipart.

    ```json
    {
      "image": "<base64 string>",
      "conf": 0.5,
      "skip_detection": false
    }
    ```
    """
    try:
        data = base64.b64decode(body.image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 data")
    image = _decode_image(data)
    return JSONResponse(_run_on_image(image, conf=body.conf,
                                      skip_detection=body.skip_detection))
