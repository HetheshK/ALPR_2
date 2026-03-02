# ocr.py
"""
Steps 4, 4b, 4c, 4d — OCR pipeline.

Exports:
    load_ocr()                                      -> PaddleOCR instance
    load_llm_ocr(host, model)                       -> dict | None
    load_trocr(model_name)                          -> (processor, model) | None
    load_crnn(model_path)                           -> (model, device) | None
    read_plate(ocr, raw, preprocessed, trocr, ...)  -> (str, float)
    read_plate_trocr_llm(trocr, raw, llm_cfg, ...)  -> (str, float)
    correct_plate_text(text)                        -> str  (sanitise only)
"""

import difflib
import time
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — PaddleOCR
# ─────────────────────────────────────────────────────────────────────────────

def load_ocr():
    """
    Initialise PaddleOCR (English, CPU).

    PaddleOCR v3 runs a DBNet detector → CRNN recogniser pipeline.
    We disable the document-level orientation classifier and unwarping
    steps — they are designed for full A4 pages, not small plate crops,
    and cause a 6 s overhead + empty results on tight crops.
    """
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,  # skip page-level rotation (slow, wrong for crops)
        use_doc_unwarping=False,             # skip perspective unwarp (not needed for plates)
        use_textline_orientation=False,      # skip per-line orientation on already-upright crops
        lang="en",
        device="cpu",
        enable_mkldnn=False,                 # avoids "ConvertPirAttribute" errors on CPU
    )
    print("[Step 4] PaddleOCR engine loaded")
    return ocr


# ─────────────────────────────────────────────────────────────────────────────
# Step 4b — LLM OCR / tiebreaker (Qwen2-VL via Ollama)
# ─────────────────────────────────────────────────────────────────────────────

def load_llm_ocr(host: str = "http://localhost:11434",
                 model: str = "qwen2.5vl:3b") -> dict | None:
    """
    Check whether an Ollama instance with the requested model is reachable.

    Returns {"host": ..., "model": ...} on success, or None if unavailable
    (the LLM pass is then silently skipped — graceful degradation).

    Setup:
        ollama pull qwen2.5vl:3b
    """
    try:
        import urllib.request, json
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as r:
            tags_data = json.loads(r.read())
        available = {m["name"] for m in tags_data.get("models", [])}
        if any(model in t for t in available):
            print(f"[Step 4b] LLM ready: {model} @ {host}")
            return {"host": host, "model": model}
        print(f"[Step 4b] Ollama reachable but '{model}' not found — "
              f"run: ollama pull {model}  (LLM disabled)")
    except Exception as e:
        print(f"[Step 4b] Ollama not reachable ({e}) — LLM disabled")
    return None


def _llm_call(llm_cfg: dict, crop: np.ndarray, prompt: str) -> str:
    """Send a plate crop + prompt to Ollama and return the raw text response."""
    import urllib.request, json, base64

    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    b64 = base64.b64encode(buf.tobytes()).decode()

    payload = {
        "model": llm_cfg["model"],
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [b64],
        }],
        "stream": False,
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{llm_cfg['host']}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    return result["message"]["content"].strip()


def _llm_ocr(llm_cfg: dict, crop: np.ndarray) -> tuple[str, float]:
    """Fresh OCR read via LLM — used when TrOCR is unavailable."""
    prompt = (
        "This is a cropped US license plate. "
        "Read the large alphanumeric characters that form the plate number. "
        "Ignore any small text (state name, slogans, stickers). "
        "Watch for common OCR mistakes: '0' vs 'O', '1' vs 'I', '5' vs 'S', '8' vs 'B'. "
        "Reply with ONLY the plate characters, no spaces, no dashes, no explanation."
    )
    raw = _llm_call(llm_cfg, crop, prompt)
    text = "".join(c for c in raw.upper() if c.isalnum())
    return text, 0.90


def _llm_tiebreak(llm_cfg: dict, crop: np.ndarray,
                  candidate_a: str, candidate_b: str) -> tuple[str, float]:
    """
    Ask the LLM to adjudicate between two conflicting OCR reads.

    Providing both candidates focuses the model on the characters that
    actually differ rather than re-reading the whole plate from scratch.
    """
    a = "".join(c for c in candidate_a if c.isalnum())
    b = "".join(c for c in candidate_b if c.isalnum())
    prompt = (
        f"Two OCR systems read this license plate differently: '{a}' vs '{b}'. "
        "Look carefully at the plate image and decide which is correct. "
        "Watch for: '0'/'O', '1'/'I', '5'/'S', '8'/'B', 'W'/'H', 'M'/'N'. "
        "Reply with ONLY the correct plate characters, no spaces, no explanation."
    )
    raw = _llm_call(llm_cfg, crop, prompt)
    text = "".join(c for c in raw.upper() if c.isalnum())
    return text, 0.92


# ─────────────────────────────────────────────────────────────────────────────
# Step 4c — TrOCR
# ─────────────────────────────────────────────────────────────────────────────

def load_trocr(model_name: str = "microsoft/trocr-base-printed"):
    """
    Load a TrOCR model from HuggingFace.

    Returns (processor, model) on success, or None if transformers/torch
    are not installed (graceful degradation).

    First run downloads ~400 MB; subsequent runs use the local cache.
    Setup:
        pip install transformers torch
    """
    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        model.eval()
        print(f"[Step 4c] TrOCR loaded: {model_name}")
        return (processor, model)
    except ImportError:
        print("[Step 4c] transformers/torch not installed — TrOCR disabled")
    except Exception as e:
        print(f"[Step 4c] TrOCR load failed ({e}) — TrOCR disabled")
    return None


def _trocr_single(trocr, image: np.ndarray) -> tuple[str, float]:
    """Run TrOCR on a BGR crop and return (raw text, confidence).

    Confidence is the mean softmax probability across all generated tokens,
    excluding EOS/PAD. Range 0.0–1.0; higher = model was more certain.
    """
    from PIL import Image
    import torch
    import torch.nn.functional as F

    processor, model = trocr
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            output_scores=True,
            return_dict_in_generate=True,
        )

    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id
    token_probs = [
        F.softmax(score, dim=-1)[0, token_id].item()
        for score, token_id in zip(outputs.scores, outputs.sequences[0][1:])
        if token_id not in (eos_id, pad_id)
    ]
    confidence = float(np.mean(token_probs)) if token_probs else 0.0

    return text, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Step 4d — Custom CRNN + CTC
# ─────────────────────────────────────────────────────────────────────────────

def load_crnn(model_path: str = "crnn_plates.pt"):
    """
    Load a trained CRNN model from a checkpoint file.

    Returns (model, device) on success, or None if unavailable.
    Train with train_crnn.py first.
    """
    from pathlib import Path
    if not Path(model_path).exists():
        print(f"[Step 4d] CRNN weights not found at '{model_path}' — CRNN disabled")
        return None
    try:
        import torch
        from crnn_model import CRNN, NUM_CLASSES
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = CRNN(num_classes=NUM_CLASSES)
        ckpt   = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        model.to(device)
        val_acc = ckpt.get("val_acc", None)
        acc_str = f"  val_acc={val_acc:.1%}" if val_acc else ""
        print(f"[Step 4d] CRNN loaded: {model_path}{acc_str}")
        return (model, device)
    except Exception as e:
        print(f"[Step 4d] CRNN load failed ({e}) — CRNN disabled")
    return None


def _crnn_single(crnn, image: np.ndarray) -> tuple[str, float]:
    """
    Run the custom CRNN on a BGR plate crop.

    Returns (text, confidence) where confidence is the mean softmax
    probability across decoded non-blank characters.
    """
    import torch
    from crnn_model import crnn_decode
    from preprocessor import preprocess_crop

    model, device = crnn

    # Preprocess → grayscale → resize to (128, 32) → tensor
    proc  = preprocess_crop(image)
    gray  = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = tensor.to(device)  # (1, 1, 32, 128)

    with torch.no_grad():
        log_probs = model(tensor)  # (T, 1, C)

    text, confidence = crnn_decode(log_probs[:, 0, :].cpu())
    return text, confidence


# ─────────────────────────────────────────────────────────────────────────────
# DBNet padding + OCR wrapper
# ─────────────────────────────────────────────────────────────────────────────

# State names to strip from OCR output (safety net for any that slip past the crop trim)
_US_STATES = {
    "ALABAMA", "ALASKA", "ARIZONA", "ARKANSAS", "CALIFORNIA", "COLORADO",
    "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA", "HAWAII", "IDAHO",
    "ILLINOIS", "INDIANA", "IOWA", "KANSAS", "KENTUCKY", "LOUISIANA",
    "MAINE", "MARYLAND", "MASSACHUSETTS", "MICHIGAN", "MINNESOTA",
    "MISSISSIPPI", "MISSOURI", "MONTANA", "NEBRASKA", "NEVADA",
    "NEW HAMPSHIRE", "NEW JERSEY", "NEW MEXICO", "NEW YORK",
    "NORTH CAROLINA", "NORTH DAKOTA", "OHIO", "OKLAHOMA", "OREGON",
    "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", "SOUTH DAKOTA",
    "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA", "WASHINGTON",
    "WEST VIRGINIA", "WISCONSIN", "WYOMING",
    "TEX", "CALIF", "CAL", "FLA", "ILL", "PENN", "WASH",
}


def _pad_for_dbnet(image: np.ndarray, pad_frac: float = 0.40) -> np.ndarray:
    """
    Add white padding around the crop so DBNet can reliably detect text at edges.
    40 % of each dimension is added on every side.
    """
    h, w = image.shape[:2]
    pad_y = max(30, int(h * pad_frac))
    pad_x = max(30, int(w * pad_frac))
    return cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))


def _ocr_single(ocr, image: np.ndarray) -> tuple[str, float]:
    """Run OCR on one image and return (text, mean_confidence)."""
    padded = _pad_for_dbnet(image)
    texts, scores = [], []
    for res in ocr.predict(padded):
        if res is None:
            continue
        for text, score in zip(res["rec_texts"], res["rec_scores"]):
            t = str(text).strip()
            alnum = "".join(c for c in t if c.isalnum())
            # Reject: empty, implausibly long, or state name (exact or OCR-mangled match)
            if not alnum or len(alnum) > 10:
                continue
            if (alnum.upper() in _US_STATES
                    or difflib.get_close_matches(alnum.upper(), _US_STATES, n=1, cutoff=0.82)):
                continue
            texts.append(t)
            scores.append(float(score))
    plate_text = "".join(texts)
    confidence = float(np.mean(scores)) if scores else 0.0
    return plate_text, confidence


def run_trocr_only(trocr, image: np.ndarray) -> tuple[str, float]:
    """
    Run TrOCR on an image and return (sanitised plate text, confidence).
    Intended for standalone use — no YOLO or PaddleOCR involved.
    """
    raw, conf = _trocr_single(trocr, image)
    return correct_plate_text(raw), conf


def read_plate_trocr_llm(trocr, raw_crop: np.ndarray,
                         llm_cfg: dict | None = None,
                         llm_threshold: float = 0.85) -> tuple[str, float]:
    """
    Lightweight two-engine pipeline: TrOCR with optional LLM fallback.
    No PaddleOCR involved.

    - Always runs TrOCR on the raw crop.
    - If TrOCR confidence < llm_threshold and llm_cfg is provided,
      the LLM does a fresh read of the raw crop.
    - Returns (plate_text, confidence).
    """
    trocr_raw, trocr_conf = _trocr_single(trocr, raw_crop)
    trocr_text = correct_plate_text(trocr_raw)
    print(f"[TrOCR] '{trocr_text}'  conf={trocr_conf:.3f}")

    if llm_cfg is not None and trocr_conf < llm_threshold:
        try:
            print(f"[TrOCR] conf below {llm_threshold} — calling LLM")
            llm_text, llm_conf = _llm_ocr(llm_cfg, raw_crop)
            if llm_text:
                print(f"[LLM]   '{llm_text}'  conf={llm_conf:.3f}")
                return llm_text, llm_conf
        except Exception as e:
            print(f"[LLM]   failed: {e}")

    return trocr_text, trocr_conf


# ─────────────────────────────────────────────────────────────────────────────
# Step 4e — fast_plate_ocr (ONNX, ~0.6 ms/image)
# ─────────────────────────────────────────────────────────────────────────────

def load_fast_ocr(model_name: str = "cct-s-v1-global-model"):
    """
    Load a fast_plate_ocr LicensePlateRecognizer (ONNX).

    Returns the recognizer on success, or None if unavailable.
    Install with: pip install fast-plate-ocr
    """
    try:
        from fast_plate_ocr import LicensePlateRecognizer
        model = LicensePlateRecognizer(model_name)
        print(f"[Step 4e] fast_plate_ocr loaded: {model_name}")
        return model
    except ImportError as e:
        print(f"[Step 4e] fast_plate_ocr import failed ({e}) — run: pip install fast-plate-ocr")
    except Exception as e:
        print(f"[Step 4e] fast_plate_ocr load failed ({e}) — disabled")
    return None


def _fast_ocr_single(fast_ocr, image: np.ndarray) -> tuple[str, float]:
    """Run fast_plate_ocr on a BGR crop. Returns (text, confidence)."""
    texts, confs = fast_ocr.run(image, return_confidence=True)
    if not texts:
        return "", 0.0
    conf = confs[0]
    conf_val = float(np.mean(conf) if hasattr(conf, "__len__") else conf)
    return str(texts[0]), conf_val


def run_fast_ocr_only(fast_ocr, image: np.ndarray) -> tuple[str, float]:
    """
    Run fast_plate_ocr on an image and return (sanitised plate text, confidence).
    Intended for standalone benchmarking — no YOLO or PaddleOCR involved.
    """
    raw, conf = _fast_ocr_single(fast_ocr, image)
    return correct_plate_text(raw), conf


def correct_plate_text(text: str) -> str:
    """
    Sanitise OCR output to valid plate characters.

    Strips everything except alphanumerics and hyphens, and uppercases.
    Spaces are OCR artifacts from DBNet gap detection and are removed;
    hyphens are real plate separators and are preserved.
    """
    return "".join(c for c in text.upper() if c.isalnum() or c == '-')


def read_plate(ocr, raw_crop: np.ndarray, preprocessed_crop: np.ndarray,
               fallback_threshold: float = 0.5,
               trocr=None,
               crnn=None,
               fast_ocr=None,
               llm_cfg: dict | None = None,
               llm_threshold: float = 0.85) -> tuple[str, float]:
    """
    Multi-pass OCR with cross-engine validation and LLM tiebreaking.

    Pass A  — PaddleOCR on fully preprocessed crop (CLAHE + sharpen).
    Pass B  — PaddleOCR on plain resize. Runs when pass A conf < fallback_threshold.
    Pass C  — CRNN (custom) on raw crop. Runs when crnn is not None.
    Pass C2 — TrOCR on raw crop. Runs when trocr is not None.
    Pass C3 — fast_plate_ocr (ONNX) on raw crop. Runs when fast_ocr is not None.
              Cross-check logic (engines A/C/C2/C3):
              • Paddle + any secondary agree → return paddle (boosted conf).
              • Any two secondaries agree → return that result (boosted conf).
              • All disagree → LLM tiebreak if available, else highest-confidence wins.
    Pass D  — LLM tiebreak (Qwen2-VL). Only on mismatch.
    Fallback— If no secondary engine, LLM runs when PaddleOCR conf < llm_threshold.

    Returns:
        (plate_text, confidence)  — confidence is 0.0 if nothing was read
    """
    # ── Passes A / B  (PaddleOCR) ────────────────────────────────────────────
    text_a, conf_a = _ocr_single(ocr, preprocessed_crop)
    print(f"[Step 4a] PaddleOCR pass A → '{text_a}'  conf={conf_a:.3f}")

    if text_a and conf_a >= fallback_threshold:
        best_text, best_conf = text_a, conf_a
    else:
        h, w = raw_crop.shape[:2]
        target_w = 400
        if w != target_w:
            scale = target_w / w
            interp = cv2.INTER_LANCZOS4 if w < target_w else cv2.INTER_AREA
            plain = cv2.resize(raw_crop, (target_w, max(1, int(h * scale))),
                               interpolation=interp)
        else:
            plain = raw_crop.copy()
        text_b, conf_b = _ocr_single(ocr, plain)
        print(f"[Step 4a] PaddleOCR pass B → '{text_b}'  conf={conf_b:.3f}")
        best_text = text_b if conf_b > conf_a else text_a
        best_conf = max(conf_a, conf_b)

    paddle_text = correct_plate_text(best_text)
    print(f"[Step 4a] PaddleOCR best   → '{paddle_text}'  conf={best_conf:.3f}")

    paddle_alnum = "".join(c for c in paddle_text if c.isalnum())

    # ── Pass C  (CRNN) ────────────────────────────────────────────────────────
    crnn_text, crnn_conf = "", 0.0
    if crnn is not None:
        try:
            t_crnn = time.time()
            crnn_raw, crnn_conf = _crnn_single(crnn, raw_crop)
            crnn_text = correct_plate_text(crnn_raw)
            print(f"[Step 4d] CRNN          → '{crnn_text}'  conf={crnn_conf:.3f}  ({(time.time()-t_crnn)*1000:.0f}ms)")
        except Exception as e:
            print(f"[Step 4d] CRNN failed: {e}")
            crnn_text, crnn_conf = "", 0.0

    # ── Pass C2  (TrOCR) ─────────────────────────────────────────────────────
    trocr_text, trocr_conf = "", 0.0
    if trocr is not None:
        try:
            t_trocr = time.time()
            trocr_raw, trocr_conf = _trocr_single(trocr, raw_crop)
            trocr_text = correct_plate_text(trocr_raw)
            print(f"[Step 4c] TrOCR         → '{trocr_text}'  conf={trocr_conf:.3f}  ({(time.time()-t_trocr)*1000:.0f}ms)")
        except Exception as e:
            print(f"[Step 4c] TrOCR failed: {e}")
            trocr_text, trocr_conf = "", 0.0

    # ── Pass C3  (fast_plate_ocr) ─────────────────────────────────────────────
    fast_text, fast_conf = "", 0.0
    if fast_ocr is not None:
        try:
            t_fast = time.time()
            fast_raw, fast_conf = _fast_ocr_single(fast_ocr, raw_crop)
            fast_text = correct_plate_text(fast_raw)
            print(f"[Step 4e] fast_plate_ocr → '{fast_text}'  conf={fast_conf:.3f}  ({(time.time()-t_fast)*1000:.0f}ms)")
        except Exception as e:
            print(f"[Step 4e] fast_plate_ocr failed: {e}")
            fast_text, fast_conf = "", 0.0

    # ── Cross-check: look for agreement across all engines ───────────────────
    crnn_alnum  = "".join(c for c in crnn_text  if c.isalnum())
    trocr_alnum = "".join(c for c in trocr_text if c.isalnum())
    fast_alnum  = "".join(c for c in fast_text  if c.isalnum())

    secondary_engines = [(crnn_text,  crnn_alnum,  crnn_conf),
                         (trocr_text, trocr_alnum, trocr_conf),
                         (fast_text,  fast_alnum,  fast_conf)]
    secondary_engines = [(t, a, c) for t, a, c in secondary_engines if a]

    if secondary_engines:
        # Check if any secondary engine agrees with PaddleOCR
        for sec_text, sec_alnum, sec_conf in secondary_engines:
            if paddle_alnum and paddle_alnum == sec_alnum:
                print(f"[Step 4] Agreement: paddle+engine → '{paddle_text}'")
                return paddle_text, min(best_conf + 0.05, 1.0)

        # Check if any two secondary engines agree with each other
        for i, (t1, a1, c1) in enumerate(secondary_engines):
            for t2, a2, c2 in secondary_engines[i + 1:]:
                if a1 == a2:
                    winner_text = t1 if c1 >= c2 else t2
                    winner_conf = max(c1, c2)
                    print(f"[Step 4] Secondary agreement → '{winner_text}'")
                    return winner_text, min(winner_conf + 0.05, 1.0)

        # Full mismatch — try LLM tiebreak
        candidates = [t for t, a, c in [(paddle_text, paddle_alnum, best_conf)]
                      + secondary_engines if a]
        if len(candidates) >= 2 and llm_cfg is not None:
            try:
                t_llm = time.time()
                text_d, conf_d = _llm_tiebreak(llm_cfg, raw_crop,
                                               candidates[0], candidates[1])
                print(f"[Step 4e] LLM tiebreak  → '{text_d}'  ({(time.time()-t_llm)*1000:.0f}ms)")
                if text_d:
                    return correct_plate_text(text_d), conf_d
            except Exception as e:
                print(f"[Step 4e] LLM tiebreak failed: {e}")

        # No LLM — highest-confidence engine wins
        all_candidates = (
            [(paddle_text, best_conf)] if paddle_alnum else []
        ) + [(t, c) for t, a, c in secondary_engines]
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        return all_candidates[0]

    # ── No secondary engines — LLM fallback on low Paddle confidence ─────────
    if llm_cfg is not None and best_conf < llm_threshold:
        try:
            t_llm = time.time()
            text_c, conf_c = _llm_ocr(llm_cfg, raw_crop)
            print(f"[Step 4b] LLM OCR       → '{text_c}'  ({(time.time()-t_llm)*1000:.0f}ms)")
            if text_c:
                return correct_plate_text(text_c), conf_c
        except Exception as e:
            print(f"[Step 4b] LLM OCR failed: {e}")

    return paddle_text, best_conf
