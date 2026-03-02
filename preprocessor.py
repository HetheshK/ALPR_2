# preprocessor.py
"""
Steps 2 & 3 — Plate cropping and image preprocessing.

Exports:
    crop_plate(image, det, ...) -> np.ndarray
    preprocess_crop(crop)       -> np.ndarray
"""

import cv2
import numpy as np


def crop_plate(image: np.ndarray, det: dict,
               h_pad_pct: float = 0.05,
               v_trim_pct: float = 0.18) -> np.ndarray:
    """
    Extract the plate region from the full image.

    Args:
        image:      Full BGR frame
        det:        Detection dict (x1, y1, x2, y2)
        h_pad_pct:  Fraction of box width to add as horizontal padding on each side.
        v_trim_pct: Fraction of box height to remove from the top AND bottom.
                    Eliminates state-name banners (e.g. "TEXAS", "CALIFORNIA")
                    that occupy the top/bottom ~15-20 % of the plate.

    Returns:
        Cropped BGR plate image
    """
    h, w = image.shape[:2]
    bw = det["x2"] - det["x1"]
    bh = det["y2"] - det["y1"]

    px = int(bw * h_pad_pct)
    py = int(bh * v_trim_pct)

    x1 = max(0, det["x1"] - px)
    x2 = min(w, det["x2"] + px)
    y1 = det["y1"] + py          # trim inward from top
    y2 = det["y2"] - py          # trim inward from bottom

    # Guard: if trim is too aggressive (very short box) keep full height
    if y2 <= y1:
        y1, y2 = det["y1"], det["y2"]

    y1 = max(0, y1)
    y2 = min(h, y2)

    return image[y1:y2, x1:x2].copy()


def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """
    Prepare a raw plate crop for OCR.

    Operations (in order):
      1. Normalise  — resize to exactly 400 px wide, min 32 px tall
      2. Denoise    — median blur to remove sensor/compression noise
      3. CLAHE      — contrast enhancement on the L channel (LAB)
      4. Sharpen    — unsharp mask to make character edges crisp for DBNet
      5. Invert     — flip dark-on-light plates so text is always dark-on-white
      6. Border     — add white padding so DBNet doesn't miss edge characters

    Returns a BGR image ready for PaddleOCR.
    """
    img = crop.copy()

    # 1. Normalise width to 400 px, enforce minimum height of 32 px ──────────
    h, w = img.shape[:2]
    target_w = 400
    scale = target_w / w
    new_h = max(32, int(h * scale))
    interp = cv2.INTER_LANCZOS4 if w < target_w else cv2.INTER_AREA
    img = cv2.resize(img, (target_w, new_h), interpolation=interp)

    # 2. Denoise ──────────────────────────────────────────────────────────────
    # Median blur removes salt-and-pepper / JPEG noise without blurring edges.
    # Kernel 3 is sufficient and fast.
    img = cv2.medianBlur(img, 3)

    # 3. CLAHE ────────────────────────────────────────────────────────────────
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. Sharpen (unsharp mask) ───────────────────────────────────────────────
    # Conservative weights (1.5 / -0.5) avoid ringing on already-sharp crops.
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    # 5. Dark-plate inversion ─────────────────────────────────────────────────
    # Some plates have light text on a dark background (e.g. black specialty
    # plates). PaddleOCR's DBNet expects dark text on a light background.
    # Compare mean brightness of a central strip vs the image mean — if the
    # centre is darker than the surroundings the plate is inverted.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape
    center_strip = gray[h_img // 4: 3 * h_img // 4, w_img // 8: 7 * w_img // 8]
    if center_strip.mean() < gray.mean() - 15:
        img = cv2.bitwise_not(img)

    # 6. Border padding ───────────────────────────────────────────────────────
    # A 10 px white border stops DBNet from clipping characters at the edge.
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10,
                             cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return img
