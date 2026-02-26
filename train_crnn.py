# train_crnn.py
"""
Train the CRNN + CTC model on the Lakh plate crops.

The images in Lakh/ are already cropped plates — no YOLO needed.
Loads images directly from Lakh/ using TRAINDATA CSV as labels.

Usage:
    python train_crnn.py                          # default settings
    python train_crnn.py --device cuda            # GPU training
    python train_crnn.py --epochs 5 --limit 1000  # quick smoke test
    python train_crnn.py --resume crnn_plates.pt # continue from checkpoint
"""

import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from crnn_model import CRNN, CHAR2IDX, BLANK, NUM_CLASSES, crnn_decode
from preprocessor import preprocess_crop

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Dataset ───────────────────────────────────────────────────────────────────

class PlateDataset(Dataset):
    """
    Loads plate images directly from the Lakh/ directory.

    Each item returns:
        image  : (1, 32, 128) normalised grayscale tensor
        target : 1-D int tensor of character indices (1-indexed, no blank)
        length : number of characters in target
    """

    def __init__(self, rows: list[dict], augment: bool = False):
        self.rows    = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row   = self.rows[idx]
        image = cv2.imread(row["path"])

        if image is None:
            image = np.ones((32, 128, 3), dtype=np.uint8) * 255
            label = "A"
        else:
            label = row["label"]
            image = preprocess_crop(image)

        if self.augment:
            image = _augment(image)

        gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray   = cv2.resize(gray, (128, 32), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(gray).float().unsqueeze(0) / 255.0  # (1,32,128)

        target = [CHAR2IDX[c] for c in label.upper() if c in CHAR2IDX]
        if not target:
            target = [CHAR2IDX.get("A", 1)]

        return tensor, torch.tensor(target, dtype=torch.long), len(target)


def _augment(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]

    if random.random() < 0.4:
        angle = random.uniform(-3, 3)
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.3:
        noise = np.random.normal(0, 5, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.4:
        factor = random.uniform(0.8, 1.2)
        image  = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return image


# ── CTC collate ───────────────────────────────────────────────────────────────

def ctc_collate(batch):
    images, targets, lengths = zip(*batch)
    images         = torch.stack(images)
    targets_concat = torch.cat(targets)
    target_lengths = torch.tensor(lengths, dtype=torch.long)
    input_lengths  = torch.full((len(images),), 32, dtype=torch.long)
    return images, targets_concat, input_lengths, target_lengths


# ── Training ──────────────────────────────────────────────────────────────────

def exact_match_accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, targets_flat, input_lengths, target_lengths in loader:
            images = images.to(device)
            log_probs = model(images)  # (T, B, C)

            offset = 0
            for b in range(images.size(0)):
                tlen   = target_lengths[b].item()
                gt     = targets_flat[offset: offset + tlen].tolist()
                gt_str = "".join(
                    chr(ord("A") + i - 1) if i <= 26 else str(i - 27)
                    for i in gt
                )
                offset += tlen

                pred_str, _ = crnn_decode(log_probs[:, b, :].cpu())
                if pred_str == gt_str:
                    correct += 1
                total += 1

    model.train()
    return correct / total if total else 0.0


def load_dataset(lakh_dir: str, labels_csv: str) -> list[dict]:
    """Load ground truth from CSV and match to images in lakh_dir."""
    lakh = Path(lakh_dir)

    gt = {}
    with open(labels_csv, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[0].strip() and row[2].strip():
                fname = row[0].strip()
                label_raw = row[2].strip().upper()
                label_alnum = "".join(c for c in label_raw if c in CHAR2IDX)
                if label_alnum:
                    gt[fname] = label_alnum  # last label wins on duplicates

    rows = []
    for fname, label in gt.items():
        path = lakh / fname
        if path.exists():
            rows.append({"path": str(path), "label": label})

    return rows


def train(args):
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available — falling back to CPU")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading dataset from {args.lakh} + {args.labels}...")
    rows = load_dataset(args.lakh, args.labels)
    print(f"Found {len(rows)} labeled images")

    if not rows:
        print("ERROR: No matching images found. Check --lakh and --labels paths.")
        return

    if args.limit:
        random.shuffle(rows)
        rows = rows[:args.limit]

    random.shuffle(rows)
    split      = int(len(rows) * (1 - args.val_split))
    train_rows = rows[:split]
    val_rows   = rows[split:]

    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}")

    train_ds = PlateDataset(train_rows, augment=True)
    val_ds   = PlateDataset(val_rows,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, collate_fn=ctc_collate,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0, collate_fn=ctc_collate)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CRNN(num_classes=NUM_CLASSES).to(device)

    start_epoch   = 0
    best_val_loss = float("inf")

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch   = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from {args.resume}  (epoch {start_epoch}, "
              f"best_val_loss={best_val_loss:.4f})")

    criterion = nn.CTCLoss(blank=BLANK, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-5)

    use_amp = (device.type == "cuda")
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for images, targets, input_lengths, target_lengths in train_loader:
            images  = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                log_probs = model(images)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets, input_lengths, target_lengths in val_loader:
                images  = images.to(device)
                targets = targets.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    log_probs = model(images)
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc      = exact_match_accuracy(model, val_loader, device)
        scheduler.step(avg_val_loss)
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{start_epoch+args.epochs}  "
              f"train={avg_train_loss:.4f}  val={avg_val_loss:.4f}  "
              f"exact={val_acc:.1%}  lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"({elapsed:.0f}s)")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model":    model.state_dict(),
                "epoch":    epoch + 1,
                "val_loss": best_val_loss,
                "val_acc":  val_acc,
            }, args.out)
            print(f"  ✓ Saved best model → {args.out}  (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best model: {args.out}")


def main():
    parser = argparse.ArgumentParser(description="Train CRNN + CTC on license plates")
    parser.add_argument("--lakh",      default="Lakh",
                        help="Folder of plate crop images (default: Lakh/)")
    parser.add_argument("--labels",    default="TRAINDATA - cleaned_output.csv.csv",
                        help="Ground-truth CSV (default: TRAINDATA - cleaned_output.csv.csv)")
    parser.add_argument("--out",       default="crnn_plates.pt",
                        help="Output checkpoint path (default: crnn_plates.pt)")
    parser.add_argument("--resume",    default="",
                        help="Resume from checkpoint path")
    parser.add_argument("--epochs",    type=int, default=40)
    parser.add_argument("--batch",     type=int, default=64)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1, dest="val_split",
                        help="Fraction of data for validation (default: 0.1)")
    parser.add_argument("--device",    default="cuda",
                        help="Training device: cuda or cpu (default: cuda)")
    parser.add_argument("--limit",     type=int, default=0,
                        help="Limit dataset size for quick tests (0 = all)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
