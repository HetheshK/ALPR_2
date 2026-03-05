"""
train_state.py — Train an EfficientNet-B0 state classifier on Lakh plate images.

Usage:
    python train_state.py --device cuda --epochs 30
    python train_state.py --device cpu  --epochs 10 --limit 2000

The model learns to classify US states from cropped plate images.
Texas (94% of raw data) is capped at --texas-cap (default 3000) to prevent
it from overwhelming all other states during training.

Output: state_classifier.pt  (checkpoint with class list embedded)
"""

import argparse
import csv
import os
import random
import time
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

# ── Canonical label normalisation ────────────────────────────────────────────
# Maps common OCR/typo variants → correct state name.
LABEL_MAP = {
    "SOUTH CAROLNNA":   "SOUTH CAROLINA",
    "NORTH CAROLONA":   "NORTH CAROLINA",
    "NORTH CAROLIONA":  "NORTH CAROLINA",
    "FORIDA":           "FLORIDA",
    "SOUTH DAK0TA":     "SOUTH DAKOTA",
    "WISGONSIN":        "WISCONSIN",
    "MISSSSPPI":        "MISSISSIPPI",
    "COLORAD0":         "COLORADO",
    "MIONESOTA":        "MINNESOTA",
    "ALASAKA":          "ALASKA",
    "MASSASCHUSETTS":   "MASSACHUSETTS",
    "CONNECTICT":       "CONNECTICUT",
    "TENNESEE":         "TENNESSEE",
    "TENNESSE":         "TENNESSEE",
    "GEORIGA":          "GEORGIA",
    "VIRGINA":          "VIRGINIA",
    "OKLAHOME":         "OKLAHOMA",
    "CALIFRONIA":       "CALIFORNIA",
    "NEVAD":            "NEVADA",
    "MONTNA":           "MONTANA",
}

MIN_SAMPLES = 50   # drop states with fewer than this after normalisation


# ── Dataset ───────────────────────────────────────────────────────────────────

class PlateStateDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], lakh_dir: str,
                 transform=None):
        self.samples   = samples   # [(image_path, class_idx), ...]
        self.lakh_dir  = lakh_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        path = os.path.join(self.lakh_dir, fname)
        img  = cv2.imread(path)
        if img is None:
            # Return blank if image is missing
            img = np.zeros((64, 128, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            from PIL import Image as PILImage
            img = PILImage.fromarray(img)
            img = self.transform(img)
        return img, label


# ── Data loading ──────────────────────────────────────────────────────────────

def load_samples(csv_path: str, lakh_dir: str,
                 texas_cap: int, limit: int) -> tuple[list, list[str]]:
    """
    Returns (samples, class_names) where samples = [(filename, class_idx), ...].
    """
    raw: dict[str, list[str]] = defaultdict(list)   # state → [filename, ...]

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            fname = row[0].strip()
            state = LABEL_MAP.get(row[1].strip().upper(),
                                  row[1].strip().upper())
            if not fname or not state:
                continue
            full_path = os.path.join(lakh_dir, fname)
            if os.path.exists(full_path):
                raw[state].append(fname)

    # Drop states below MIN_SAMPLES
    raw = {s: fnames for s, fnames in raw.items() if len(fnames) >= MIN_SAMPLES}

    # Cap Texas
    if "TEXAS" in raw and len(raw["TEXAS"]) > texas_cap:
        random.shuffle(raw["TEXAS"])
        raw["TEXAS"] = raw["TEXAS"][:texas_cap]

    # Build class list (sorted for reproducibility)
    class_names = sorted(raw.keys())
    state_to_idx = {s: i for i, s in enumerate(class_names)}

    # Flatten to (filename, label) list
    all_samples: list[tuple[str, int]] = []
    for state, fnames in raw.items():
        idx = state_to_idx[state]
        for fname in fnames:
            all_samples.append((fname, idx))

    random.shuffle(all_samples)
    if limit:
        all_samples = all_samples[:limit]

    return all_samples, class_names


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    print(f"Device: {device}")

    # ── Load samples ──────────────────────────────────────────────────────────
    print("Loading dataset...")
    all_samples, class_names = load_samples(
        args.labels, args.lakh, args.texas_cap, args.limit)
    num_classes = len(class_names)

    counts = Counter(label for _, label in all_samples)
    print(f"Classes  : {num_classes}")
    print(f"Samples  : {len(all_samples)}")
    print(f"Texas cap: {args.texas_cap}  (raw Texas = counts.get(class_names.index('TEXAS') if 'TEXAS' in class_names else -1, 0))")
    for i, name in enumerate(class_names):
        print(f"  [{i:2d}] {name:<25} {counts[i]:>5}")

    # ── Train / val split ─────────────────────────────────────────────────────
    val_n  = max(1, int(len(all_samples) * args.val_split))
    val_samples   = all_samples[:val_n]
    train_samples = all_samples[val_n:]
    print(f"\nTrain: {len(train_samples)}  Val: {len(val_samples)}")

    # ── Transforms ────────────────────────────────────────────────────────────
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((128, 256)),
        transforms.RandomHorizontalFlip(p=0.0),   # plates shouldn't be flipped
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((128, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = PlateStateDataset(train_samples, args.lakh, train_tf)
    val_ds   = PlateStateDataset(val_samples,   args.lakh, val_tf)

    # ── Weighted sampler (balance classes in each batch) ─────────────────────
    class_counts = [counts.get(i, 1) for i in range(num_classes)]
    weights = [1.0 / class_counts[label] for _, label in train_samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_samples),
                                    replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes).to(device)

    # Weight loss by inverse class frequency (extra safeguard vs imbalance)
    class_weights = torch.tensor(
        [1.0 / max(counts.get(i, 1), 1) for i in range(num_classes)],
        dtype=torch.float32, device=device)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_path    = args.out

    print(f"\nTraining for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train epoch ───────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total   += imgs.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc   = train_correct / train_total

        # ── Val epoch ─────────────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)
        val_acc = val_correct / val_total if val_total else 0.0

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={train_loss:.4f}  train={train_acc*100:.1f}%  "
              f"val={val_acc*100:.1f}%  ({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(),
                        "classes": class_names,
                        "num_classes": num_classes},
                       best_path)
            print(f"  → saved best ({val_acc*100:.1f}%)")

    print(f"\nBest val accuracy: {best_val_acc*100:.2f}%")
    print(f"Checkpoint saved : {best_path}")
    return class_names


# ── Inference helper (importable) ─────────────────────────────────────────────

def load_state_classifier(path: str = "state_classifier.pt", device: str = "cpu"):
    """Load a saved state classifier. Returns (model, class_names) or (None, [])."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        class_names = ckpt["classes"]
        model = build_model(len(class_names))
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        print(f"[State] Loaded state classifier: {len(class_names)} classes from {path}")
        return model, class_names
    except FileNotFoundError:
        print(f"[State] No state classifier found at {path}")
        return None, []
    except Exception as e:
        print(f"[State] Failed to load state classifier ({e})")
        return None, []


def predict_state(model, class_names: list[str], image: np.ndarray,
                  device: str = "cpu") -> tuple[str, float]:
    """
    Predict US state from a BGR plate image.
    Returns (state_name, confidence).
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((128, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    from PIL import Image as PILImage
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = tf(PILImage.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
    return class_names[idx.item()], round(conf.item(), 4)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train state classifier on Lakh plates")
    ap.add_argument("--lakh",      default="Lakh",
                    help="Folder of plate images")
    ap.add_argument("--labels",    default="TRAINDATA - cleaned_output.csv.csv",
                    help="Ground-truth CSV (filename, state, plate)")
    ap.add_argument("--out",       default="state_classifier.pt",
                    help="Output checkpoint path")
    ap.add_argument("--epochs",    type=int,   default=30)
    ap.add_argument("--batch",     type=int,   default=64)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.15,
                    dest="val_split")
    ap.add_argument("--texas-cap", type=int,   default=3000,
                    dest="texas_cap",
                    help="Max Texas samples to include (default 3000)")
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--limit",     type=int,   default=0,
                    help="Cap total samples (0 = all). Useful for smoke-tests.")
    args = ap.parse_args()

    train(args)


if __name__ == "__main__":
    main()
