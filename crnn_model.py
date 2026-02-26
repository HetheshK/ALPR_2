# crnn_model.py
"""
CRNN — Convolutional Recurrent Neural Network for license-plate OCR.

Architecture:
    Input  : (B, 1, 32, 128)  grayscale image, height=32, width=128
    CNN    : 4 conv blocks → feature map (B, 256, 1, 32)
    RNN    : 2-layer BiLSTM, hidden=256  → (32, B, 512)
    FC     : Linear(512 → num_classes)
    Output : (T=32, B, num_classes)  log-softmax — ready for nn.CTCLoss

Character set (37 classes):
    Index 0       : CTC blank
    Index 1 – 26  : A–Z
    Index 27 – 36 : 0–9

Exports:
    CHARS          — str of all non-blank characters (len 36)
    CHAR2IDX       — {char: index}  (1-indexed, blank=0)
    IDX2CHAR       — {index: char}
    CRNN           — the model class
    crnn_decode    — greedy CTC decode → (text, confidence)
"""

import string
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Character vocabulary ──────────────────────────────────────────────────────
CHARS    = string.ascii_uppercase + string.digits   # 36 chars: A-Z 0-9
BLANK    = 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 1-indexed
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # 37 (includes blank)


# ── Model ─────────────────────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=None):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(pool))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CRNN(nn.Module):
    """
    CRNN for variable-length license-plate text recognition.

    Input shape  : (B, 1, 32, 128)
    Output shape : (T, B, NUM_CLASSES)  — log-softmax over vocab
    """

    def __init__(self, num_classes: int = NUM_CLASSES, rnn_hidden: int = 256):
        super().__init__()

        # ── CNN backbone ──────────────────────────────────────────────────────
        # After each block the spatial dims change as annotated.
        # Starting from (B, 1, 32, 128):
        self.cnn = nn.Sequential(
            # Block 1: (B, 1,   32, 128) → (B, 64,  16, 64)
            _ConvBlock(1,   64,  pool=(2, 2)),
            # Block 2: (B, 64,  16, 64)  → (B, 128,  8, 32)
            _ConvBlock(64,  128, pool=(2, 2)),
            # Block 3: (B, 128,  8, 32)  → (B, 256,  8, 32)
            _ConvBlock(128, 256),
            # Block 4: (B, 256,  8, 32)  → (B, 256,  4, 32)
            _ConvBlock(256, 256, pool=(2, 1)),
            # Block 5: (B, 256,  4, 32)  → (B, 256,  2, 32)
            _ConvBlock(256, 256, pool=(2, 1)),
            # Collapse height to 1 via adaptive pool → (B, 256, 1, 32)
            nn.AdaptiveAvgPool2d((1, 32)),
        )

        # ── Sequence model ────────────────────────────────────────────────────
        # Reshape (B, 256, 1, 32) → (32, B, 256) then feed to BiLSTM
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=False,   # input is (T, B, input_size)
            bidirectional=True,
            dropout=0.25,
        )

        # ── Classifier ────────────────────────────────────────────────────────
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 1, H, W)  normalised grayscale image

        Returns:
            log_probs : (T, B, num_classes)  log-softmax activations
        """
        # CNN feature extraction
        features = self.cnn(x)             # (B, 256, 1, T)
        B, C, H, T = features.shape
        features = features.squeeze(2)    # (B, 256, T)
        features = features.permute(2, 0, 1)  # (T, B, 256)

        # RNN
        rnn_out, _ = self.rnn(features)   # (T, B, hidden*2)

        # Classification
        logits = self.fc(rnn_out)          # (T, B, num_classes)
        return F.log_softmax(logits, dim=2)
        

# ── Decoding ──────────────────────────────────────────────────────────────────

def crnn_decode(log_probs: torch.Tensor) -> tuple[str, float]:
    """
    Greedy CTC decode.

    Args:
        log_probs : (T, num_classes)  log-softmax output for a single sample

    Returns:
        text       : decoded plate string (uppercase alphanumeric)
        confidence : mean softmax probability of each predicted non-blank char
    """
    probs    = log_probs.exp()              # (T, C)
    best_idx = probs.argmax(dim=1)          # (T,)

    chars = []
    conf_vals = []
    prev = BLANK

    for t, idx in enumerate(best_idx.tolist()):
        if idx != prev and idx != BLANK:
            chars.append(IDX2CHAR.get(idx, ""))
            conf_vals.append(probs[t, idx].item())
        prev = idx

    text       = "".join(chars)
    confidence = float(sum(conf_vals) / len(conf_vals)) if conf_vals else 0.0
    return text, confidence
