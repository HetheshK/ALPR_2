# confusion.py
"""
Read eval_results.csv and print a character-level confusion matrix.
Shows which characters TrOCR (or any engine) commonly mistakes for others.

Usage:
    python evaluate.py --trocr-only --out eval_results.csv
    python confusion.py                          # reads eval_results.csv
    python confusion.py --input my_results.csv   # custom input
    python confusion.py --top 30                 # show top 30 confusion pairs
    python confusion.py --matrix                 # also print full grid matrix
"""

import csv
import argparse
from collections import defaultdict


def normalize(text: str) -> str:
    return "".join(c for c in text.upper() if c.isalnum())


def align_substitutions(pred: str, gt: str) -> list[tuple[str, str]]:
    """
    Edit-distance traceback to find character substitutions.
    Returns list of (gt_char, pred_char) pairs where they differ.
    Insertions and deletions are not counted — only substitutions.
    """
    m, n = len(pred), len(gt)
    # Build full DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    # Traceback from bottom-right
    subs = []
    i, j = m, n
    while i > 0 and j > 0:
        if pred[i - 1] == gt[j - 1]:
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            # Substitution
            subs.append((gt[j - 1], pred[i - 1]))  # (actual, predicted)
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            # Deletion from pred (extra char in gt)
            i -= 1
        else:
            # Insertion into pred (extra char in pred)
            j -= 1

    return subs


def load_results(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def print_top_pairs(confusion: dict, top_n: int):
    pairs = sorted(
        [(cnt, a, p) for (a, p), cnt in confusion.items() if a != p],
        reverse=True,
    )
    print(f"\n{'Rank':>4}  {'Actual':>6}  {'Predicted':>9}  {'Count':>5}")
    print("  " + "─" * 32)
    for rank, (cnt, actual, pred) in enumerate(pairs[:top_n], 1):
        print(f"  {rank:>3}.  {actual:>6}  →  {pred:>6}    {cnt:>4}")


def print_matrix(confusion: dict):
    # Only include characters that appear in actual errors
    chars = sorted(
        {c for (a, p) in confusion if a != p for c in (a, p)}
    )
    if not chars:
        print("No confusion data to display.")
        return

    col_w = 4
    header = "     " + "".join(f"{c:>{col_w}}" for c in chars)
    print(f"\nConfusion matrix  (row = ACTUAL, col = PREDICTED)\n")
    print(header)
    print("     " + "─" * (col_w * len(chars)))

    for actual in chars:
        row_vals = []
        for pred in chars:
            cnt = confusion.get((actual, pred), 0)
            if cnt == 0:
                row_vals.append(".")
            else:
                row_vals.append(str(cnt))
        row_str = "".join(f"{v:>{col_w}}" for v in row_vals)
        print(f"  {actual} │{row_str}")


def main():
    parser = argparse.ArgumentParser(description="Character confusion matrix from eval results")
    parser.add_argument("--input",  default="eval_results.csv",
                        help="eval_results.csv produced by evaluate.py  (default: eval_results.csv)")
    parser.add_argument("--top",    type=int, default=20,
                        help="Number of top confusion pairs to list  (default: 20)")
    parser.add_argument("--matrix", action="store_true",
                        help="Also print the full grid confusion matrix")
    args = parser.parse_args()

    rows = load_results(args.input)
    print(f"Loaded {len(rows)} result rows from '{args.input}'")

    confusion: dict[tuple[str, str], int] = defaultdict(int)
    wrong_images = 0
    total_subs   = 0

    for row in rows:
        gt_raw   = row.get("ground_truth", "")
        pred_raw = row.get("predicted", "")
        gt_norm  = normalize(gt_raw)
        pred_norm = normalize(pred_raw)

        if gt_norm == pred_norm:
            continue  # exact match — skip

        wrong_images += 1
        subs = align_substitutions(pred_norm, gt_norm)
        for actual, predicted in subs:
            confusion[(actual, predicted)] += 1
            total_subs += 1

    print(f"Wrong predictions : {wrong_images}/{len(rows)}")
    print(f"Character errors  : {total_subs} substitutions extracted")

    if total_subs == 0:
        print("No substitution errors found — perfect score or empty results.")
        return

    print_top_pairs(confusion, args.top)

    if args.matrix:
        print_matrix(confusion)


if __name__ == "__main__":
    main()
