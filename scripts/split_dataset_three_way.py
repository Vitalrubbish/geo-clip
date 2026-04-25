#!/usr/bin/env python3
"""Split dataset CSV into train/val/test CSV files.

Expected CSV columns: IMG_FILE, LAT, LON
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--input-csv", type=Path, default=Path("data/all_subset.csv"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/train_subset.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/val_subset.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test_subset.csv"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")

    df = pd.read_csv(args.input_csv)
    required = {"IMG_FILE", "LAT", "LON"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSV: {sorted(missing)}")

    df = df.drop_duplicates(subset=["IMG_FILE"]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n = len(df)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)

    args.train_csv.parent.mkdir(parents=True, exist_ok=True)
    args.val_csv.parent.mkdir(parents=True, exist_ok=True)
    args.test_csv.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(args.train_csv, index=False)
    val_df.to_csv(args.val_csv, index=False)
    test_df.to_csv(args.test_csv, index=False)

    print("Split complete:")
    print(f"- input: {args.input_csv} ({n})")
    print(f"- train: {args.train_csv} ({len(train_df)})")
    print(f"- val: {args.val_csv} ({len(val_df)})")
    print(f"- test: {args.test_csv} ({len(test_df)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
