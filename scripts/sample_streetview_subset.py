#!/usr/bin/env python3
"""Sample a reproducible 900/100 feasibility split for StreetView Pano."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["IMG_FILE", "LAT", "LON"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample feasibility train/val subset from StreetView Pano")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/streetview_pano/train_subset.csv"),
        help="Path to official StreetView Pano train csv",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("data/streetview_pano/val_subset.csv"),
        help="Path to official StreetView Pano val csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/streetview_pano/feasibility"),
        help="Output directory for sampled feasibility CSV files",
    )
    parser.add_argument("--total", type=int, default=1000, help="Total number of sampled rows")
    parser.add_argument("--train-size", type=int, default=900, help="Number of train rows in feasibility split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame, source: Path) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {source}")


def main() -> int:
    args = parse_args()
    if args.train_size >= args.total:
        raise ValueError("--train-size must be smaller than --total")

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    _validate_columns(train_df, args.train_csv)
    _validate_columns(val_df, args.val_csv)

    pool = pd.concat([train_df, val_df], ignore_index=True)
    pool = pool.drop_duplicates(subset=["IMG_FILE"]).reset_index(drop=True)

    if len(pool) < args.total:
        raise ValueError(f"Not enough unique rows to sample: need {args.total}, found {len(pool)}")

    sampled = pool.sample(n=args.total, random_state=args.seed).reset_index(drop=True)
    feasibility_train = sampled.iloc[: args.train_size].copy()
    feasibility_val = sampled.iloc[args.train_size :].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    out_train = args.output_dir / "train_subset.csv"
    out_val = args.output_dir / "val_subset.csv"

    feasibility_train.to_csv(out_train, index=False)
    feasibility_val.to_csv(out_val, index=False)

    print(f"Saved feasibility train subset: {out_train} ({len(feasibility_train)} rows)")
    print(f"Saved feasibility val subset: {out_val} ({len(feasibility_val)} rows)")
    print(f"Sampling seed: {args.seed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
