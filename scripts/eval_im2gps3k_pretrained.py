#!/usr/bin/env python3
"""Evaluate pretrained GeoCLIP on Im2GPS3K and generate comparison outputs.

This script runs evaluation using GeoCLIP pretrained weights that are bundled in
the repository and compares results with paper-reported metrics from
GeoCLIP Table 1(a) on Im2GPS3k.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from geoclip import GeoCLIP
from geoclip.train.dataloader import GeoDataLoader, img_val_transform
from geoclip.train.eval import eval_images


# Source: GeoCLIP paper (arXiv:2309.16020v2), Table 1(a), Im2GPS3k, Ours row.
PAPER_METRICS = {
    "acc_1_km": 0.1411,
    "acc_25_km": 0.3447,
    "acc_200_km": 0.5065,
    "acc_750_km": 0.6967,
    "acc_2500_km": 0.8382,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained GeoCLIP on Im2GPS3K")
    parser.add_argument("--test-csv", type=Path, default=Path("data/im2gps3k/test_subset.csv"))
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/im2gps3k/images/im2gps3ktest"),
        help="Directory containing Im2GPS3K test JPG files",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. 'auto' picks cuda when available",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/im2gps3k/geoclip_pretrained_eval.json"),
        help="Where to save evaluation and comparison JSON",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/im2gps3k_reproduction_table.md"),
        help="Where to save markdown comparison table",
    )
    return parser.parse_args()


def collate_image_gps(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    gps = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return images, gps


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def compute_comparison_rows(eval_metrics: dict[str, float]) -> list[dict[str, float]]:
    rows = []
    for threshold in [1, 25, 200, 750, 2500]:
        key = f"acc_{threshold}_km"
        paper = PAPER_METRICS[key]
        ours = float(eval_metrics[key])
        rows.append(
            {
                "threshold_km": threshold,
                "paper": paper,
                "ours": ours,
                "delta": ours - paper,
            }
        )
    return rows


def write_markdown_table(output_md: Path, rows: list[dict[str, float]]) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Im2GPS3K Reproduction Comparison",
        "",
        "Paper source: GeoCLIP arXiv:2309.16020v2, Table 1(a), Im2GPS3k (Ours).",
        "",
        "| Threshold | Paper (Acc) | Reproduced (Acc) | Delta (Reproduced - Paper) |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold_km']} km | {row['paper']:.4f} | {row['ours']:.4f} | {row['delta']:+.4f} |"
        )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    dataset = GeoDataLoader(str(args.test_csv), str(args.image_dir), transform=img_val_transform())
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check --test-csv and --image-dir")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_image_gps,
    )

    model = GeoCLIP(from_pretrained=True).to(device)
    model.gps_gallery = model.gps_gallery.to(device)

    eval_metrics = eval_images(dataloader, model, device=device)
    comparison_rows = compute_comparison_rows(eval_metrics)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "paper_source": "GeoCLIP arXiv:2309.16020v2 Table 1(a) Im2GPS3k (Ours)",
        "device": device,
        "dataset_size": len(dataset),
        "test_csv": str(args.test_csv),
        "image_dir": str(args.image_dir),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "metrics": eval_metrics,
        "paper_metrics": PAPER_METRICS,
        "comparison": comparison_rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_table(args.output_md, comparison_rows)

    print(f"Saved JSON: {args.output_json}")
    print(f"Saved table: {args.output_md}")
    print("\nComparison summary:")
    for row in comparison_rows:
        print(
            f"  Acc@{row['threshold_km']}km | paper={row['paper']:.4f} "
            f"reproduced={row['ours']:.4f} delta={row['delta']:+.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
