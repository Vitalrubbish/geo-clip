#!/usr/bin/env python3
"""Run a lightweight baseline training/validation/test pipeline for GeoCLIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from geoclip import GeoCLIP
from geoclip.train.dataloader import GeoDataLoader, img_train_transform, img_val_transform
from geoclip.train.eval import eval_images
from geoclip.train.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline train/val/test")
    parser.add_argument("--train-csv", type=Path, default=Path("data/train_subset.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/val_subset.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test_subset.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/train_images"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=Path, default=Path("data/baseline_results.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/baseline_model.pt"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_image_gps(batch: List[Tuple[torch.Tensor, Tuple[float, float]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([item[0] for item in batch], dim=0)
    gps = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return images, gps


def make_loader(csv_path: Path, image_dir: Path, batch_size: int, num_workers: int, train_mode: bool) -> DataLoader:
    transform = img_train_transform() if train_mode else img_val_transform()
    dataset = GeoDataLoader(str(csv_path), str(image_dir), transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_mode,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train_mode,
        collate_fn=collate_image_gps,
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader = make_loader(args.train_csv, args.image_dir, args.batch_size, args.num_workers, train_mode=True)
    val_loader = make_loader(args.val_csv, args.image_dir, args.batch_size, args.num_workers, train_mode=False)
    test_loader = make_loader(args.test_csv, args.image_dir, args.batch_size, args.num_workers, train_mode=False)

    model = GeoCLIP(from_pretrained=True).to(device)
    model.gps_gallery = model.gps_gallery.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)

    best_val = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train(
            train_dataloader=train_loader,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            batch_size=args.batch_size,
            device=device,
            scheduler=None,
        )

        val_metrics = eval_images(val_loader, model, device=device)
        history.append({"epoch": epoch, "val": val_metrics})

        current_val = float(val_metrics.get("acc_200_km", 0.0))
        if current_val > best_val:
            best_val = current_val
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, args.checkpoint)

    test_metrics = eval_images(test_loader, model, device=device)

    result = {
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "test_size": len(test_loader.dataset),
        "best_val_acc_200_km": best_val,
        "test_metrics": test_metrics,
        "history": history,
        "checkpoint": str(args.checkpoint),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Saved baseline results:")
    print(f"- results json: {args.output_json}")
    print(f"- checkpoint: {args.checkpoint}")
    print("- test metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
