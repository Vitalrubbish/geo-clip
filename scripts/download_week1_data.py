#!/usr/bin/env python3
"""Download geo-tagged images and compress them to 224x224 JPEG on the fly.

This script downloads parquet shards one-by-one from Hugging Face, converts each
image to RGB, resizes it to 224x224, and saves as compressed JPEG.

It then writes:
1) local image files under data/train_images/
2) all metadata CSV (IMG_FILE, LAT, LON)
3) train/val CSV split (no test split)
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_url, list_repo_files
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and compress geo dataset")
    parser.add_argument(
        "--dataset",
        default="blalexa/google-streetview-panoramas-geotagged",
        help="Hugging Face dataset id",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of images to download; -1 means all images",
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=-1,
        help="Maximum parquet shards to process; -1 means all shards",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/train_images"),
        help="Directory where images are saved",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/all_subset.csv"),
        help="Output CSV path with IMG_FILE/LAT/LON before split",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of samples used for train split (0.0 to 1.0)",
    )
    parser.add_argument(
        "--train-csv-path",
        type=Path,
        default=Path("data/train_subset.csv"),
        help="Output train CSV path with IMG_FILE/LAT/LON",
    )
    parser.add_argument(
        "--val-csv-path",
        type=Path,
        default=Path("data/val_subset.csv"),
        help="Output val CSV path with IMG_FILE/LAT/LON",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG compression quality (1-95)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="HTTP timeout in seconds per request",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries for downloading one parquet shard",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Progress/report batch size; show progress per batch of images",
    )
    return parser.parse_args()


def download_shard_to_temp(
    dataset: str,
    shard_filename: str,
    timeout: int,
    max_retries: int,
) -> Path:
    url = hf_hub_url(repo_id=dataset, filename=shard_filename, repo_type="dataset")
    last_exc = None

    for attempt in range(1, max_retries + 1):
        tmp_name = None
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                    tmp_name = tmp.name
                    with tqdm(
                        total=total_size,
                        unit="iB",
                        unit_scale=True,
                        desc=f"Download {shard_filename}",
                        leave=False,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                tmp.write(chunk)
                                pbar.update(len(chunk))
                    return Path(tmp_name)
        except Exception as exc:
            if tmp_name and os.path.exists(tmp_name):
                os.remove(tmp_name)
            last_exc = exc
            wait_s = min(30, 2 ** attempt)
            print(f"Retry {attempt}/{max_retries} for {shard_filename} after error: {exc}")
            time.sleep(wait_s)

    raise RuntimeError(f"Failed to download {shard_filename}: {last_exc}")


def compress_and_save_jpg(img_bytes: bytes, out_path: Path, jpeg_quality: int) -> None:
    with Image.open(io.BytesIO(img_bytes)) as image:
        image = image.convert("RGB")
        image = image.resize((224, 224), Image.Resampling.BICUBIC)
        image.save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    args.train_csv_path.parent.mkdir(parents=True, exist_ok=True)
    args.val_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not (0.0 < args.train_ratio < 1.0):
        print("--train-ratio must be in (0.0, 1.0)")
        return 1
    if not (1 <= args.jpeg_quality <= 95):
        print("--jpeg-quality must be in [1, 95]")
        return 1
    if args.batch_size <= 0:
        print("--batch-size must be > 0")
        return 1

    print(f"Listing dataset files: {args.dataset}")
    files = list_repo_files(args.dataset, repo_type="dataset")
    parquets = sorted([f for f in files if f.endswith(".parquet")])
    if not parquets:
        print("No parquet shards found in the dataset repo.")
        return 1

    if args.max_shards > 0:
        parquets = parquets[: args.max_shards]

    rows = []
    failures = 0
    processed = 0

    total_target = None if args.max_samples < 0 else args.max_samples
    overall_pbar = tqdm(total=total_target, desc="Total images", position=0)

    batch_index = 1
    batch_done = 0

    def next_batch_target() -> int:
        if args.max_samples < 0:
            return args.batch_size
        remaining = args.max_samples - processed
        return max(0, min(args.batch_size, remaining))

    current_batch_target = next_batch_target()
    batch_pbar = tqdm(
        total=current_batch_target,
        desc=f"Batch {batch_index}",
        position=1,
        leave=False,
    )

    for shard in parquets:
        if args.max_samples >= 0 and processed >= args.max_samples:
            break

        temp_parquet = None
        try:
            temp_parquet = download_shard_to_temp(
                dataset=args.dataset,
                shard_filename=shard,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
            )
            shard_df = pd.read_parquet(temp_parquet, columns=["pano_id", "lat", "lon", "image"])
        except Exception as exc:
            failures += 1
            print(f"Skipping shard {shard} due to error: {exc}")
            continue
        finally:
            if temp_parquet is not None and temp_parquet.exists():
                try:
                    temp_parquet.unlink()
                except Exception:
                    pass

        for idx, sample in shard_df.iterrows():
            if args.max_samples >= 0 and processed >= args.max_samples:
                break

            lat = sample.get("lat")
            lon = sample.get("lon")
            image = sample.get("image")
            pano_id = sample.get("pano_id", f"sample_{idx}")

            if lat is None or lon is None or image is None:
                failures += 1
                continue

            img_bytes = image.get("bytes") if isinstance(image, dict) else None
            if not img_bytes:
                failures += 1
                continue

            file_name = f"{pano_id}.jpg"
            file_path = args.out_dir / file_name

            try:
                if not file_path.exists():
                    compress_and_save_jpg(img_bytes, file_path, args.jpeg_quality)
                rows.append({"IMG_FILE": file_name, "LAT": float(lat), "LON": float(lon)})
                overall_pbar.update(1)
                batch_pbar.update(1)
                processed += 1
                batch_done += 1

                if batch_done >= current_batch_target and current_batch_target > 0:
                    batch_pbar.close()
                    print(
                        f"Completed batch {batch_index}: +{batch_done} images "
                        f"(processed={processed}, failures={failures})"
                    )
                    batch_index += 1
                    batch_done = 0
                    current_batch_target = next_batch_target()
                    if current_batch_target > 0:
                        batch_pbar = tqdm(
                            total=current_batch_target,
                            desc=f"Batch {batch_index}",
                            position=1,
                            leave=False,
                        )
            except Exception:
                failures += 1
                continue

    if not batch_pbar.disable:
        batch_pbar.close()
    overall_pbar.close()

    if not rows:
        print("No valid samples downloaded. Exiting.")
        return 1

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["IMG_FILE"]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    df.to_csv(args.csv_path, index=False)

    train_count = int(len(df) * args.train_ratio)
    train_df = df.iloc[:train_count].reset_index(drop=True)
    val_df = df.iloc[train_count:].reset_index(drop=True)

    train_df.to_csv(args.train_csv_path, index=False)
    val_df.to_csv(args.val_csv_path, index=False)

    print("Saved files:")
    print(f"- All CSV: {args.csv_path} ({len(df)} rows)")
    print(f"- Train CSV: {args.train_csv_path} ({len(train_df)} rows)")
    print(f"- Val CSV: {args.val_csv_path} ({len(val_df)} rows)")
    print(f"- Images dir: {args.out_dir}")
    print(f"- Skipped/failed records: {failures}")

    return 0


if __name__ == "__main__":
    code = main()
    # Work around intermittent interpreter-finalization crash from PIL/datasets threads.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
