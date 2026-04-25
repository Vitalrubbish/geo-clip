#!/usr/bin/env python3
"""Download Im2GPS3k test images and generate a GeoCLIP-compatible CSV.

Outputs:
1) data/im2gps3k/images/im2gps3ktest/*.jpg
2) data/im2gps3k/im2gps3k_places365.csv (raw metadata from HF mirror)
3) data/im2gps3k/test_subset.csv with columns: IMG_FILE,LAT,LON
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Im2GPS3k and build test CSV")
    parser.add_argument(
        "--mediafire-page-url",
        type=str,
        default="https://www.mediafire.com/file/7ht7sn78q27o9we/im2gps3ktest.zip/file",
        help="MediaFire page URL for im2gps3ktest.zip",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("data/im2gps3k"),
        help="Root directory for Im2GPS3k assets",
    )
    parser.add_argument(
        "--zip-name",
        type=str,
        default="im2gps3ktest.zip",
        help="Local zip filename under root-dir",
    )
    parser.add_argument(
        "--image-subdir",
        type=Path,
        default=Path("images"),
        help="Subdirectory under root-dir to extract images",
    )
    parser.add_argument(
        "--metadata-repo-id",
        type=str,
        default="Jia-py/G3-checkpoint",
        help="Hugging Face repo id hosting im2gps3k metadata",
    )
    parser.add_argument(
        "--metadata-filename",
        type=str,
        default="im2gps3k_places365.csv",
        help="Metadata CSV filename inside Hugging Face repo",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/im2gps3k/test_subset.csv"),
        help="Output CSV with IMG_FILE,LAT,LON",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--force-redownload-zip",
        action="store_true",
        help="Force re-downloading the image zip even if it exists",
    )
    parser.add_argument(
        "--force-reextract",
        action="store_true",
        help="Force re-extracting zip",
    )
    return parser.parse_args()


def resolve_mediafire_download_url(page_url: str, timeout: int) -> str:
    response = requests.get(
        page_url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    html = response.text

    patterns = [
        r'href="(https://download[^\"]+im2gps3ktest\.zip[^\"]*)"',
        r'"downloadUrl"\s*:\s*"(https:[^\"]+im2gps3ktest\.zip[^\"]*)"',
        r"https://download[^\"']+im2gps3ktest\\.zip[^\"']*",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, html)
        if matches:
            return matches[0]

    raise RuntimeError("Could not resolve direct MediaFire download URL for im2gps3ktest.zip")


def stream_download(url: str, out_file: Path, timeout: int) -> None:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with out_file.open("wb") as f, tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc=f"Downloading {out_file.name}",
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def maybe_extract_zip(zip_path: Path, image_dir: Path, force_reextract: bool) -> None:
    extracted_root = image_dir / "im2gps3ktest"
    if extracted_root.exists() and not force_reextract:
        print(f"Skip extraction (already exists): {extracted_root}")
        return

    image_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(image_dir)
    print(f"Extracted zip to: {image_dir}")


def build_test_csv(metadata_csv: Path, extracted_image_root: Path, output_csv: Path) -> None:
    df = pd.read_csv(metadata_csv)
    required = {"IMG_ID", "LAT", "LON"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in metadata CSV: {sorted(missing)}")

    image_files = sorted(p.name for p in extracted_image_root.glob("*.jpg"))
    image_set = set(image_files)

    df = df[["IMG_ID", "LAT", "LON"]].copy()
    df = df[df["IMG_ID"].isin(image_set)].copy()
    df = df.rename(columns={"IMG_ID": "IMG_FILE"})
    df = df.drop_duplicates(subset=["IMG_FILE"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    meta_set = set(pd.read_csv(metadata_csv)["IMG_ID"].astype(str).tolist())
    missing_meta = sorted(image_set - meta_set)
    missing_images = sorted(meta_set - image_set)

    print("Saved test CSV:")
    print(f"- {output_csv} ({len(df)} rows)")
    print(f"- extracted images: {len(image_set)}")
    print(f"- metadata rows: {len(meta_set)}")
    print(f"- images without metadata: {len(missing_meta)}")
    print(f"- metadata without local image: {len(missing_images)}")


def main() -> int:
    args = parse_args()

    args.root_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.root_dir / args.image_subdir
    zip_path = args.root_dir / args.zip_name

    if args.force_redownload_zip or not zip_path.exists():
        direct_url = resolve_mediafire_download_url(args.mediafire_page_url, args.request_timeout)
        print(f"Resolved MediaFire URL: {direct_url[:120]}...")
        stream_download(direct_url, zip_path, args.request_timeout)
    else:
        print(f"Skip zip download (already exists): {zip_path}")

    maybe_extract_zip(zip_path, image_dir, args.force_reextract)

    metadata_local = hf_hub_download(
        repo_id=args.metadata_repo_id,
        filename=args.metadata_filename,
        repo_type="model",
        local_dir=str(args.root_dir),
    )
    metadata_csv = Path(metadata_local)
    print(f"Metadata CSV: {metadata_csv}")

    extracted_image_root = image_dir / "im2gps3ktest"
    if not extracted_image_root.exists():
        raise FileNotFoundError(f"Extracted image folder not found: {extracted_image_root}")

    build_test_csv(metadata_csv, extracted_image_root, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
