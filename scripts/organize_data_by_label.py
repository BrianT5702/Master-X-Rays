"""Organize NIH ChestX-ray14 images into fibrosis / nodules / negative folders."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _find_image_path(images_root: Path, image_index: str) -> Path:
    for i in range(1, 13):
        candidate = images_root / f"images_{i:03d}" / "images" / image_index
        if candidate.exists():
            return candidate
    for pattern in (
        f"images_*/images/{image_index}",
        f"images/{image_index}",
        image_index,
    ):
        matches = list(images_root.glob(pattern))
        if matches:
            return matches[0]
    direct = images_root / image_index
    if direct.exists():
        return direct
    raise FileNotFoundError(f"Image not found: {image_index}")


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        import shutil

        shutil.copy2(src, dst)


def organize(
    csv_path: Path,
    images_root: Path,
    output_root: Path,
    mode: str = "hardlink",
) -> None:
    metadata = pd.read_csv(csv_path)
    if "Finding Labels" not in metadata.columns:
        raise ValueError("CSV must contain 'Finding Labels' column")

    labels = metadata["Finding Labels"].astype(str)
    has_nodule = labels.str.contains("Nodule", regex=False, na=False)
    has_fibrosis = labels.str.contains("Fibrosis", regex=False, na=False)

    folders = {
        "nodules": output_root / "nodules",
        "fibrosis": output_root / "fibrosis",
        "negative": output_root / "negative",
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    stats = {"nodules": 0, "fibrosis": 0, "negative": 0, "missing": 0, "skipped_existing": 0}

    for idx, row in metadata.iterrows():
        image_index = row["Image Index"]
        try:
            src = _find_image_path(images_root, image_index)
        except FileNotFoundError:
            stats["missing"] += 1
            continue

        targets: list[str] = []
        if has_nodule.iloc[idx]:
            targets.append("nodules")
        if has_fibrosis.iloc[idx]:
            targets.append("fibrosis")
        if not targets:
            targets.append("negative")

        for name in targets:
            dst = folders[name] / image_index
            if dst.exists():
                stats["skipped_existing"] += 1
            else:
                _link_or_copy(src, dst, mode)
            stats[name] += 1

        if (idx + 1) % 10000 == 0:
            print(f"  processed {idx + 1:,} / {len(metadata):,} rows...")

    print("\nDone.")
    print(f"  Nodule images linked:    {stats['nodules']:,}")
    print(f"  Fibrosis images linked:  {stats['fibrosis']:,}")
    print(f"  Negative images linked:  {stats['negative']:,}")
    print(f"  Missing source files:    {stats['missing']:,}")
    print(f"  Output root: {output_root.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split NIH images into data/nodules, data/fibrosis, data/negative."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("datasets/NIH Chest X-Rays Master Datasets/archive/Data_Entry_2017.csv"),
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("datasets/NIH Chest X-Rays Master Datasets/archive"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Parent folder for nodules/, fibrosis/, negative/",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="hardlink (default, no extra disk); copy duplicates files",
    )
    args = parser.parse_args()

    print(f"[INFO] CSV: {args.csv}")
    print(f"[INFO] Images: {args.images_root}")
    print(f"[INFO] Mode: {args.mode}")
    organize(args.csv, args.images_root, args.output_root, args.mode)


if __name__ == "__main__":
    main()
