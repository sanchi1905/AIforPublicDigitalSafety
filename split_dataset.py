"""
split_dataset.py
================
Splits currency_images_augmented/real and .../fake into a reproducible
train / val / test structure (70 / 15 / 15) under currency_dataset_split/.

Leakage-safe design
-------------------
Images are split at the *source* level before copying.  If augmented
variants derived from the same source photo are present (files whose
names start with aug_<N>_ where N is the source index, or orig_<N>),
they are grouped by source index so that all variants of one source
photo land in exactly one split.

Current state: augment_dataset.py was run with --target < source count,
so every file is orig_<N>.jpg (one unique source each).  The grouping
logic handles both cases gracefully.

Output layout
-------------
currency_dataset_split/
    train/
        real/   (70 %)
        fake/   (70 %)
    val/
        real/   (15 %)
        fake/   (15 %)
    test/
        real/   (15 %)
        fake/   (15 %)
"""

import os
import re
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ── CONFIG ─────────────────────────────────────────────────────────────────
SRC_BASE   = r"c:\Users\Asus\Downloads\ETA Hackathon\currency_images_augmented"
DEST_BASE  = r"c:\Users\Asus\Downloads\ETA Hackathon\currency_dataset_split"
CLASSES    = ["real", "fake"]
SEED       = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── HELPERS ────────────────────────────────────────────────────────────────
def source_index(filename: str) -> str:
    """
    Extract the source-image index from a filename so that all augmented
    variants of the same source photo are grouped together.

    Naming conventions handled:
      orig_42.jpg          → source key "42"
      aug_42_sometext.jpg  → source key "42"
      aug_1042_foo.jpg     → source key "1042"
      anything_else.jpg    → source key = full filename (treated as unique)
    """
    name = Path(filename).stem  # drop extension
    # orig_<N>
    m = re.match(r'^orig_(\d+)$', name)
    if m:
        return m.group(1)
    # aug_<N>_...
    m = re.match(r'^aug_(\d+)_', name)
    if m:
        return m.group(1)
    # aug_<N> (no trailing _)
    m = re.match(r'^aug_(\d+)$', name)
    if m:
        return m.group(1)
    return name  # unique fallback


def split_indices(keys: list, seed: int, train_f: float, val_f: float):
    """Shuffle keys and return (train_keys, val_keys, test_keys)."""
    rng = random.Random(seed)
    shuffled = keys[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * train_f)
    n_val   = round(n * val_f)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :]
    )


def copy_files(file_list, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src_path in file_list:
        shutil.copy2(src_path, dest_dir / src_path.name)


# ── MAIN ───────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*62}")
    print("  Dataset Train / Val / Test Split")
    print(f"{'='*62}")
    print(f"  Source  : {SRC_BASE}")
    print(f"  Output  : {DEST_BASE}")
    print(f"  Split   : {int(TRAIN_FRAC*100)} / {int(VAL_FRAC*100)} / {int((1-TRAIN_FRAC-VAL_FRAC)*100)}")
    print(f"  Seed    : {SEED}")
    print(f"{'='*62}\n")

    summary = {}   # class -> {split -> count}

    for cls in CLASSES:
        src_dir = Path(SRC_BASE) / cls
        if not src_dir.is_dir():
            print(f"  ⚠  {src_dir} not found — skipping '{cls}'.")
            continue

        # ── Collect all images and group by source index ──────────────────
        all_files = sorted(
            p for p in src_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

        groups: dict[str, list[Path]] = defaultdict(list)
        for f in all_files:
            groups[source_index(f.name)].append(f)

        unique_sources = sorted(groups.keys())
        print(f"  [{cls.upper()}]")
        print(f"    Total images      : {len(all_files):>5,}")
        print(f"    Unique sources    : {len(unique_sources):>5,}")

        # ── Split at source level ─────────────────────────────────────────
        train_keys, val_keys, test_keys = split_indices(
            unique_sources, SEED, TRAIN_FRAC, VAL_FRAC
        )

        splits = {
            "train": train_keys,
            "val":   val_keys,
            "test":  test_keys,
        }

        cls_summary = {}
        for split_name, keys in splits.items():
            files_in_split = [f for k in keys for f in groups[k]]
            dest = Path(DEST_BASE) / split_name / cls
            copy_files(files_in_split, dest)
            cls_summary[split_name] = len(files_in_split)
            print(f"    {split_name:5s}: {len(keys):>4,} sources -> {len(files_in_split):>5,} images  ->  {dest}")

        summary[cls] = cls_summary
        print()

    # ── Final count table ─────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  FINAL COUNTS PER SPLIT PER CLASS")
    print(f"{'='*62}")
    print(f"  {'Split':<8} {'Real':>8} {'Fake':>8} {'Total':>8}")
    print(f"  {'-'*36}")
    grand_total = 0
    for split in ["train", "val", "test"]:
        real_n = summary.get("real", {}).get(split, 0)
        fake_n = summary.get("fake", {}).get(split, 0)
        tot    = real_n + fake_n
        grand_total += tot
        print(f"  {split:<8} {real_n:>8,} {fake_n:>8,} {tot:>8,}")
    print(f"  {'-'*36}")
    total_real = sum(summary.get("real", {}).values())
    total_fake = sum(summary.get("fake", {}).values())
    print(f"  {'TOTAL':<8} {total_real:>8,} {total_fake:>8,} {grand_total:>8,}")

    # ── Leakage verification ──────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  LEAKAGE VERIFICATION")
    print(f"{'='*62}")
    for cls in CLASSES:
        src_dir = Path(SRC_BASE) / cls
        if not src_dir.is_dir():
            continue
        all_files = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        groups: dict[str, list] = defaultdict(list)
        for f in all_files:
            groups[source_index(f.name)].append(f)
        unique_sources = sorted(groups.keys())
        train_keys, val_keys, test_keys = split_indices(unique_sources, SEED, TRAIN_FRAC, VAL_FRAC)

        train_set = set(train_keys)
        val_set   = set(val_keys)
        test_set  = set(test_keys)

        tv_overlap  = train_set & val_set
        tt_overlap  = train_set & test_set
        vt_overlap  = val_set   & test_set

        if tv_overlap or tt_overlap or vt_overlap:
            print(f"  [FAIL] [{cls}] LEAKAGE DETECTED!")
            if tv_overlap:  print(f"     Train & Val  : {len(tv_overlap)} sources")
            if tt_overlap:  print(f"     Train & Test : {len(tt_overlap)} sources")
            if vt_overlap:  print(f"     Val   & Test : {len(vt_overlap)} sources")
        else:
            print(f"  [OK]   [{cls}] No leakage -- train / val / test are fully disjoint.")

    print(f"\n  Split saved to: {DEST_BASE}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
