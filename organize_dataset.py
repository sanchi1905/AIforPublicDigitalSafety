"""
Currency Fraud Detection - Dataset Organizer
=============================================
Extracts 500 and 2000 genuine/fake note images from the zip archive,
organizes them into currency_images/real/ and currency_images/fake/,
then reports quality issues (corrupted, duplicates, too-small files).
"""

import zipfile
import os
import hashlib
from pathlib import Path
from collections import defaultdict

# CONFIG
ZIP_PATH       = r"c:\Users\Asus\Downloads\ETA Hackathon\dataset currency fake real.zip"
OUTPUT_BASE    = r"c:\Users\Asus\Downloads\ETA Hackathon\currency_images"
DENOMINATIONS  = {"500", "2000"}
MIN_FILE_SIZE  = 5_000   # bytes -- below this is flagged as too small
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

REAL_PREFIX = "data/data/real/"
FAKE_PREFIX = "data/data/fake/"


def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTS


def classify(entry_name: str):
    """Return (label, denomination) or (None, None) if not relevant."""
    en = entry_name.replace("\\", "/")
    if en.startswith(REAL_PREFIX):
        label = "real"
    elif en.startswith(FAKE_PREFIX):
        label = "fake"
    else:
        return None, None

    parts = en.split("/")
    if len(parts) >= 4:
        denom = parts[3]
        if denom in DENOMINATIONS:
            return label, denom
    return label, None


def main():
    real_dir = Path(OUTPUT_BASE) / "real"
    fake_dir = Path(OUTPUT_BASE) / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    counters      = defaultdict(int)
    by_denom      = defaultdict(int)
    hashes        = defaultdict(list)
    corrupted     = []
    too_small     = []
    skipped_ext   = 0
    skipped_denom = 0

    print(f"\n{'='*60}")
    print("  Currency Dataset Organizer")
    print(f"{'='*60}")
    print(f"  Source ZIP  : {ZIP_PATH}")
    print(f"  Output dir  : {OUTPUT_BASE}")
    print(f"  Denominations: {DENOMINATIONS}")
    print(f"{'='*60}\n")

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        entries = [e for e in zf.infolist() if not e.is_dir()]
        print(f"  Total entries in ZIP: {len(entries):,}")

        for i, entry in enumerate(entries, 1):
            if i % 1000 == 0:
                print(f"  Processing {i:,}/{len(entries):,} ...", flush=True)

            name = entry.filename.replace("\\", "/")

            if not is_image(name):
                skipped_ext += 1
                continue

            label, denom = classify(name)

            if label is None:
                continue

            if denom is None:
                skipped_denom += 1
                continue

            try:
                data = zf.read(entry.filename)
            except Exception as e:
                corrupted.append((name, str(e)))
                continue

            if len(data) < MIN_FILE_SIZE:
                too_small.append((name, len(data)))

            # Validate magic bytes
            valid = (
                data[:2] == b'\xff\xd8'    # JPEG
                or (len(data) > 8 and data[:8] == b'\x89PNG\r\n\x1a\n')  # PNG
                or data[:2] == b'BM'       # BMP
                or len(data) > 1000        # fallback: assume ok if reasonably sized
            )
            if not valid:
                corrupted.append((name, "unrecognized image format"))
                continue

            h = file_hash(data)
            hashes[h].append(name)

            # Skip exact duplicate (already written)
            dest_dir  = real_dir if label == "real" else fake_dir
            out_name  = f"{denom}_{Path(name).name}"
            dest_path = dest_dir / out_name

            # Resolve filename collision (different content, same name)
            counter = 1
            while dest_path.exists():
                existing_hash = file_hash(dest_path.read_bytes())
                if existing_hash == h:
                    break  # exact duplicate, skip
                stem   = f"{denom}_{Path(name).stem}"
                suffix = Path(name).suffix
                dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            else:
                dest_path.write_bytes(data)
                counters[label] += 1
                by_denom[f"{label}_{denom}"] += 1
                continue

            # If we hit break (exact dup), don't count it
            pass

    # ── Summary ───────────────────────────────────────────────────────────────
    duplicates = {h: v for h, v in hashes.items() if len(v) > 1}

    print(f"\n{'='*60}")
    print("  FINAL IMAGE COUNT PER CLASS")
    print(f"{'='*60}")
    print(f"  Real (genuine)  : {counters['real']:>5,}")
    print(f"    |- Rs.500     : {by_denom.get('real_500',  0):>5,}")
    print(f"    +- Rs.2000    : {by_denom.get('real_2000', 0):>5,}")
    print(f"  Fake            : {counters['fake']:>5,}")
    print(f"    |- Rs.500     : {by_denom.get('fake_500',  0):>5,}")
    print(f"    +- Rs.2000    : {by_denom.get('fake_2000', 0):>5,}")
    total = counters["real"] + counters["fake"]
    print(f"  ----------------------------")
    print(f"  TOTAL           : {total:>5,}")
    print(f"\n  Skipped: {skipped_ext:,} non-image + {skipped_denom:,} other-denomination files")

    print(f"\n{'='*60}")
    print("  QUALITY FLAGS")
    print(f"{'='*60}")

    if corrupted:
        print(f"\n  [!] CORRUPTED ({len(corrupted)}):")
        for fname, reason in corrupted[:20]:
            print(f"      {Path(fname).name}  ->  {reason}")
        if len(corrupted) > 20:
            print(f"      ... and {len(corrupted)-20} more")
    else:
        print("  [OK] No corrupted files.")

    if too_small:
        print(f"\n  [!] UNUSUALLY SMALL < {MIN_FILE_SIZE:,} bytes ({len(too_small)}):")
        for fname, size in sorted(too_small, key=lambda x: x[1])[:20]:
            print(f"      {Path(fname).name:45s}  {size:,} B")
        if len(too_small) > 20:
            print(f"      ... and {len(too_small)-20} more")
    else:
        print("  [OK] No unusually small files.")

    if duplicates:
        print(f"\n  [!] EXACT DUPLICATES ({len(duplicates)} groups):")
        for h, paths in list(duplicates.items())[:10]:
            print(f"      MD5:{h[:8]}  ({len(paths)} copies)")
            for p in paths[:3]:
                print(f"        - {p}")
            if len(paths) > 3:
                print(f"        ... +{len(paths)-3} more")
        if len(duplicates) > 10:
            print(f"      ... and {len(duplicates)-10} more groups")
    else:
        print("  [OK] No exact duplicates.")

    print(f"\n{'='*60}")
    print(f"  Saved to: {OUTPUT_BASE}/real  and  .../fake")
    print(f"{'='*60}\n")

    # Write text report
    report_path = Path(OUTPUT_BASE) / "dataset_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CURRENCY DATASET ORGANIZATION REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Real total          : {counters['real']}\n")
        f.write(f"  Rs.500            : {by_denom.get('real_500', 0)}\n")
        f.write(f"  Rs.2000           : {by_denom.get('real_2000', 0)}\n\n")
        f.write(f"Fake total          : {counters['fake']}\n")
        f.write(f"  Rs.500            : {by_denom.get('fake_500', 0)}\n")
        f.write(f"  Rs.2000           : {by_denom.get('fake_2000', 0)}\n\n")
        f.write(f"TOTAL               : {total}\n\n")
        f.write(f"Corrupted files     : {len(corrupted)}\n")
        f.write(f"Too-small files     : {len(too_small)}\n")
        f.write(f"Duplicate groups    : {len(duplicates)}\n\n")
        if corrupted:
            f.write("--- CORRUPTED ---\n")
            for fname, reason in corrupted:
                f.write(f"  {fname}: {reason}\n")
        if too_small:
            f.write("\n--- TOO SMALL ---\n")
            for fname, size in sorted(too_small, key=lambda x: x[1]):
                f.write(f"  {fname}: {size} bytes\n")
        if duplicates:
            f.write("\n--- DUPLICATES ---\n")
            for h, paths in duplicates.items():
                f.write(f"  MD5 {h}:\n")
                for p in paths:
                    f.write(f"    {p}\n")

    print(f"  Report saved -> {report_path}")


if __name__ == "__main__":
    main()
