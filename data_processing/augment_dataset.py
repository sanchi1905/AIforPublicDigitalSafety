"""
augment_dataset.py

Expands a small folder of real banknote photos into a much larger
training set using randomized augmentation - the same strategy used
in published Indian-currency detection projects when only a handful
of unique note photos are available.

Expected input structure:
    raw_images/
        real/   <- your photographed genuine notes
        fake/   <- your photographed fake/test notes

Output structure:
    augmented_dataset/
        real/   <- expanded to N images
        fake/   <- expanded to N images

Usage:
    python3 augment_dataset.py --input raw_images --output augmented_dataset --target 1000
"""

import os
import argparse
from PIL import Image, ImageEnhance
import numpy as np
import random

random.seed(42)
np.random.seed(42)


def random_augment(img):
    """Apply a random combination of realistic augmentations."""
    # Random rotation (small angles - notes are usually photographed
    # roughly upright, don't rotate wildly)
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    # Random brightness (simulates different lighting conditions)
    brightness = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    # Random contrast
    contrast = random.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # Random slight blur or sharpness (simulates camera focus variation)
    if random.random() > 0.5:
        sharpness = random.uniform(0.5, 1.5)
        img = ImageEnhance.Sharpness(img).enhance(sharpness)

    # Random crop + resize back (simulates different zoom/framing)
    w, h = img.size
    crop_pct = random.uniform(0.85, 1.0)
    new_w, new_h = int(w * crop_pct), int(h * crop_pct)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    img = img.crop((left, top, left + new_w, top + new_h))
    img = img.resize((w, h))

    return img


def expand_folder(src_folder, dest_folder, target_count):
    os.makedirs(dest_folder, exist_ok=True)
    source_files = [f for f in os.listdir(src_folder)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not source_files:
        print(f"  No images found in {src_folder}, skipping.")
        return

    print(f"  Found {len(source_files)} source images in {src_folder}")

    count = 0
    # First copy originals as-is
    for f in source_files:
        img = Image.open(os.path.join(src_folder, f)).convert('RGB')
        img.save(os.path.join(dest_folder, f"orig_{count}.jpg"))
        count += 1

    # Then generate augmented versions until we hit target_count
    while count < target_count:
        src_file = random.choice(source_files)
        img = Image.open(os.path.join(src_folder, src_file)).convert('RGB')
        aug_img = random_augment(img)
        aug_img.save(os.path.join(dest_folder, f"aug_{count}.jpg"), quality=90)
        count += 1

    print(f"  -> Expanded to {count} images in {dest_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder with real/ and fake/ subfolders")
    parser.add_argument("--output", required=True, help="Output folder for expanded dataset")
    parser.add_argument("--target", type=int, default=1000, help="Target images per class")
    args = parser.parse_args()

    for label in ["real", "fake"]:
        src = os.path.join(args.input, label)
        dest = os.path.join(args.output, label)
        if os.path.isdir(src):
            print(f"Processing '{label}' class...")
            expand_folder(src, dest, args.target)
        else:
            print(f"Warning: {src} does not exist, skipping '{label}' class.")

    print("\nDone. Next step: run the train/val/test split script on the expanded dataset.")
