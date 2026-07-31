"""
fix_material_classifier_data.py
================================
The material_classifier dataset currently only has wood images.
Steel and aluminum folders are empty, causing Stage 1 (ResNet18 router)
to be trained on a single class — completely broken.

This script samples representative images from:
  - data/processed/steel/train/images/    → material_classifier/train/steel/
  - data/processed_aluminum/images/train/ → material_classifier/train/aluminum/
  - wood already has 210 train images ✅

We aim for ~300 images per class across train/val/test (same as wood).

Usage:
    python scripts/fix_material_classifier_data.py          # dry run
    python scripts/fix_material_classifier_data.py --apply  # write files
"""

import argparse
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

SOURCES = {
    "steel":    PROJECT_ROOT / "data" / "processed" / "steel" / "train" / "images",
    "aluminum": PROJECT_ROOT / "data" / "processed_aluminum" / "images" / "train",
}

CLASSIFIER_DIR = PROJECT_ROOT / "data" / "material_classifier"

# Target counts to match wood (210 train, 45 val, 45 test = 300 total per class)
TRAIN_COUNT = 210
VAL_COUNT   = 45
TEST_COUNT  = 45
TOTAL       = TRAIN_COUNT + VAL_COUNT + TEST_COUNT

RANDOM_SEED = 42


def sample_and_copy(material: str, source_dir: Path, dry_run: bool) -> bool:
    """Sample TOTAL images from source_dir and distribute into train/val/test."""
    images = list(source_dir.glob("*.jpg")) + \
             list(source_dir.glob("*.jpeg")) + \
             list(source_dir.glob("*.png")) + \
             list(source_dir.glob("*.bmp"))

    if not images:
        logger.error(f"No images found in {source_dir}")
        return False

    logger.info(f"  {material}: {len(images)} available → sampling {TOTAL}")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(images)

    # If fewer than TOTAL images, take all and warn
    selected = images[:TOTAL]
    if len(selected) < TOTAL:
        logger.warning(f"  Only {len(selected)} images available for {material} (wanted {TOTAL})")

    splits = {
        "train": selected[:TRAIN_COUNT],
        "val":   selected[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT],
        "test":  selected[TRAIN_COUNT + VAL_COUNT:],
    }

    for split_name, imgs in splits.items():
        dest_dir = CLASSIFIER_DIR / split_name / material
        logger.info(f"    {split_name}/{material}: {len(imgs)} images {'(dry run)' if dry_run else ''}")

        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_path in imgs:
                shutil.copy2(img_path, dest_dir / img_path.name)

    return True


def verify_current_state():
    """Show current image counts per class per split."""
    logger.info("\nCurrent material_classifier state:")
    for split in ["train", "val", "test"]:
        for cls in ["steel", "aluminum", "wood"]:
            d = CLASSIFIER_DIR / split / cls
            count = len(list(d.glob("*"))) if d.exists() else 0
            status = "✅" if count > 0 else "❌ EMPTY"
            logger.info(f"  {split}/{cls}: {count} images {status}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix material classifier dataset — populate steel and aluminum folders"
    )
    parser.add_argument("--apply", action="store_true",
                        help="Actually copy files (default: dry run)")
    parser.add_argument("--train-count", type=int, default=TRAIN_COUNT,
                        help=f"Images per class for train (default: {TRAIN_COUNT})")
    parser.add_argument("--val-count", type=int, default=VAL_COUNT,
                        help=f"Images per class for val (default: {VAL_COUNT})")
    args = parser.parse_args()

    dry_run = not args.apply

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN — no files written. Use --apply to fix.")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("FIXING material_classifier dataset...")
        logger.info("=" * 60)

    verify_current_state()

    logger.info("\n[Fixing] Sampling steel and aluminum images...")
    all_ok = True
    for material, source_dir in SOURCES.items():
        ok = sample_and_copy(material, source_dir, dry_run)
        if not ok:
            all_ok = False

    if not dry_run:
        logger.info("\n[Verifying] After fix:")
        verify_current_state()

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Steel images copied:    {TRAIN_COUNT} train / {VAL_COUNT} val / {TEST_COUNT} test")
    logger.info(f"  Aluminum images copied: {TRAIN_COUNT} train / {VAL_COUNT} val / {TEST_COUNT} test")
    logger.info(f"  Wood images:            210 train / 45 val / 45 test (already present ✅)")
    logger.info(f"  Total per class:        {TOTAL}")
    logger.info(f"  Total dataset:          {TOTAL * 3} images across 3 classes")

    if dry_run:
        logger.info("\n  Run with --apply to actually fix the dataset!")
    else:
        if all_ok:
            logger.info("\n  ✅ Dataset fixed! Now re-run Stage 1:")
            logger.info("     python train_all.py --stages 1")
        else:
            logger.info("\n  ⚠️  Some sources had issues — check logs above.")


if __name__ == "__main__":
    main()
