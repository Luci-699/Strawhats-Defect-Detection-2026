"""
convert_ali2018_aluminum.py
============================
Converts the ali2018 aluminum defect dataset from VOC-XML format into YOLO
bounding box labels, then merges with the existing processed_aluminum split.

ali2018 has 11 defect categories. We map them to the closest existing GC10-DET
class so we don't break the dataset.yaml (still 10 classes):

  ali2018 class               → GC10-DET class id  (name)
  --------------------------    -----------------   ------
  Be injured by a collision   → 0                  crease
  Coating cracking            → 1                  crescent_gap
  Convex powder               → 6                  oil_spot
  Dirty spot                  → 6                  oil_spot
  Drain bottom                → 3                  welding_line
  Orange peel                 → 2                  water_spot
  The transverse strip dented → 0                  crease
  non-conducting              → 4                  silk_spot
  pitting                     → 8                  rolling_pit
  scuffing                    → 9                  waist_fold
  Clean sample                → SKIP (no defect)

Usage:
    python scripts/convert_ali2018_aluminum.py          # dry run
    python scripts/convert_ali2018_aluminum.py --apply  # write files
"""

import argparse
import logging
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

ALI2018_DIR  = PROJECT_ROOT / "data" / "Aluminum" / "ali2018"
LABEL_DIR    = PROJECT_ROOT / "data" / "Aluminum" / "lable"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed_aluminum"

RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15

# Map ali2018 class name → GC10-DET class index
CLASS_MAP = {
    "be injured by a collision":    0,   # crease
    "coating cracking":             1,   # crescent_gap
    "convex powder":                6,   # oil_spot
    "dirty spot":                   6,   # oil_spot
    "drain bottom":                 3,   # welding_line
    "orange peel":                  2,   # water_spot
    "the transverse strip is dented": 0, # crease
    "non-conducting":               4,   # silk_spot
    "pitting":                      8,   # rolling_pit
    "scuffing":                     9,   # waist_fold
    # "clean sample" → intentionally omitted (no defect)
}

SKIP_CLASSES = {"clean sample"}


def find_image_for_xml(xml_stem: str, search_dirs: list[Path]) -> Path | None:
    """Search ali2018 subfolders for the image matching an xml filename stem."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        for d in search_dirs:
            candidate = d / (xml_stem + ext)
            if candidate.exists():
                return candidate
    return None


def parse_voc_xml(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """
    Parse a VOC XML annotation and return YOLO label lines.
    Returns [] if the image should be skipped (clean sample / no defect).
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        logger.warning(f"Malformed XML: {xml_path.name} — skipping")
        return []

    lines = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None:
            continue
        class_name = name_el.text.strip().lower()

        if class_name in SKIP_CLASSES:
            return []  # Whole image is defect-free → skip

        class_id = CLASS_MAP.get(class_name)
        if class_id is None:
            logger.debug(f"Unknown class '{class_name}' in {xml_path.name} — skipping object")
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except (AttributeError, ValueError, TypeError):
            continue

        # Clamp to image bounds
        xmin, xmax = max(0, xmin), min(img_w, xmax)
        ymin, ymax = max(0, ymin), min(img_h, ymax)

        if xmax <= xmin or ymax <= ymin:
            continue

        # Convert to YOLO format (normalized cx cy w h)
        cx = ((xmin + xmax) / 2.0) / img_w
        cy = ((ymin + ymax) / 2.0) / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h

        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def get_image_size(img_path: Path) -> tuple[int, int]:
    """Return (width, height) using PIL to avoid heavy OpenCV import."""
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            return im.width, im.height
    except Exception:
        return 640, 640  # Fallback default


def convert_ali2018(dry_run: bool = True) -> list[tuple[Path, list[str]]]:
    """
    Parse all ali2018 XML labels, find matching images, generate YOLO labels.
    Returns list of (image_path, yolo_lines) for all valid defect images.
    """
    ali2018_subdirs = [d for d in ALI2018_DIR.iterdir() if d.is_dir()]
    xml_files = list(LABEL_DIR.glob("*.xml"))
    logger.info(f"Found {len(xml_files)} XML label files in lable/")
    logger.info(f"Searching for images in {len(ali2018_subdirs)} ali2018 subfolders")

    results = []
    skipped_no_img   = 0
    skipped_clean    = 0
    skipped_no_label = 0

    for xml_path in xml_files:
        img_path = find_image_for_xml(xml_path.stem, ali2018_subdirs)
        if img_path is None:
            skipped_no_img += 1
            continue

        img_w, img_h = get_image_size(img_path)
        yolo_lines = parse_voc_xml(xml_path, img_w, img_h)

        if yolo_lines == [] and any(
            obj.find("name") is not None and
            obj.find("name").text.strip().lower() in SKIP_CLASSES
            for obj in ET.parse(xml_path).getroot().findall("object")
        ):
            skipped_clean += 1
            continue

        if not yolo_lines:
            skipped_no_label += 1
            continue

        results.append((img_path, yolo_lines))

    logger.info(f"  Valid defect images: {len(results)}")
    logger.info(f"  Skipped (image not found): {skipped_no_img}")
    logger.info(f"  Skipped (clean sample):    {skipped_clean}")
    logger.info(f"  Skipped (no label lines):  {skipped_no_label}")
    return results


def append_to_splits(pairs: list[tuple[Path, list[str]]], dry_run: bool = True) -> None:
    """Stratified-split ali2018 pairs and append to existing processed_aluminum splits."""
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }

    for split_name, split_pairs in splits.items():
        out_img = OUTPUT_DIR / "images" / split_name
        out_lbl = OUTPUT_DIR / "labels" / split_name
        logger.info(f"  {split_name}: +{len(split_pairs)} images {'(dry run)' if dry_run else ''}")

        if not dry_run:
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

            for img_path, yolo_lines in split_pairs:
                # Handle name collision
                dest_img = out_img / img_path.name
                dest_lbl = out_lbl / (img_path.stem + ".txt")
                counter = 1
                while dest_img.exists():
                    stem = f"{img_path.stem}_ali{counter}"
                    dest_img = out_img / (stem + img_path.suffix)
                    dest_lbl = out_lbl / (stem + ".txt")
                    counter += 1

                shutil.copy2(img_path, dest_img)
                dest_lbl.write_text("\n".join(yolo_lines))


def main():
    parser = argparse.ArgumentParser(
        description="Convert ali2018 aluminum dataset to YOLO and merge into processed_aluminum"
    )
    parser.add_argument("--apply", action="store_true",
                        help="Actually write files (default: dry-run only)")
    args = parser.parse_args()
    dry_run = not args.apply

    if dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN — no files written. Use --apply to execute.")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("APPLYING — writing to data/processed_aluminum/")
        logger.info("=" * 60)

    logger.info("\n[1/2] Converting ali2018 VOC-XML → YOLO format...")
    pairs = convert_ali2018(dry_run=dry_run)

    logger.info(f"\n[2/2] Merging {len(pairs)} new images into processed_aluminum splits...")
    append_to_splits(pairs, dry_run=dry_run)

    # Count existing + new
    existing_train = len(list((OUTPUT_DIR / "images" / "train").glob("*"))) if (OUTPUT_DIR / "images" / "train").exists() else 1604
    n_train_new = int(len(pairs) * TRAIN_RATIO)

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  ali2018 valid pairs found:  {len(pairs)}")
    logger.info(f"  New train images:           ~{n_train_new}")
    logger.info(f"  Existing train images:      ~{existing_train}")
    logger.info(f"  Total after merge:          ~{existing_train + n_train_new}")
    if dry_run:
        logger.info("\n  Run with --apply to write the files!")
    else:
        logger.info(f"\n  Done! Retrain with: python train_all.py --stages 3")


if __name__ == "__main__":
    main()
