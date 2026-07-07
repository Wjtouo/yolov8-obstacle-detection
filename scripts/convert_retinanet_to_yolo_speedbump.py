from __future__ import annotations

from pathlib import Path
import csv
import shutil
from collections import defaultdict

SRC = Path(r"H:\yolov8_obstacle_thesis\datasets\incoming\speed_bump\raw\road dataset 2.v1i.retinanet")
OUT = Path(r"H:\yolov8_obstacle_thesis\datasets\incoming\speed_bump_retina_yolo")
KEEP_CLASS = "speedbump"
YOLO_CLASS_ID = 5
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dirs():
    for split in ["train", "val", "test"]:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


def find_image(name: str) -> Path | None:
    # Search common image locations inside the source dataset.
    candidates = [
        SRC / name,
        SRC / "train" / name,
        SRC / "valid" / name,
        SRC / "test" / name,
        SRC / "images" / name,
        SRC / "train" / "images" / name,
        SRC / "valid" / "images" / name,
        SRC / "test" / "images" / name,
    ]
    for c in candidates:
        if c.exists() and c.suffix.lower() in IMG_EXTS:
            return c
    # Slow recursive fallback if needed.
    for c in SRC.rglob(name):
        if c.is_file() and c.suffix.lower() in IMG_EXTS:
            return c
    return None


def clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def to_yolo_bbox(x1, y1, x2, y2, img_w, img_h):
    x1 = float(x1); y1 = float(y1); x2 = float(x2); y2 = float(y2)
    img_w = float(img_w); img_h = float(img_h)
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return clamp(xc), clamp(yc), clamp(w), clamp(h)


def process_csv(csv_path: Path, split: str):
    kept = 0
    dropped = 0
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.reader(fh)
        grouped = defaultdict(list)
        for row in reader:
            if len(row) < 6:
                continue
            img_name, x1, y1, x2, y2, cls = row[:6]
            if cls.strip().lower() != KEEP_CLASS:
                dropped += 1
                continue
            grouped[img_name].append((x1, y1, x2, y2))

    for img_name, boxes in grouped.items():
        src_img = find_image(img_name)
        if src_img is None:
            continue
        dst_stem = Path(img_name).stem
        dst_img = OUT / "images" / split / f"{dst_stem}{src_img.suffix.lower()}"
        dst_txt = OUT / "labels" / split / f"{dst_stem}.txt"
        shutil.copy2(src_img, dst_img)

        # Need image size for normalization.
        try:
            from PIL import Image
            with Image.open(src_img) as im:
                img_w, img_h = im.size
        except Exception:
            # If Pillow isn't available, skip this image.
            dst_img.unlink(missing_ok=True)
            continue

        lines = []
        for x1, y1, x2, y2 in boxes:
            xc, yc, w, h = to_yolo_bbox(x1, y1, x2, y2, img_w, img_h)
            lines.append(f"{YOLO_CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        if lines:
            dst_txt.write_text("\n".join(lines), encoding="utf-8")
            kept += 1
        else:
            dst_img.unlink(missing_ok=True)
    return kept, dropped


def main():
    ensure_dirs()
    total_kept = 0
    total_dropped = 0
    mapping = {
        "train": "_annotations_train.csv",
        "val": "_annotations_valid.csv",
        "test": "_annotations_test.csv",
    }
    for split, fn in mapping.items():
        kept, dropped = process_csv(SRC / fn, split)
        total_kept += kept
        total_dropped += dropped
        print(f"[{split}] kept_images={kept}, dropped_non_speedbump={dropped}")
    print(f"[TOTAL] kept_images={total_kept}, dropped_non_speedbump={total_dropped}")
    print(f"YOLO dataset saved to: {OUT}")


if __name__ == "__main__":
    main()
