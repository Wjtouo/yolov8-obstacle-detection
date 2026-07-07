from __future__ import annotations

from pathlib import Path
import csv
import shutil
from collections import defaultdict
from PIL import Image

SRC = Path(r"H:\yolov8_obstacle_thesis\datasets\incoming\speed_bump\raw\road dataset.v2i.retinanet")
OUT = Path(r"H:\yolov8_obstacle_thesis\datasets\incoming\speed_bump_retina2_yolo")
KEEP_CLASS = "speedbump"
YOLO_CLASS_ID = 5


def ensure_dirs():
    for split in ["train", "val"]:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


def to_yolo(x1, y1, x2, y2, w, h):
    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh


def process(split_src: str, split_out: str):
    csv_path = SRC / split_src / "_annotations.csv"
    grouped = defaultdict(list)
    dropped = 0

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        r = csv.reader(fh)
        for row in r:
            if len(row) < 6:
                continue
            name, x1, y1, x2, y2, cls = row[:6]
            if cls.strip().lower() != KEEP_CLASS:
                dropped += 1
                continue
            grouped[name].append((x1, y1, x2, y2))

    kept_images = 0
    for name, boxes in grouped.items():
        src_img = SRC / split_src / name
        if not src_img.exists():
            continue

        with Image.open(src_img) as im:
            w, h = im.size

        dst_img = OUT / "images" / split_out / src_img.name
        dst_txt = OUT / "labels" / split_out / f"{src_img.stem}.txt"
        shutil.copy2(src_img, dst_img)

        lines = []
        for b in boxes:
            xc, yc, bw, bh = to_yolo(*b, w, h)
            lines.append(f"{YOLO_CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        dst_txt.write_text("\n".join(lines), encoding="utf-8")
        kept_images += 1

    return kept_images, dropped


def main():
    ensure_dirs()
    t_kept, t_drop = process("train", "train")
    v_kept, v_drop = process("valid", "val")
    print(f"train_kept_images={t_kept}, train_dropped_non_speedbump={t_drop}")
    print(f"val_kept_images={v_kept}, val_dropped_non_speedbump={v_drop}")
    print(f"saved_to={OUT}")


if __name__ == "__main__":
    main()
