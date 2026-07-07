from __future__ import annotations

from pathlib import Path
import shutil
import uuid
from collections import Counter

ROOT = Path(r"H:\yolov8_obstacle_thesis")
SOURCE = Path(r"H:\yolo_datasets\OOD.v2i.yolov8")
TARGET = ROOT / "datasets" / "incoming" / "ood_filtered"

# Map OOD dataset class names to the project's final classes.
CLASS_MAP = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "bus": "large_vehicle",
    "truck": "large_vehicle",
    "tree": "tree_trunk",
    "spherical_roadblock": "stone_bollard",
    "warning_column": "stone_bollard",
}

# Final project class ids.
TARGET_ID = {
    "person": 0,
    "bicycle": 1,
    "e_bike": 2,
    "car": 3,
    "large_vehicle": 4,
    "speed_bump": 5,
    "tree_trunk": 6,
    "stone_bollard": 7,
}

# OOD dataset class ids from data.yaml.
OOD_NAMES = [
    "bench",
    "bicycle",
    "bus",
    "bus_stop",
    "cane",
    "car",
    "curb",
    "dog",
    "fire_hydrant",
    "motorcycle",
    "person",
    "pole",
    "spherical_roadblock",
    "stairs",
    "stop_sign",
    "street_light",
    "traffic_light",
    "train",
    "tree",
    "truck",
    "warning_column",
    "waste_container",
]
OOD_ID = {name: idx for idx, name in enumerate(OOD_NAMES)}
KEEP_IDS = {OOD_ID[k] for k in CLASS_MAP.keys()}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dirs() -> None:
    for split in ["train", "valid", "test"]:
        (TARGET / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET / "labels" / split).mkdir(parents=True, exist_ok=True)


def target_split_name(source_split: str) -> str:
    return "val" if source_split == "valid" else source_split


def remap_label_file(src_txt: Path, dst_txt: Path) -> Counter:
    counts = Counter()
    out_lines = []
    for ln in src_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = ln.strip().split()
        if len(parts) < 5:
            continue
        if not parts[0].isdigit():
            continue
        src_id = int(parts[0])
        src_name = OOD_NAMES[src_id] if 0 <= src_id < len(OOD_NAMES) else None
        if src_name not in CLASS_MAP:
            continue
        final_name = CLASS_MAP[src_name]
        parts[0] = str(TARGET_ID[final_name])
        out_lines.append(" ".join(parts))
        counts[final_name] += 1
    dst_txt.write_text("\n".join(out_lines), encoding="utf-8")
    return counts


def process_split(split: str) -> Counter:
    src_img_dir = SOURCE / split / "images"
    src_lab_dir = SOURCE / split / "labels"
    if not src_img_dir.exists() or not src_lab_dir.exists():
        raise FileNotFoundError(f"Missing OOD split dirs: {src_img_dir} or {src_lab_dir}")

    stats = Counter()
    for img_path in src_img_dir.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        src_txt = src_lab_dir / f"{img_path.stem}.txt"
        if not src_txt.exists():
            continue

        # Keep only images that contain at least one wanted class.
        out_split = target_split_name(split)
        temp_dst = TARGET / "labels" / out_split / f"{img_path.stem}_{uuid.uuid4().hex[:8]}.txt"
        cls_counts = remap_label_file(src_txt, temp_dst)
        if not cls_counts:
            temp_dst.unlink(missing_ok=True)
            continue

        new_stem = temp_dst.stem
        dst_img = TARGET / "images" / out_split / f"{new_stem}{img_path.suffix.lower()}"
        shutil.copy2(img_path, dst_img)
        stats.update(cls_counts)
    return stats


def main() -> None:
    ensure_dirs()
    total = Counter()
    for split in ["train", "valid", "test"]:
        stats = process_split(split)
        total.update(stats)
        print(f"[{split}] {dict(sorted(stats.items()))}")
    print(f"[TOTAL] {dict(sorted(total.items()))}")
    print(f"Filtered dataset saved to: {TARGET}")


if __name__ == "__main__":
    main()
