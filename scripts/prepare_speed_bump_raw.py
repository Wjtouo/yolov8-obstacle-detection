from pathlib import Path
import random
import shutil

RAW_DIR = Path(r"H:\yolov8_obstacle_thesis\datasets\incoming\speed_bump\raw")
TARGET_ROOT = Path(r"H:\yolov8_obstacle_thesis\datasets\incoming\speed_bump")
TRAIN_RATIO = 0.8
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def ensure_dirs() -> None:
    for split in ["train", "val"]:
        (TARGET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def collect_pairs():
    pairs = []
    for img in RAW_DIR.rglob("*"):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        txt = img.with_suffix(".txt")
        if txt.exists():
            pairs.append((img, txt))
    return pairs


def main() -> None:
    ensure_dirs()
    pairs = collect_pairs()
    if not pairs:
        print("No image-label pairs found in raw directory.")
        return

    random.seed(SEED)
    random.shuffle(pairs)
    train_count = int(len(pairs) * TRAIN_RATIO)

    stats = {"train": 0, "val": 0}
    for idx, (img, txt) in enumerate(pairs):
        split = "train" if idx < train_count else "val"
        dst_img = TARGET_ROOT / "images" / split / img.name
        dst_txt = TARGET_ROOT / "labels" / split / txt.name
        shutil.copy2(img, dst_img)
        shutil.copy2(txt, dst_txt)
        stats[split] += 1

    print(f"total_pairs={len(pairs)}")
    print(f"train={stats['train']}")
    print(f"val={stats['val']}")
    print("Prepared speed_bump dataset under incoming/speed_bump.")


if __name__ == "__main__":
    main()
