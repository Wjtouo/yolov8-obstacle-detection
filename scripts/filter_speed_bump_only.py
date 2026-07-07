from pathlib import Path
import shutil
import uuid
from collections import Counter

SRC = Path(r"H:\yolov8_obstacle_thesis\datasets\incoming\speed_bump")
DST = Path(r"H:\yolov8_obstacle_thesis\datasets\incoming\speed_bump_only")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEEP_ID = 5


def ensure_dirs():
    for split in ["train", "val"]:
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)


def filter_label(src_txt: Path, dst_txt: Path):
    out = []
    kept = 0
    for ln in src_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = ln.strip().split()
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        if int(parts[0]) != KEEP_ID:
            continue
        out.append(ln.strip())
        kept += 1
    if kept:
        dst_txt.write_text("\n".join(out), encoding="utf-8")
    return kept


def main():
    ensure_dirs()
    total = Counter()
    for split in ["train", "val"]:
        src_img = SRC / "images" / split
        src_lab = SRC / "labels" / split
        for img in src_img.glob("*"):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lab = src_lab / f"{img.stem}.txt"
            if not lab.exists():
                continue
            tmp = DST / "labels" / split / f"{img.stem}_{uuid.uuid4().hex[:8]}.txt"
            kept = filter_label(lab, tmp)
            if kept == 0:
                tmp.unlink(missing_ok=True)
                continue
            shutil.copy2(img, DST / "images" / split / f"{tmp.stem}{img.suffix.lower()}")
            total[split] += 1
    print(f"train_val_images={dict(total)}")
    # class count check
    ctr = Counter()
    for split in ["train", "val"]:
        for f in (DST / "labels" / split).glob("*.txt"):
            for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = ln.strip().split()
                if s and s[0].isdigit():
                    ctr[int(s[0])] += 1
    print(f"class_counts={dict(sorted(ctr.items()))}")


if __name__ == "__main__":
    main()
