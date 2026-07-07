from pathlib import Path
import shutil
import uuid

ROOT = Path(r"H:\yolov8_obstacle_thesis")
INCOMING = ROOT / "datasets" / "incoming"
TARGET = ROOT / "datasets" / "custom_new"

CLASS_ID = {
    "speed_bump": 5,
    "tree_trunk": 6,
    "stone_bollard": 7,
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dirs():
    for s in ["train", "val"]:
        (TARGET / "images" / s).mkdir(parents=True, exist_ok=True)
        (TARGET / "labels" / s).mkdir(parents=True, exist_ok=True)


def remap_label_file(src_txt: Path, dst_txt: Path, target_id: int):
    lines_out = []
    txt = src_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    for ln in txt:
        sp = ln.strip().split()
        if len(sp) < 5:
            continue
        sp[0] = str(target_id)
        lines_out.append(" ".join(sp))
    dst_txt.write_text("\n".join(lines_out), encoding="utf-8")


def process_class(cls_name: str):
    cls_dir = INCOMING / cls_name
    if not cls_dir.exists():
        return 0, 0

    target_id = CLASS_ID[cls_name]
    n_img, n_lab = 0, 0

    for split in ["train", "val"]:
        src_img_dir = cls_dir / "images" / split
        src_lab_dir = cls_dir / "labels" / split
        if not src_img_dir.exists() or not src_lab_dir.exists():
            continue

        for img_path in src_img_dir.rglob("*"):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            stem = img_path.stem
            src_txt = src_lab_dir / f"{stem}.txt"
            if not src_txt.exists():
                continue

            new_stem = f"{stem}_{uuid.uuid4().hex[:8]}"
            dst_img = TARGET / "images" / split / f"{new_stem}{img_path.suffix.lower()}"
            dst_txt = TARGET / "labels" / split / f"{new_stem}.txt"

            shutil.copy2(img_path, dst_img)
            remap_label_file(src_txt, dst_txt, target_id)
            n_img += 1
            n_lab += 1

    return n_img, n_lab


def main():
    ensure_dirs()
    total_img, total_lab = 0, 0
    for cls_name in ["speed_bump", "tree_trunk", "stone_bollard"]:
        n_img, n_lab = process_class(cls_name)
        total_img += n_img
        total_lab += n_lab
        print(f"[{cls_name}] merged images={n_img}, labels={n_lab}")
    print(f"[DONE] total images={total_img}, labels={total_lab}")


if __name__ == "__main__":
    main()
