# -*- coding: utf-8 -*-
"""
将两个 YOLO 格式数据集合并到同一根目录（train/val）。

前提：class_id 与类别顺序一致（本项目 0..5 与 dataset.yaml 一致）。
辅助集文件默认加前缀，避免与主集重名。
"""
import argparse
import shutil
from pathlib import Path

from tqdm import tqdm


def copy_split_rglob(src_img_dir: Path, src_lbl_dir: Path, dst_img: Path, dst_lbl: Path, prefix: str) -> int:
    """递归收集图片；标签默认同名 .txt 位于 src_lbl_dir 根目录（与 prepare_bdd 脚本一致）。"""
    n = 0
    if not src_img_dir.exists():
        return n
    for img_path in tqdm(list(src_img_dir.rglob("*")), desc=f"merge {'[' + prefix + ']' if prefix else '[primary]'} "):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        rel = img_path.relative_to(src_img_dir)
        if prefix:
            flat_name = prefix + str(rel).replace("\\", "_").replace("/", "_")
        else:
            flat_name = str(rel).replace("\\", "/")
        dst_image = dst_img / flat_name
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dst_image)

        stem = img_path.stem
        lbl = src_lbl_dir / f"{stem}.txt"
        if not lbl.exists():
            lbl = src_lbl_dir / rel.with_suffix(".txt")
        if lbl.exists():
            dst_lbl.mkdir(parents=True, exist_ok=True)
            out_lbl = dst_lbl / (Path(flat_name).stem + ".txt")
            shutil.copy2(lbl, out_lbl)
        n += 1
    return n


def clear_dir(d: Path):
    if not d.exists():
        return
    for p in d.rglob("*"):
        if p.is_file():
            p.unlink()


def main():
    parser = argparse.ArgumentParser(description="Merge two YOLO datasets (same class ids)")
    parser.add_argument("--primary", type=str, required=True)
    parser.add_argument("--secondary", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--secondary-prefix", type=str, default="coco_")
    parser.add_argument("--clear-out", action="store_true")
    args = parser.parse_args()

    primary = Path(args.primary)
    secondary = Path(args.secondary)
    out = Path(args.out)

    for split in ("train", "val"):
        img_o = out / "images" / split
        lbl_o = out / "labels" / split
        if args.clear_out:
            clear_dir(img_o)
            clear_dir(lbl_o)
        img_o.mkdir(parents=True, exist_ok=True)
        lbl_o.mkdir(parents=True, exist_ok=True)

        n1 = copy_split_rglob(primary / "images" / split, primary / "labels" / split, img_o, lbl_o, prefix="")
        pre = args.secondary_prefix or ""
        n2 = copy_split_rglob(secondary / "images" / split, secondary / "labels" / split, img_o, lbl_o, prefix=pre)
        print(f"[{split}] primary={n1}, secondary={n2}, total={n1 + n2}")

    print(f"\nDone. Output: {out.resolve()}")


if __name__ == "__main__":
    main()
