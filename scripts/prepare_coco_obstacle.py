import argparse
import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# 毕设导向类别（更贴近“障碍物检测”）
# 注意：COCO里没有 e-bike / cone 等专门类，这里做近似映射
TARGET_CLASSES = [
    "person",          # 行人
    "bicycle",         # 自行车
    "e_bike",          # 由 COCO motorcycle 近似
    "car",             # 小客车
    "large_vehicle",   # 由 COCO bus + truck 合并
    "obstacle",       # 交通设施代理类：traffic light + stop sign（与 BDD 子集第 6 类同名，便于合并训练）
]

# COCO 原类 -> 毕设目标类
SOURCE_TO_TARGET = {
    "person": "person",
    "bicycle": "bicycle",
    "motorcycle": "e_bike",
    "car": "car",
    "bus": "large_vehicle",
    "truck": "large_vehicle",
    "traffic light": "obstacle",
    "stop sign": "obstacle",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path):
    if not path.exists():
        return
    for p in path.rglob("*"):
        if p.is_file():
            p.unlink()


def build_image_records(coco_json_path: Path, class_name_to_new_id: dict):
    """
    把 COCO 标注转成“每张图的 YOLO 标签行”，先不落盘。
    这样可以后续灵活切分 train/val（尤其是 val-only 模式）。
    """
    coco = load_json(coco_json_path)

    coco_catid_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}

    anns_by_image = {}
    for ann in coco["annotations"]:
        image_id = ann["image_id"]
        anns_by_image.setdefault(image_id, []).append(ann)

    records = []

    for image_id, info in tqdm(images.items(), desc=f"Parsing {coco_json_path.name}"):
        file_name = info["file_name"]
        w = info["width"]
        h = info["height"]

        anns = anns_by_image.get(image_id, [])
        yolo_lines = []

        for ann in anns:
            cat_id = ann["category_id"]
            src_name = coco_catid_to_name.get(cat_id)
            target_name = SOURCE_TO_TARGET.get(src_name)
            if target_name is None:
                continue

            new_cls_id = class_name_to_new_id[target_name]
            x, y, bw, bh = ann["bbox"]

            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw_n = bw / w
            bh_n = bh / h

            if bw_n <= 0 or bh_n <= 0:
                continue

            yolo_lines.append(f"{new_cls_id} {x_center:.6f} {y_center:.6f} {bw_n:.6f} {bh_n:.6f}")

        # 没有目标类别则不保留
        if not yolo_lines:
            continue

        records.append({"file_name": file_name, "yolo_lines": yolo_lines})

    return records


def write_split(records, image_src_dir: Path, image_dst_dir: Path, label_dst_dir: Path):
    kept_images = 0
    kept_boxes = 0

    ensure_dir(image_dst_dir)
    ensure_dir(label_dst_dir)

    for item in tqdm(records, desc=f"Writing -> {image_dst_dir.name}"):
        file_name = item["file_name"]
        yolo_lines = item["yolo_lines"]

        src_img = image_src_dir / file_name
        dst_img = image_dst_dir / file_name
        ensure_dir(dst_img.parent)

        if not src_img.exists():
            continue

        shutil.copy2(src_img, dst_img)

        label_name = Path(file_name).with_suffix(".txt").name
        dst_label = label_dst_dir / label_name
        ensure_dir(dst_label.parent)

        with open(dst_label, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

        kept_images += 1
        kept_boxes += len(yolo_lines)

    return kept_images, kept_boxes


def main():
    parser = argparse.ArgumentParser(description="Convert COCO to YOLOv8 obstacle subset")
    parser.add_argument("--coco-root", type=str, required=True, help="COCO root directory")
    parser.add_argument("--out-root", type=str, default="datasets/coco_obstacle", help="Output dataset root")

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "val-only"],
        help="full: 用 train2017+val2017；val-only: 只用 val2017 后再切分 train/val",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="val-only 模式下验证集占比")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（保证可复现）")
    parser.add_argument("--clear-out", action="store_true", help="清空 out-root 下已有图片/标签")

    args = parser.parse_args()

    coco_root = Path(args.coco_root)
    out_root = Path(args.out_root)

    img_train_dst = out_root / "images" / "train"
    img_val_dst = out_root / "images" / "val"
    lbl_train_dst = out_root / "labels" / "train"
    lbl_val_dst = out_root / "labels" / "val"

    for p in [img_train_dst, img_val_dst, lbl_train_dst, lbl_val_dst]:
        ensure_dir(p)

    if args.clear_out:
        for p in [img_train_dst, img_val_dst, lbl_train_dst, lbl_val_dst]:
            clear_dir(p)

    class_name_to_new_id = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

    print("\nTarget classes:")
    for k, v in class_name_to_new_id.items():
        print(f"  {v}: {k}")

    if args.mode == "full":
        ann_train = coco_root / "annotations" / "instances_train2017.json"
        ann_val = coco_root / "annotations" / "instances_val2017.json"
        img_train_src = coco_root / "train2017"
        img_val_src = coco_root / "val2017"

        train_records = build_image_records(ann_train, class_name_to_new_id)
        val_records = build_image_records(ann_val, class_name_to_new_id)

        tr_imgs, tr_boxes = write_split(train_records, img_train_src, img_train_dst, lbl_train_dst)
        va_imgs, va_boxes = write_split(val_records, img_val_src, img_val_dst, lbl_val_dst)

    else:
        ann_val = coco_root / "annotations" / "instances_val2017.json"
        img_val_src = coco_root / "val2017"

        all_records = build_image_records(ann_val, class_name_to_new_id)

        random.seed(args.seed)
        random.shuffle(all_records)

        n_total = len(all_records)
        n_val = int(n_total * args.val_ratio)
        n_val = max(1, min(n_val, n_total - 1))

        val_records = all_records[:n_val]
        train_records = all_records[n_val:]

        print(f"\n[val-only] total kept images = {n_total}")
        print(f"[val-only] split -> train={len(train_records)}, val={len(val_records)}")

        tr_imgs, tr_boxes = write_split(train_records, img_val_src, img_train_dst, lbl_train_dst)
        va_imgs, va_boxes = write_split(val_records, img_val_src, img_val_dst, lbl_val_dst)

    print("\nDone!")
    print(f"Train: images={tr_imgs}, boxes={tr_boxes}")
    print(f"Val:   images={va_imgs}, boxes={va_boxes}")
    print(f"Output dataset path: {out_root.resolve()}")


if __name__ == "__main__":
    main()
