import argparse
import json
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 毕设导向类别（校园/小区门口）
TARGET_CLASSES = [
    "person",         # 行人
    "bicycle",        # 自行车
    "e_bike",         # 电动车/摩托/骑行者
    "car",            # 小客车
    "large_vehicle",  # 大型车辆（bus/truck/train）
    "obstacle",       # 交通设施/障碍物（traffic light/sign 近似）
]

# BDD100K 原类 -> 目标类
SOURCE_TO_TARGET = {
    "person": "person",
    "rider": "e_bike",
    "bicycle": "bicycle",
    "motorcycle": "e_bike",
    "car": "car",
    "truck": "large_vehicle",
    "bus": "large_vehicle",
    "train": "large_vehicle",
    "traffic light": "obstacle",
    "traffic sign": "obstacle",
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


def build_image_records(bdd_json_path: Path):
    bdd = load_json(bdd_json_path)
    records = {}

    for item in tqdm(bdd, desc=f"Parsing {bdd_json_path.name}"):
        file_name = item.get("name")
        labels = item.get("labels", [])
        if not file_name or not labels:
            continue

        width = item.get("width")
        height = item.get("height")

        for lab in labels:
            category = lab.get("category")
            target_name = SOURCE_TO_TARGET.get(category)
            if target_name is None:
                continue

            box = lab.get("box2d")
            if not box:
                continue

            x1, y1, x2, y2 = box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")
            if None in (x1, y1, x2, y2):
                continue

            rec = records.setdefault(
                file_name,
                {"width": width, "height": height, "boxes": []},
            )
            if rec["width"] is None and width is not None:
                rec["width"] = width
            if rec["height"] is None and height is not None:
                rec["height"] = height

            rec["boxes"].append((target_name, x1, y1, x2, y2))

    return records


def write_split(records_by_file, image_src_dirs, image_dst_dir: Path, label_dst_dir: Path, class_name_to_new_id: dict):
    kept_images = 0
    kept_boxes = 0

    ensure_dir(image_dst_dir)
    ensure_dir(label_dst_dir)

    for file_name, rec in tqdm(records_by_file.items(), desc=f"Writing -> {image_dst_dir.name}"):
        src_img = None
        for src_dir in image_src_dirs:
            candidate = src_dir / file_name
            if candidate.exists():
                src_img = candidate
                break

        if src_img is None:
            continue

        width = rec["width"]
        height = rec["height"]
        if width is None or height is None:
            try:
                with Image.open(src_img) as im:
                    width, height = im.size
            except Exception:
                continue

        yolo_lines = []
        for target_name, x1, y1, x2, y2 in rec["boxes"]:
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            bw_n = bw / width
            bh_n = bh / height

            if bw_n <= 0 or bh_n <= 0:
                continue

            new_cls_id = class_name_to_new_id[target_name]
            yolo_lines.append(f"{new_cls_id} {x_center:.6f} {y_center:.6f} {bw_n:.6f} {bh_n:.6f}")

        if not yolo_lines:
            continue

        dst_img = image_dst_dir / file_name
        ensure_dir(dst_img.parent)
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
    parser = argparse.ArgumentParser(description="Convert BDD100K to YOLOv8 obstacle subset")
    parser.add_argument("--bdd-root", type=str, required=True, help="BDD100K root directory")
    parser.add_argument("--out-root", type=str, default="datasets/bdd100k_obstacle", help="Output dataset root")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Split ratio when using train subset only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit", type=int, default=0, help="Limit images per split (0=all)")
    parser.add_argument("--clear-out", action="store_true", help="Clear out-root before write")

    args = parser.parse_args()

    bdd_root = Path(args.bdd_root)
    out_root = Path(args.out_root)

    # 兼容两种目录：train 或 trainA+trainB
    train_root = bdd_root / "images" / "100k"
    img_train_src_dirs = []
    if (train_root / "train").exists():
        img_train_src_dirs.append(train_root / "train")
    if (train_root / "trainA").exists():
        img_train_src_dirs.append(train_root / "trainA")
    if (train_root / "trainB").exists():
        img_train_src_dirs.append(train_root / "trainB")

    img_val_src_dirs = [bdd_root / "images" / "100k" / "val"]

    ann_train = bdd_root / "labels" / "det_20" / "det_train.json"
    ann_val = bdd_root / "labels" / "det_20" / "det_val.json"

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

    train_records = build_image_records(ann_train)
    val_records = build_image_records(ann_val)

    # 以文件级别打乱/裁剪
    train_items = list(train_records.items())
    val_items = list(val_records.items())

    random.seed(args.seed)
    random.shuffle(train_items)
    random.shuffle(val_items)

    if args.limit and args.limit > 0:
        train_items = train_items[: args.limit]
        val_items = val_items[: max(1, int(args.limit * args.val_ratio))]

    train_records = dict(train_items)
    val_records = dict(val_items)

    if not img_train_src_dirs:
        raise FileNotFoundError("No train image directory found. Expected train or trainA/trainB under images/100k")
    if not img_val_src_dirs[0].exists():
        raise FileNotFoundError("No val image directory found at images/100k/val")

    tr_imgs, tr_boxes = write_split(train_records, img_train_src_dirs, img_train_dst, lbl_train_dst, class_name_to_new_id)
    va_imgs, va_boxes = write_split(val_records, img_val_src_dirs, img_val_dst, lbl_val_dst, class_name_to_new_id)

    print("\nDone!")
    print(f"Train: images={tr_imgs}, boxes={tr_boxes}")
    print(f"Val:   images={va_imgs}, boxes={va_boxes}")
    print(f"Output dataset path: {out_root.resolve()}")


if __name__ == "__main__":
    main()
