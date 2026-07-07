import argparse
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


def apply_perturbation(image, perturbation):
    if perturbation == "low_light":
        out = cv2.convertScaleAbs(image, alpha=0.65, beta=-25)
    elif perturbation == "high_light":
        out = cv2.convertScaleAbs(image, alpha=1.20, beta=20)
    elif perturbation == "occlusion":
        out = image.copy()
        h, w = out.shape[:2]
        occ_w = int(w * 0.22)
        occ_h = int(h * 0.22)
        x0 = int(w * 0.38)
        y0 = int(h * 0.38)
        cv2.rectangle(out, (x0, y0), (x0 + occ_w, y0 + occ_h), (0, 0, 0), thickness=-1)
    else:
        out = image
    return out


def load_dataset_yaml(dataset_yaml_path):
    with open(dataset_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    root = Path(data_cfg["path"])
    val_rel = data_cfg["val"]
    val_images = (root / val_rel).resolve()
    val_labels = (root / "labels" / "val").resolve()

    return data_cfg, root, val_images, val_labels


def evaluate_map(model, data_yaml, split):
    metrics = model.val(data=data_yaml, split=split, verbose=False)
    return float(metrics.box.map50), float(metrics.box.map)


def benchmark_fps(model, image_files, imgsz, device, warmup, runs):
    if len(image_files) == 0:
        return 0.0

    warmup_files = image_files[: max(1, min(warmup, len(image_files)))]
    for image_path in warmup_files:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        model.predict(source=img, imgsz=imgsz, device=device, verbose=False)

    timing_files = image_files[: max(1, min(runs, len(image_files)))]
    t0 = time.perf_counter()
    count = 0
    for image_path in timing_files:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        model.predict(source=img, imgsz=imgsz, device=device, verbose=False)
        count += 1
    t1 = time.perf_counter()

    if count == 0:
        return 0.0
    return count / max(t1 - t0, 1e-6)


def build_perturbed_dataset(tmp_dir, original_data_cfg, src_val_images, src_val_labels, perturbation, max_images):
    tmp_root = Path(tmp_dir) / f"eval_{perturbation}"
    tmp_images_val = tmp_root / "images" / "val"
    tmp_labels_val = tmp_root / "labels" / "val"
    tmp_images_val.mkdir(parents=True, exist_ok=True)
    tmp_labels_val.mkdir(parents=True, exist_ok=True)

    image_files = sorted(src_val_images.glob("*"))
    image_files = [p for p in image_files if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    if max_images > 0:
        image_files = image_files[:max_images]

    for src_img in image_files:
        img = cv2.imread(str(src_img))
        if img is None:
            continue

        dst_img = tmp_images_val / src_img.name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        out = apply_perturbation(img, perturbation)
        cv2.imwrite(str(dst_img), out)

        label_name = src_img.with_suffix(".txt").name
        src_label = src_val_labels / label_name
        if src_label.exists():
            shutil.copy2(src_label, tmp_labels_val / label_name)

    tmp_yaml = tmp_root / "dataset.yaml"
    tmp_cfg = {
        "path": str(tmp_root),
        "train": "images/val",
        "val": "images/val",
        "names": original_data_cfg["names"],
    }
    with open(tmp_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(tmp_cfg, f, sort_keys=False, allow_unicode=True)

    return tmp_yaml


def main():
    parser = argparse.ArgumentParser(description="Evaluate thesis requirements: mAP/FPS/robustness")
    parser.add_argument("--model", type=str, required=True, help="Path to trained best.pt")
    parser.add_argument("--data", type=str, default="configs/dataset.yaml", help="Dataset yaml path")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--fps-warmup", type=int, default=20)
    parser.add_argument("--fps-runs", type=int, default=200)
    parser.add_argument("--robust-max-images", type=int, default=500)
    parser.add_argument("--target-map50", type=float, default=0.80)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--max-map50-drop", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default="runs_obstacle/requirement_report.json")
    args = parser.parse_args()

    random.seed(args.seed)
    model = YOLO(args.model)

    data_cfg, _, val_images_dir, val_labels_dir = load_dataset_yaml(Path(args.data))
    val_images = sorted(val_images_dir.glob("*"))
    val_images = [p for p in val_images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    random.shuffle(val_images)

    map50, map5095 = evaluate_map(model, args.data, args.split)
    fps = benchmark_fps(model, val_images, args.imgsz, args.device, args.fps_warmup, args.fps_runs)

    robustness = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        for perturb in ["low_light", "high_light", "occlusion"]:
            tmp_yaml = build_perturbed_dataset(
                tmp_dir=tmp_dir,
                original_data_cfg=data_cfg,
                src_val_images=val_images_dir,
                src_val_labels=val_labels_dir,
                perturbation=perturb,
                max_images=args.robust_max_images,
            )
            p_map50, p_map5095 = evaluate_map(model, str(tmp_yaml), "val")
            robustness[perturb] = {"map50": p_map50, "map50_95": p_map5095, "drop": map50 - p_map50}

    max_drop = max([v["drop"] for v in robustness.values()]) if robustness else 1.0
    pass_map = map50 >= args.target_map50
    pass_fps = fps >= args.target_fps
    pass_robust = max_drop <= args.max_map50_drop

    report = {
        "model": args.model,
        "data": args.data,
        "metrics": {
            "map50": map50,
            "map50_95": map5095,
            "fps": fps,
        },
        "robustness": robustness,
        "targets": {
            "map50_min": args.target_map50,
            "fps_min": args.target_fps,
            "max_map50_drop": args.max_map50_drop,
        },
        "passed": {
            "map50": pass_map,
            "fps": pass_fps,
            "robustness": pass_robust,
            "all": pass_map and pass_fps and pass_robust,
        },
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Requirement Evaluation ===")
    print(f"mAP50:     {map50:.4f} (target >= {args.target_map50:.2f}) -> {pass_map}")
    print(f"mAP50-95:  {map5095:.4f}")
    print(f"FPS:       {fps:.2f} (target >= {args.target_fps:.1f}) -> {pass_fps}")
    print(f"Max drop:  {max_drop:.4f} (target <= {args.max_map50_drop:.2f}) -> {pass_robust}")
    print(f"ALL PASS:  {report['passed']['all']}")
    print(f"Saved report: {out_path.resolve()}")


if __name__ == "__main__":
    main()
