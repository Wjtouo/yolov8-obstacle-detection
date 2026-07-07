# -*- coding: utf-8 -*-
"""
在 BDD + COCO 合并数据集上微调（继续训练），提升精度。

前置步骤：
1) 已存在 H:/yolo_datasets/bdd100k_obstacle 与 H:/yolo_datasets/coco_obstacle
2) 运行合并（示例见下）
3) 再运行本脚本

合并示例：
  .venv\\Scripts\\python.exe scripts\\merge_yolo_datasets.py ^
    --primary H:/yolo_datasets/bdd100k_obstacle ^
    --secondary H:/yolo_datasets/coco_obstacle ^
    --out H:/yolo_datasets/bdd_coco_merged ^
    --secondary-prefix coco_ --clear-out
"""
from pathlib import Path

from ultralytics import YOLO
import torch


def main():
    base = Path("H:/yolo_runs/bdd_s_fast/weights/best.pt")
    if not base.exists():
        print(f"未找到 {base}，请修改为本机已有 best.pt 路径")
        return

    data_yaml = Path("H:/yolov8_obstacle_thesis/configs/dataset_merged.yaml")
    merged_root = Path("H:/yolo_datasets/bdd_coco_merged/images/train")
    if not merged_root.exists() or not any(merged_root.iterdir()):
        print("未检测到合并数据目录或为空，请先运行 scripts\\merge_yolo_datasets.py")
        return

    print("=" * 60)
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    model = YOLO(str(base))
    model.train(
        data=str(data_yaml),
        epochs=40,
        imgsz=640,
        batch=12,
        device="0",
        workers=2,
        project="H:/yolo_runs",
        name="bdd_coco_finetune",
        exist_ok=True,
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=2,
        patience=15,
        close_mosaic=10,
        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.0,
        amp=False,
        pretrained=False,
        verbose=True,
        plots=True,
    )
    print("\n完成。最优权重: H:/yolo_runs/bdd_coco_finetune/weights/best.pt")


if __name__ == "__main__":
    main()
