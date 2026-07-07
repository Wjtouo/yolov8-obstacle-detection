# -*- coding: utf-8 -*-
"""
BDD100K 子集 + COCO 子集合并后再训练（在已有 best.pt 上微调）。

- 模型仍是 YOLOv8s，推理速度与此前一致（FPS 取决于显卡与 imgsz）。
- 数据：H:/yolo_datasets/bdd_coco_merged（若无请先 merge）。
- 输出：H:/yolo_runs/bdd_plus_coco/weights/best.pt

运行（在 H 盘工程目录）:
  .venv\\Scripts\\python.exe train_bdd_plus_coco.py
"""
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    ckpt = Path("H:/yolo_runs/bdd_s_fast/weights/best.pt")
    if not ckpt.exists():
        print(f"未找到基准权重 {ckpt}，请改为本机已有的 yolov8s 训练 best.pt 路径")
        return

    data_yaml = Path("H:/yolov8_obstacle_thesis/configs/dataset_merged.yaml")
    merged_train = Path("H:/yolo_datasets/bdd_coco_merged/images/train")
    if not merged_train.exists() or not any(merged_train.iterdir()):
        print("合并数据为空。请先执行：")
        print(
            r'.venv\Scripts\python.exe scripts\merge_yolo_datasets.py '
            r'--primary H:/yolo_datasets/bdd100k_obstacle '
            r'--secondary H:/yolo_datasets/coco_obstacle '
            r'--out H:/yolo_datasets/bdd_coco_merged --secondary-prefix coco_ --clear-out'
        )
        return

    print("=" * 60)
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"数据: {data_yaml}")
    print(f"初始权重: {ckpt}")
    print("=" * 60)

    model = YOLO(str(ckpt))
    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=12,
        device="0",
        workers=2,
        project="H:/yolo_runs",
        name="bdd_plus_coco",
        exist_ok=True,
        # 微调：较小学习率，利于在合并集上稳步涨 mAP
        lr0=0.0008,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        patience=20,
        close_mosaic=10,
        mosaic=1.0,
        mixup=0.08,
        copy_paste=0.05,
        amp=False,
        pretrained=False,
        verbose=True,
        plots=True,
    )
    print("\n完成。请用 GUI 或 val：H:/yolo_runs/bdd_plus_coco/weights/best.pt")


if __name__ == "__main__":
    main()
