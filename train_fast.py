# -*- coding: utf-8 -*-
"""
YOLOv8s 障碍物检测训练脚本 (快速版)
目标: 快速验证模型，降低mAP要求以换取训练速度
硬件: RTX 4060 Laptop 8GB
"""
from ultralytics import YOLO
import torch

def main():
    print("=" * 60)
    print("启动 YOLOv8s 快速训练模式...")
    print("=" * 60)

    # 换用 yolov8s.pt (Small版本)，参数量只有 yolov8m 的三分之一，训练速度快 2-3 倍
    model = YOLO("yolov8s.pt")

    model.train(
        data="configs/dataset.yaml",

        # ── 提速核心参数 ────────────────────────────────
        epochs=50,          # 总轮数大幅减少到 50 轮
        imgsz=640,
        batch=16,           # 降低batch防止显存/内存溢出
        device="0",
        workers=2,          # 降低workers防止内存溢出
        cache=False,        # 关闭cache防止虚拟内存不足

        # ── 输出路径 ────────────────────────────────────
        project="H:/yolo_runs",
        name="bdd_s_fast",
        exist_ok=True,
        save_period=10,

        # ── 优化器 / 学习率 ─────────────────────────────
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,    # 缩短预热期
        momentum=0.937,
        weight_decay=0.0005,

        # ── 早停 ────────────────────────────────────────
        patience=20,        # 20轮不提升就早停

        # ── 数据增强 (适度简化以加快收敛) ───────────────
        close_mosaic=10,    # 最后10轮关闭mosaic
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,          # 关闭 mixup (计算量大且对快速收敛帮助有限)
        copy_paste=0.0,     # 关闭 copy_paste

        # ── 其他 ────────────────────────────────────────
        amp=False,          # 保持关闭，防止 4060 报错
        pretrained=True,
        verbose=True,
        plots=True,
    )

    print("\n训练完成！最优模型保存在 H:/yolo_runs/bdd_s_fast/weights/best.pt")

if __name__ == "__main__":
    main()
