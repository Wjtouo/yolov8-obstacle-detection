# YOLOv8 障碍物检测系统设计

基于 YOLOv8 的毕业设计项目，面向障碍物实时检测任务，包含数据准备、模型训练、推理测试、实时摄像头检测，以及达标评估脚本。

## 中文摘要

本项目围绕障碍物检测这一实际应用场景，基于 YOLOv8 构建了一套较完整的目标检测系统。系统覆盖数据集准备、标签转换、模型训练、推理测试、摄像头实时检测和性能评估等关键环节，并支持多组数据配置与实验脚本。项目目标是在保证检测精度的同时兼顾实时性，使系统能够对行人、车辆、交通标志等常见障碍物进行稳定识别，并为毕业设计答辩、实验复现和后续改进提供可用基础。

## English Abstract

This project presents an obstacle detection system based on YOLOv8 for a graduation thesis scenario. It covers the full workflow of dataset preparation, label conversion, model training, image and video inference, real-time camera detection, and requirement evaluation. The repository is designed to support experiment reproduction and future extension, with the goal of achieving both reliable detection accuracy and practical real-time performance for common road obstacles such as pedestrians, vehicles, and traffic signs.

## 项目简介

本项目以 YOLOv8 为核心，围绕障碍物检测场景搭建了一套较完整的实验流程：

- 数据集准备与标签转换
- 多版本数据配置与训练脚本
- 图片、视频、摄像头实时推理
- mAP、FPS、抗干扰能力评估

适合作为毕业设计展示、实验复现和后续功能扩展的基础仓库。

## 项目亮点

- 基于 YOLOv8 构建完整的障碍物检测实验流程
- 支持 COCO、BDD100K 以及自定义合并数据集配置
- 提供图片、视频、摄像头三种推理方式
- 包含 PyQt5 图形界面，便于毕设演示
- 提供达标评估脚本，可统一输出精度、速度和抗干扰结果
- 训练、过滤、合并、转换脚本相对齐，便于继续扩展类别和数据来源

## 检测目标

默认关注的障碍物类别包括：

- person
- bicycle
- motorcycle
- car
- bus
- truck
- traffic light
- stop sign

项目达标目标：

- `mAP50 >= 0.80`
- `FPS >= 15`
- 对光照变化、部分遮挡具备一定鲁棒性

## 仓库说明

本仓库公开的是项目代码、配置文件和使用说明，不包含以下内容：

- 数据集原始图片与标注
- 训练输出目录
- 模型权重文件，如 `*.pt`

如果你要复现实验，需要先自行准备数据集，并根据本地环境修改 `configs/` 下对应配置文件中的路径。

## 复现注意事项

- 本仓库不会上传数据集、训练输出和模型权重，避免仓库体积过大并尊重数据集分发规则。
- 克隆仓库后需要先安装依赖，再准备本地数据集路径。
- `configs/` 中的 `path` 字段需要根据你电脑上的实际数据目录修改。
- 如果使用 GUI 演示，请先准备训练好的 `best.pt` 权重，并在 `gui_main.py` 中确认默认权重路径是否存在。
- 实验指标会受到数据集版本、训练轮数、模型大小、GPU 性能和阈值设置影响，README 中的达标目标用于毕业设计实验参考。

## 项目结构

```text
.
├─ configs/                     数据集与训练配置
├─ scripts/                     数据准备、训练、推理、评估脚本
├─ gui_main.py                  图形界面主程序
├─ requirements.txt             Python 依赖
├─ train_*.py                   不同实验入口脚本
└─ README.md
```

## 目录说明

### `configs/`

用于维护不同实验设置下的数据集路径和类别配置：

- `configs/dataset.yaml`：主数据集配置
- `configs/dataset_coco_only.yaml`：偏 COCO 数据的训练配置
- `configs/dataset_merged.yaml`：多来源数据合并后的训练配置
- `configs/dataset_new_7cls.yaml`：7 类版本配置
- `configs/dataset_new_8cls.yaml`：8 类版本配置

### `scripts/`

用于承载数据处理、训练、推理和评估的核心脚本：

- `scripts/train.py`：通用训练入口，支持批量参数控制与数据增强
- `scripts/predict.py`：图片或视频推理脚本
- `scripts/realtime_detect.py`：实时摄像头检测脚本
- `scripts/evaluate_requirements.py`：输出 mAP、FPS、抗干扰表现的达标评估脚本
- `scripts/prepare_coco_obstacle.py`：将 COCO 中障碍物相关类别转换为 YOLO 标签
- `scripts/prepare_bdd100k_obstacle.py`：整理 BDD100K 数据集用于检测训练
- `scripts/merge_yolo_datasets.py`：合并多个 YOLO 格式数据集
- `scripts/merge_incoming.py`：处理新增样本并合并进现有数据集
- `scripts/filter_speed_bump_only.py`：针对减速带样本做筛选
- `scripts/filter_ood_for_obstacles.py`：筛选障碍物场景中的 OOD 样本
- `scripts/convert_retinanet_dataset2_to_yolo.py`：将特定数据格式转换为 YOLO 标注
- `scripts/convert_retinanet_to_yolo_speedbump.py`：将减速带相关标注转换为 YOLO 格式
- `scripts/prepare_speed_bump_raw.py`：预处理减速带原始数据

### 其他入口文件

- `gui_main.py`：PyQt5 图形界面，适合本地演示和毕设展示
- `train_fast.py`：偏快速训练的实验入口
- `train_m.py`：面向更大模型配置的训练入口
- `train_bdd_plus_coco.py`：BDD100K 与 COCO 混合训练入口
- `train_finetune_merged.py`：面向合并数据集的微调入口

## 环境安装

在 Windows PowerShell 中执行：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

安装完成后可执行：

```bash
yolo checks
```

## 数据集准备

### COCO2017

准备如下目录结构，例如 `D:\datasets\coco2017`：

- `train2017/`
- `val2017/`
- `annotations/instances_train2017.json`
- `annotations/instances_val2017.json`

转换障碍物类别：

```bash
python scripts/prepare_coco_obstacle.py --coco-root "D:\datasets\coco2017" --out-root "datasets/coco_obstacle"
```

### BDD100K 或合并数据集

如果使用 BDD100K 或自定义合并数据集，请同步检查以下配置文件中的 `path`：

- `configs/dataset.yaml`
- `configs/dataset_merged.yaml`
- `configs/dataset_new_7cls.yaml`
- `configs/dataset_new_8cls.yaml`

## 模型训练

GPU 训练示例：

```bash
python scripts/train.py --data configs/dataset.yaml --model yolov8n.pt --epochs 120 --imgsz 640 --batch 16 --device 0 --name yolov8n_obstacle_exp
```

CPU 训练示例：

```bash
python scripts/train.py --data configs/dataset.yaml --model yolov8n.pt --epochs 120 --imgsz 640 --batch 8 --device cpu --name yolov8n_obstacle_cpu
```

训练建议：

- 优先从 `yolov8n.pt` 开始，便于快速验证流程
- 可再尝试 `yolov8s.pt` 做精度对比
- 如果显存足够，可适当增大 `batch`

## 推理与实时检测

图片或视频推理：

```bash
python scripts/predict.py --model runs_obstacle/yolov8n_obstacle_exp/weights/best.pt --source "你的图片或视频路径" --conf 0.25 --imgsz 640 --device 0 --save
```

摄像头实时检测：

```bash
python scripts/realtime_detect.py --model runs_obstacle/yolov8n_obstacle_exp/weights/best.pt --source 0 --conf 0.25 --imgsz 640 --device 0
```

运行时：

- 按 `q` 或 `ESC` 退出
- 窗口左上角显示实时 FPS

## 达标评估

执行评估脚本：

```bash
python scripts/evaluate_requirements.py --model runs_obstacle/yolov8n_obstacle_exp/weights/best.pt --data configs/dataset.yaml --device 0 --imgsz 640
```

输出内容包括：

- `mAP50`
- `mAP50-95`
- `FPS`
- `low_light`、`high_light`、`occlusion` 三类扰动测试结果
- `ALL PASS: True/False`

默认报告文件：

```text
runs_obstacle/requirement_report.json
```

## 常见问题

`ModuleNotFoundError`

- 原因：虚拟环境未激活或依赖未安装
- 解决：重新执行环境安装步骤

`CUDA out of memory`

- 原因：显存不足
- 解决：减小 `batch` 或 `imgsz`

训练速度慢

- 检查是否使用了 `--device 0`
- 检查 GPU 是否正常工作

没有检测框

- 先将 `--conf` 降到 `0.15`
- 确认使用的是训练后的 `best.pt`

## 毕设报告可展示指标

- Precision
- Recall
- mAP50
- mAP50-95
- FPS
- 光照变化与部分遮挡下的鲁棒性对比

建议至少比较两个模型版本，例如 `yolov8n` 与 `yolov8s`。

## 开源许可

本项目采用 MIT License，详见 `LICENSE` 文件。代码可用于学习、研究和二次开发；数据集、预训练权重和第三方模型仍需遵守其各自的许可与使用条款。
