# YOLOv8 障碍物检测系统设计

基于 YOLOv8 的毕业设计项目，面向障碍物实时检测任务，包含数据准备、模型训练、推理测试、实时摄像头检测，以及达标评估脚本。

## 项目简介

本项目以 YOLOv8 为核心，围绕障碍物检测场景搭建了一套较完整的实验流程：

- 数据集准备与标签转换
- 多版本数据配置与训练脚本
- 图片、视频、摄像头实时推理
- mAP、FPS、抗干扰能力评估

适合作为毕业设计展示、实验复现和后续功能扩展的基础仓库。

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

主要文件说明：

- `configs/dataset.yaml`：主数据集配置
- `configs/dataset_*.yaml`：不同数据集版本配置
- `scripts/prepare_coco_obstacle.py`：COCO 转 YOLO 标签
- `scripts/prepare_bdd100k_obstacle.py`：BDD100K 数据准备
- `scripts/train.py`：训练入口
- `scripts/predict.py`：图片/视频推理
- `scripts/realtime_detect.py`：摄像头实时检测
- `scripts/evaluate_requirements.py`：达标评估脚本

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
