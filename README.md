# YOLOv8 毕设项目：障碍物实时检测

## 仓库说明

本仓库公开的是项目代码、配置文件和使用说明，不包含以下内容：

- 数据集原始图片与标注
- 训练输出与实验结果目录
- 模型权重文件（如 `*.pt`）

如果你要复现实验，请先自行准备数据集，并根据本地路径修改 `configs/` 下的数据配置文件。

## 1. 项目目标

本项目基于 YOLOv 系列（默认 YOLOv8）实现障碍物实时检测，覆盖常见障碍物类型，并提供完整达标验证流程：

- 检测类别：行人、车辆、固定障碍（红绿灯、停止标志等）
- 指标目标：
  - mAP50 >= 0.80
  - FPS >= 15（一般 GPU 环境）
  - 对光照变化、部分遮挡具备一定鲁棒性

默认障碍物类别：
- person
- bicycle
- motorcycle
- car
- bus
- truck
- traffic light
- stop sign

---

## 2. 项目结构

- `configs/dataset.yaml`：YOLO 数据配置
- `configs/dataset_*.yaml`：不同数据集版本的训练配置
- `scripts/prepare_coco_obstacle.py`：COCO 转 YOLO 标签脚本
- `scripts/prepare_bdd100k_obstacle.py`：BDD100K 障碍物数据准备脚本
- `scripts/train.py`：训练脚本
- `scripts/predict.py`：推理脚本
- `scripts/realtime_detect.py`：摄像头/视频实时检测（显示 FPS）
- `scripts/evaluate_requirements.py`：一键评估 mAP/FPS/抗干扰并输出报告
- `requirements.txt`：依赖

---

## 3. 环境安装（Windows PowerShell）

在项目根目录执行：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

验证：

```bash
yolo checks
```

---

## 4. 下载 COCO2017

你需要准备以下内容到同一个 COCO 根目录，例如 `D:\datasets\coco2017`：

- `train2017/`（训练图片）
- `val2017/`（验证图片）
- `annotations/instances_train2017.json`
- `annotations/instances_val2017.json`

如果使用 BDD100K 或合并数据集训练，请同步检查 `configs/dataset.yaml`、`configs/dataset_merged.yaml` 等文件中的 `path` 是否与你本机的数据目录一致。

---

## 5. 转换数据集（只保留障碍物类别）

```bash
python scripts/prepare_coco_obstacle.py --coco-root "D:\datasets\coco2017" --out-root "datasets/coco_obstacle"
```

成功后，数据会在：
- `datasets/coco_obstacle/images/train`
- `datasets/coco_obstacle/images/val`
- `datasets/coco_obstacle/labels/train`
- `datasets/coco_obstacle/labels/val`

---

## 6. 开始训练

GPU 训练：

```bash
python scripts/train.py --data configs/dataset.yaml --model yolov8n.pt --epochs 120 --imgsz 640 --batch 16 --device 0 --name yolov8n_obstacle_exp
```

CPU 训练（慢很多）：

```bash
python scripts/train.py --data configs/dataset.yaml --model yolov8n.pt --epochs 120 --imgsz 640 --batch 8 --device cpu --name yolov8n_obstacle_cpu
```

结果目录默认在：
- `runs_obstacle/yolov8n_coco_obstacle/`
- 最优权重：`weights/best.pt`

---

建议：
- 先跑 `yolov8n.pt`（速度优先），再尝试 `yolov8s.pt`（精度优先）做对比
- 若显存足够可尝试 `--batch 24` 或更大提升收敛稳定性

---

## 7. 推理测试（图片/视频）

```bash
python scripts/predict.py --model runs_obstacle/yolov8n_obstacle_exp/weights/best.pt --source "你的图片或视频路径" --conf 0.25 --imgsz 640 --device 0 --save
```

---

## 8. 实时检测（摄像头）

```bash
python scripts/realtime_detect.py --model runs_obstacle/yolov8n_obstacle_exp/weights/best.pt --source 0 --conf 0.25 --imgsz 640 --device 0
```

- 按 `q` 或 `ESC` 退出
- 窗口左上角显示实时 FPS

---

## 9. 一键评估是否达标（mAP/FPS/抗干扰）

```bash
python scripts/evaluate_requirements.py --model runs_obstacle/yolov8n_obstacle_exp/weights/best.pt --data configs/dataset.yaml --device 0 --imgsz 640
```

执行后会输出：
- 基础指标：`mAP50`、`mAP50-95`、`FPS`
- 抗干扰测试：`low_light`、`high_light`、`occlusion` 三种扰动下的 mAP
- 结论：`ALL PASS: True/False`
- 报告文件：`runs_obstacle/requirement_report.json`

默认判定阈值：
- `mAP50 >= 0.80`
- `FPS >= 15`
- 抗干扰：最大 mAP50 下降 `<= 0.12`

---

## 10. 常见问题（新手必看）

1) `ModuleNotFoundError`
- 原因：没激活虚拟环境或没安装依赖
- 解决：重新执行第 3 步

2) `CUDA out of memory`
- 原因：显存不够
- 解决：降低 `--batch`（如 16 -> 8 -> 4）或减小 `--imgsz`（640 -> 512）

3) 训练速度慢
- 检查是否用了 `--device 0`（GPU）
- 检查 `nvidia-smi` 是否有显存占用变化

4) 没有检测框
- 先把 `--conf` 降到 `0.15`
- 确认使用的是 `best.pt` 而不是随机模型

---

## 11. 毕设报告建议最少指标

- Precision
- Recall
- mAP50
- mAP50-95
- FPS
- 扰动鲁棒性对比（光照变化、部分遮挡）

建议至少比较两个模型：`yolov8n` 和 `yolov8s`。
