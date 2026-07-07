import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on obstacle dataset")
    parser.add_argument("--data", type=str, default="configs/dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs_obstacle")
    parser.add_argument("--name", type=str, default="yolov8n_coco_obstacle")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--hsv-h", type=float, default=0.02, help="Hue augmentation")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="Saturation augmentation")
    parser.add_argument("--hsv-v", type=float, default=0.5, help="Value augmentation")
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=2.0)
    parser.add_argument("--perspective", type=float, default=0.0005)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=0.8)
    parser.add_argument("--mixup", type=float, default=0.1)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
        workers=args.workers,
        patience=args.patience,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
    )


if __name__ == "__main__":
    main()
