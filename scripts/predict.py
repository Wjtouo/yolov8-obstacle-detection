import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--source", type=str, required=True, help="Image/video/folder path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
    )


if __name__ == "__main__":
    main()
