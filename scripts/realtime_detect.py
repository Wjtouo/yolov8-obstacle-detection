import argparse
import time

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Real-time obstacle detection with YOLOv8")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--window", type=str, default="YOLOv8 Real-time Obstacle Detection")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    model = YOLO(args.model)

    prev = time.perf_counter()
    ema_fps = 0.0
    alpha = 0.12

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        vis = results[0].plot()

        now = time.perf_counter()
        dt = max(now - prev, 1e-6)
        inst_fps = 1.0 / dt
        ema_fps = inst_fps if ema_fps == 0.0 else (alpha * inst_fps + (1 - alpha) * ema_fps)
        prev = now

        cv2.putText(
            vis,
            f"FPS: {ema_fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(args.window, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
