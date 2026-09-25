"""Inference cepat ke video lokal (2 mp4 kamu) pakai best.pt hasil training.
Usage:
  python predict_local.py --weights runs/yolov8n_pothole_local/weights/best.pt --source "jalan berlubang.mp4"
"""
import argparse
import pathlib
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="runs/yolov8n_pothole_local/weights/best.pt")
    p.add_argument("--source", default="jalan berlubang.mp4")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="pred_local")
    return p.parse_args()


def main():
    args = parse_args()
    if not pathlib.Path(args.weights).exists():
        raise FileNotFoundError(f"weights tidak ketemu: {args.weights} — training dulu.")
    if not pathlib.Path(args.source).exists():
        raise FileNotFoundError(f"video tidak ketemu: {args.source}")
    m = YOLO(args.weights)
    m.predict(
        source=args.source, imgsz=640, conf=args.conf, iou=0.5, device=0,
        save=True, save_txt=True, project=args.project, name=args.name, exist_ok=True,
    )
    print(f"selesai -> {args.project}/{args.name}")


if __name__ == "__main__":
    main()
