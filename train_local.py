"""Training YOLO pothole di laptop RTX 4050 6GB (Windows).
Setting hemat VRAM: yolov8n, imgsz=640, batch=8, workers=4, amp=True, cache=False.
Usage:
  pip install -r requirements_local.txt   # pastikan torch CUDA, cek: python -c "import torch; print(torch.cuda.is_available())"
  python train_local.py --data datasets/pothole/data.yaml --epochs 50
  python train_local.py --data datasets/pothole/data.yaml --epochs 50 --resume runs/yolov8n_pothole_local/weights/last.pt
"""
import argparse
import pathlib
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/pothole/data.yaml")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="yolov8n_pothole_local")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    data = pathlib.Path(args.data)
    if not data.exists():
        raise FileNotFoundError(
            f"data.yaml tidak ketemu: {data}\n"
            "Download dataset publik (misal Kaggle varadpisale/pothole-detection-dataset-annotated-yolov8), "
            "extract ke datasets/pothole/ dengan struktur train/images, train/labels, valid/images, valid/labels."
        )
    model_src = args.resume or args.model
    print(f"load: {model_src} | data: {data} | epochs={args.epochs} batch={args.batch}")
    model = YOLO(model_src)
    resume = bool(args.resume)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=0,
        amp=True,
        cache=False,
        patience=15,
        optimizer="AdamW",
        lr0=0.001,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True,
        resume=resume,
    )
    print("Validasi best.pt ...")
    best = pathlib.Path(args.project) / args.name / "weights" / "best.pt"
    m = YOLO(str(best) if best.exists() else args.model)
    print(m.val(data=str(data), imgsz=args.imgsz, device=0))


if __name__ == "__main__":
    main()
