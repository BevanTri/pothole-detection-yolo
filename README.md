# AI Pothole Detection & Mapping

![Python](https://img.shields.io/badge/python-3.14-blue) ![YOLO](https://img.shields.io/badge/YOLO-v8n-green) ![GPU](https://img.shields.io/badge/GPU-RTX%204050%206GB-76B900) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

Sistem computer vision untuk **mendeteksi jalan berlubang dari video** (MP4/YouTube) memakai YOLO + object tracking, siap dikembangkan jadi web app dengan peta lokasi.

> Train di dataset publik beranotasi, inference bisa ke video apapun — termasuk video jalan Indonesia milik sendiri. Video YouTube umumnya tanpa GPS, jadi pemetaan butuh koordinat eksternal (bbox ≠ GPS).

## Fitur

- Training YOLOv8n hemat VRAM (RTX 4050 6GB / Colab T4)
- Inference video → bounding box + jumlah temuan + bukti frame
- Script siap pakai: `train_local.py`, `predict_local.py`
- Notebook Colab T4: `colab_T4_train_pothole.ipynb`
- Struktur dataset YOLO standar + `early stopping`

## Struktur

```text
.
├── train_local.py            # training laptop (RTX 4050)
├── predict_local.py          # inference ke mp4 lokal
├── colab_T4_train_pothole.ipynb  # alternatif training Colab T4
├── data.yaml                 # template config (colab path)
├── datasets/
│   ├── README.md             # cara download dataset
│   └── pothole/data.yaml     # config lokal (path relatif)
├── requirements.txt
└── runs/                     # output training (di-ignore git)
```

## Quickstart (laptop RTX)

```powershell
# 1. Torch CUDA (wajib, cek: harus True)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2. Dependensi
pip install -r requirements.txt

# 3. Dataset — download lalu extract ke datasets/pothole/
# Rekomendasi: https://www.kaggle.com/datasets/varadpisale/pothole-detection-dataset-annotated-yolov8
# Harus ada: train/images, train/labels, valid/images, valid/labels

# 4. Training (50 = max, early stopping patience=15)
python train_local.py --epochs 50

# 5. Inference ke video sendiri
python predict_local.py --source "jalan berlubang.mp4"
# output: runs/pred_local/
```

Setting hemat 6GB: `yolov8n`, `imgsz=640`, `batch=8`, `workers=4`, `amp=True`, `cache=False`.
Kalau OOM: `--batch 4 --imgsz 512`. Kalau Colab disconnect / laptop panas: resume dari `runs/.../weights/last.pt` via `--resume`.

## Dataset

| Sumber | Ukuran | Kelas | Cocok untuk |
|---|---|---|---|
| [Pothole Annotated YOLOv8 (Kaggle)](https://www.kaggle.com/datasets/varadpisale/pothole-detection-dataset-annotated-yolov8) | 8k train | `pothole` | Mulai cepat, ringan |
| [RDD2022 (Kaggle)](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022) | 11GB | 5 kelas kerusakan | Generalisasi multi-negara |
| [Pothole RDD2022 (Roboflow)](https://universe.roboflow.com/yolov8-study/pothole-detection-rdd2022) | bervariasi | pothole + retak | Alternatif Roboflow |

`data.yaml` minimal:

```yaml
path: datasets/pothole
train: train/images
val: valid/images
names:
  0: pothole
```

## Colab T4

Upload `colab_T4_train_pothole.ipynb` ke Colab → Runtime T4 → ikuti cell 0-8 (cek GPU → train → val mAP → predict → simpan `best.pt` ke Drive).

## Batasan

- Akurasi tergantung kualitas video, cahaya, sudut kamera, dan domain dataset.
- Mendeteksi lubang yang terlihat, bukan mengukur kedalaman.
- Peta butuh GPS valid dari metadata/sumber eksternal.
- Hasil AI perlu verifikasi sebelum jadi laporan resmi.

## Roadmap

- [x] Training + inference video lokal
- [ ] Tracking (ByteTrack) + counting-line anti double-hitung
- [ ] Backend FastAPI + frontend upload video
- [ ] Peta Leaflet + tabel `videos/detections/locations`
- [ ] Fine-tune 100-300 frame lokal bila domain shift

## Lisensi

MIT — lihat [LICENSE](LICENSE).
