# datasets/ — tidak di-push ke Git (lihat .gitignore)
# Download dataset publik lalu extract ke datasets/pothole/ :
#
# Rekomendasi (single-class, ringan):
#   https://www.kaggle.com/datasets/varadpisale/pothole-detection-dataset-annotated-yolov8
#   8236 train / 1327 valid / 559 test, 1 kelas: pothole
#
# Alternatif multi-negara (11GB, 5 kelas):
#   https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022
#
# Struktur akhir yang diharapkan:
# datasets/pothole/
#   train/images/*.jpg  train/labels/*.txt
#   valid/images/*.jpg  valid/labels/*.txt
#   test/images/*.jpg   test/labels/*.txt (opsional)
#   data.yaml
