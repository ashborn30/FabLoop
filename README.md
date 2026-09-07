# FabLoop — software-only baselines

Pipeline này huấn luyện **một model EfficientAD riêng cho từng `pcb1`–`pcb4`** bằng implementation chính thức của Anomalib 2.6.0. Teacher PDN pretrained được nạp bởi Anomalib, đóng băng hoàn toàn và được kiểm tra checksum; chỉ Student và Autoencoder nằm trong optimizer. Pipeline độc lập với `train.py` của RF-DETR.

## Photometric stereo validation (Người 1)

Nhánh hình học photometric stereo được tách khỏi anomaly detection. Script DiLiGenT adapter nằm ở `src/fabloop/photometric_stereo/diligent_validate.py` và dùng `yasumat/RobustPhotometricStereo` như thư viện ngoài, không copy lại thuật toán vào repo này.

Chuẩn bị dependency nghiên cứu và dữ liệu:

```bash
pip install -r requirements.txt
git clone https://github.com/yasumat/RobustPhotometricStereo.git third_party/RobustPhotometricStereo
```

Đặt một object DiLiGenT dưới `data/diligent/<object>/` sao cho thư mục có `mask.png`, `light_directions.txt`, `Normal_gt.mat`, và ảnh quan sát. Nếu dataset có `filenames.txt`/`light_intensities.txt`, script sẽ tự dùng để đọc đúng thứ tự và chuẩn hoá cường độ đèn.

Chạy test sát điều kiện hộp 4 đèn:

```bash
PYTHONPATH=src python -m fabloop.photometric_stereo.diligent_validate \
  --object-dir data/diligent/BallPNG \
  --rps-root third_party/RobustPhotometricStereo \
  --image-counts 4 \
  --solvers l2 l1
```

Chạy full 96 ảnh với L2 để đối chiếu baseline công bố của DiLiGenT:

```bash
PYTHONPATH=src python -m fabloop.photometric_stereo.diligent_validate \
  --object-dir data/diligent/BallPNG \
  --rps-root third_party/RobustPhotometricStereo \
  --image-counts all \
  --solvers l2
```

Output nằm mặc định ở `outputs/photometric_stereo/<object>/`, gồm normal-map RGB, normal-map GT, angular-error heatmap, height-map Frankot-Chellappa, mesh PyVista nếu môi trường render hỗ trợ, và `summary.csv`/`summary.json`. Bảng baseline L2 96 ảnh được lưu ở `references/diligent_main_l2_baseline.csv`; script chỉ attach số baseline khi object/solver/số ảnh khớp, không tự đặt ngưỡng pass/fail.

Chi tiết workflow ở `docs/photometric_stereo_diligent.md`.

## Chuẩn bị

Chạy trong WSL tại thư mục repository:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Config mặc định yêu cầu official one-class CSV đúng tại `third-party/spot-diff/split_csv/1cls.csv`. Repository hiện có một bản CSV do Anomalib chuẩn bị ở `data/visa/split_csv/1cls.csv`; chỉ copy sau khi đã xác minh đó là file official mong muốn:

```bash
if [ ! -f third-party/spot-diff/split_csv/1cls.csv ]; then
  mkdir -p third-party/spot-diff/split_csv
  cp data/visa/split_csv/1cls.csv third-party/spot-diff/split_csv/1cls.csv
fi
```

Không dùng test để tạo validation, tính quantile hoặc chọn threshold. Với VisA và MVTec AD, `data.calibration_ratio` giữ lại một phần deterministic của **official train-normal** làm validation-normal. MVTec LOCO không dùng ratio mà lấy nguyên official `validation/good`; official test rows luôn được giữ nguyên.

## Chạy từng bước

Ví dụ cho `pcb1`, seed 42:

```bash
source venv/bin/activate
python src/anomaly.py preflight --config configs/efficientad.yaml --category pcb1 --seed 42
python src/anomaly.py train --config configs/efficientad.yaml --category pcb1 --seed 42
python src/anomaly.py calibrate --config configs/efficientad.yaml --category pcb1 --seed 42
python src/anomaly.py evaluate --config configs/efficientad.yaml --category pcb1 --seed 42
python src/anomaly.py benchmark --config configs/efficientad.yaml --category pcb1 --seed 42
python src/anomaly.py validate --config configs/efficientad.yaml --category pcb1 --seed 42
python src/anomaly.py report --config configs/efficientad.yaml
```

Chạy trọn pipeline cho cả bốn category, tuần tự và vẫn tạo bốn model riêng:

```bash
python src/anomaly.py run --config configs/efficientad.yaml --category all --seed 42
```

Trong lúc train, terminal hiển thị category, seed, iteration, ETA, local loss, AE loss, STAE loss, total loss và learning rate. Tần suất cập nhật loss được điều khiển bởi `training.progress_refresh_interval`; mỗi periodic checkpoint cũng được thông báo trên terminal.

Tiến trình đã khởi chạy trước khi cập nhật tính năng này sẽ không tự nạp code mới. Có thể theo dõi CSV hiện tại từ terminal khác mà không dừng train:

```bash
tail -f outputs/efficientad/pcb1/checkpoints/losses_seed_43.csv
```

Hoặc theo dõi iteration mới nhất của cả bốn category trong seed 43:

```bash
watch -n 5 'for category in pcb1 pcb2 pcb3 pcb4; do file="outputs/efficientad/$category/checkpoints/losses_seed_43.csv"; if [ -f "$file" ]; then printf "%s: " "$category"; tail -n 1 "$file"; fi; done'
```

## MVTec diagnostic configs

`configs/transistor.yaml`, `configs/pushpins.yaml` và `configs/splicing_connectors.yaml` dùng trực tiếp datamodule `MVTecAD`/`MVTecLOCO` của Anomalib. Các dataset này đọc official directory split, không đọc VisA `1cls.csv`. Transistor tách calibration deterministic từ official train-normal; pushpins và splicing_connectors dùng nguyên official LOCO validation-normal.

Luôn chạy preflight trước. Mỗi config chỉ có seed diagnostic 42 và ghi vào cây `outputs/efficientad_diagnostic/<category>` riêng, không chạm baseline VisA:

```bash
python src/anomaly.py preflight --config configs/transistor.yaml --seed 42
python src/anomaly.py preflight --config configs/pushpins.yaml --seed 42
python src/anomaly.py preflight --config configs/splicing_connectors.yaml --seed 42

python src/anomaly.py run --config configs/transistor.yaml --seed 42
python src/anomaly.py run --config configs/pushpins.yaml --seed 42
python src/anomaly.py run --config configs/splicing_connectors.yaml --seed 42
```

Nếu checkpoint đã train và calibration xong, chỉ chạy lại evaluator LOCO bằng entrypoint duy nhất:

```bash
python src/anomaly.py evaluate --config configs/pushpins.yaml --seed 42
python src/anomaly.py evaluate --config configs/splicing_connectors.yaml --seed 42
```

Không dùng `--category all` với config MVTec. Nếu preflight LOCO báo không tìm thấy category, hoàn tất giải nén dataset để `data/mvtec_loco/` chứa trực tiếp `pushpins/` và `splicing_connectors/` trước khi train.

Kiểm tra contract của official split:

```bash
PYTHONPATH=src python -m unittest \
  tests.test_efficientad_pipeline \
  tests.test_evaluator_policies -v
```

Lặp `train → calibrate → evaluate → benchmark → validate` với các giá trị `--seed` khác để báo cáo độ lệch chuẩn qua seed. Có thể override thiết bị bằng `--device cuda:0` hoặc `--device cpu`.

Nếu checkpoint đã train từ trước nhưng calibration chưa có phân phối image score, không cần train lại. Backfill calibration và evaluation; `pcb2` được chạy trước vì recall default thấp nhất:

```bash
for category in pcb2 pcb1 pcb3 pcb4; do
  python src/anomaly.py calibrate --config configs/efficientad.yaml --category "$category" --seed 42
  python src/anomaly.py evaluate --config configs/efficientad.yaml --category "$category" --seed 42
done
python src/anomaly.py report --config configs/efficientad.yaml
```

Backfill riêng evaluator cho seed 44 khi calibration schema 2 đã có đủ score. `target_fpr: 0.10` được đọc từ config nên CLI không nhận thêm `--target_fpr`; vẫn giữ `src/anomaly.py` là entrypoint duy nhất:

```bash
grep -A4 '^evaluation:' configs/efficientad.yaml
for category in pcb2 pcb1 pcb3 pcb4; do
  python src/anomaly.py evaluate --config configs/efficientad.yaml --category "$category" --seed 44
done
python src/anomaly.py validate --config configs/efficientad.yaml --category all --seed 44
python src/anomaly.py report --config configs/efficientad.yaml
```

Lệnh `report` luôn nạp lại `reporter.py` tại thời điểm tổng hợp. Vì vậy một lệnh `run` kéo dài không thể ghi đè `summary.csv` bằng reporter đã được import trước khi source được cập nhật.

## Output và reproduction gate

Mỗi run lưu checkpoint tại `outputs/efficientad/<category>/checkpoints/seed_<seed>.ckpt`, calibration tại `calibration/seed_<seed>.json`, raw local/global/final maps cùng metrics tại `predictions/seed_<seed>/`, và PNG input/mask/maps/overlay tại `visualizations/seed_<seed>/`. Báo cáo category nằm ở `report.json`; bảng tổng hợp và gate nằm ở:

```text
outputs/efficientad/summary.csv
outputs/efficientad/reproduction_gate.json
```

Trong `summary.csv`, `image_level_auroc_std` của từng hàng `category_aggregate` là độ lệch chuẩn qua các seed của category đó (`across_seeds`); ở hàng `overall`, đây là độ lệch chuẩn giữa bốn mean-category (`across_category_means`). Reproduction gate vẫn so sánh ngưỡng 0.02 với `max_category_seed_std`, không dùng độ lệch chuẩn của hàng `overall`.

`metrics.json` giữ nguyên nhánh default qua `recall_default`, `false_positive_rate_default`, `threshold_default` và các alias cũ. Nhánh benchmark dùng toàn bộ test labels để báo cáo operating point tại target FPR 10%; threshold này chỉ dùng phân tích, không dùng deployment. Với MVTec LOCO, cùng một threshold gộp này được dùng để báo thêm `recall_at_fpr_10_logical_anomalies` và `recall_at_fpr_10_structural_anomalies` (kèm TP/FN), tuyệt đối không chọn threshold riêng cho từng subset. Nhánh deployment khóa threshold từ percentile 90% của calibration-normal scores, hoàn toàn không dùng test để chọn threshold.

Các ca sai tại benchmark operating point được lồng theo seed:

```text
outputs/efficientad/<category>/predictions/seed_<seed>/false_negatives_at_fpr_10.jsonl
outputs/efficientad/<category>/predictions/seed_<seed>/false_positives_at_fpr_10.jsonl
outputs/efficientad/<category>/predictions/seed_<seed>/operating_point_at_fpr_10.json
outputs/efficientad/<category>/predictions/seed_<seed>/deployment_threshold.json
```

Gate chỉ PASS khi đủ `pcb1`–`pcb4` và đủ seed `42/43/44`, mean image AUROC ≥ 0.90, không category dưới 0.85, độ lệch chuẩn seed ≤ 0.02, Teacher checksum giữ nguyên, calibration chỉ dùng validation-normal, toàn bộ validator PASS và review anomaly maps đã được xác nhận. Metric benchmark/deployment mới không tham gia reproduction gate. Sau khi review các overlay, đặt `reproduction_gate.visual_review_pass: true` rồi chạy lại lệnh `report`. Mốc 2 ms / 600 ảnh/giây của paper chỉ được ghi làm tham chiếu workstation/GPU hiện đại, không phải yêu cầu cho Jetson.
