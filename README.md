# FabLoop — standalone EfficientAD baseline

Pipeline này huấn luyện **một model EfficientAD riêng cho từng `pcb1`–`pcb4`** bằng implementation chính thức của Anomalib 2.6.0. Teacher PDN pretrained được nạp bởi Anomalib, đóng băng hoàn toàn và được kiểm tra checksum; chỉ Student và Autoencoder nằm trong optimizer. Pipeline độc lập với `train.py` của RF-DETR.

## Chuẩn bị

Môi trường hiện tại là **Windows / PowerShell, `.venv`, Python 3.12**. Chạy tại thư mục repository:

```powershell
. .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
python scripts/check_environment.py
```

Config mặc định dùng CSV đã có tại `third-party/spot-diff/split_csv/1cls.csv`. Các thư mục `data/visa`, `data/mvtec_ad` và `data/mvtec_loco` hiện chưa có; kiểm tra môi trường sẽ báo `BLOCKED_DATA`. Hai thư mục PCBA hiện có cần adapter và quy tắc nhãn/split riêng, chưa thể dùng trực tiếp với config VisA. Xem [báo cáo môi trường](docs/preflight/README.md).

## Photometric stereo và PCBA 4 đèn

Đã merge phần photometric stereo từ thư mục con `FabLoop` vào `src/fabloop/photometric_stereo/`, chuyển baseline vào `references/`, giữ `.venv` và requirements chính. Dependency nghiên cứu nằm tại `third-party/RobustPhotometricStereo`. [Quy trình DiLiGenT và phân tích review](docs/photometric_stereo_diligent.md) phân biệt kiểm định có ground truth với [xử lý PCBA](docs/photometric_stereo_pcba.md).

Đã kiểm kê **34 nhóm / 170 ảnh edited và 170 ảnh gốc**: đọc được toàn bộ, nhưng **33 nhóm edited lệch kích thước giữa bốn đèn**. Hiện chưa có nhãn normal/anomaly, split, mask và calibration đèn. Manifest ở `data/processed/pcba_4light/inventory.json`; điền thông tin đã xác minh vào `data/processed/pcba_4light/labels.json` (không bị ghi đè khi chạy lại).

```powershell
.\.venv\Scripts\python.exe scripts\prepare_pcba_4light.py
.\.venv\Scripts\python.exe scripts\validate_diligent.py --help
.\.venv\Scripts\python.exe scripts\process_pcba_photometric.py --help
```

`prepare_pcba_4light.py --require-ready` hiện trả lỗi theo [báo cáo](docs/preflight/pcba_4light_readiness.json). Bridge PCBA nhận capture đã căn chỉnh/hiệu chuẩn theo [config mẫu](configs/pcba_photometric_capture.example.json), xuất normal/height tương đối mà không cần normal ground truth. Không tự suy ra hướng đèn, resize để giả lập registration hoặc gán toàn bộ board là normal. Chưa tạo tập train PCBA hoặc thay routing VisA/MVTec.

Để xem **bản thử normal map và pseudo-3D từ ảnh edited khi chưa có calibration**, dùng lệnh riêng sau. Nó ước lượng căn chỉnh, dùng hướng đèn danh định và ghi rõ mọi giả định; output giữ `calibrated=false`, `registration_verified=false`, `training_ready=false`.

```powershell
.\.venv\Scripts\python.exe scripts\export_pcba_preview.py --nominal-lights --boards PCB1 --output-root outputs/pcba_preview_PCB1
.\.venv\Scripts\python.exe scripts\export_pcba_preview.py --nominal-lights --output-root outputs/pcba_photometric_preview
```

Lệnh đầu xuất một board để xem trước; lệnh sau xử lý toàn bộ `data/PCBA_4Light_edited`. Xem `normal_rgb.png`, `height_preview.png`, `pseudo3d.png` và `report.json` trong thư mục từng board. `summary.json` ghi các board thất bại. Hướng dẫn tham số và giới hạn nằm trong [quy trình PCBA](docs/photometric_stereo_pcba.md).

Bản gốc đầy đủ của thư mục con, gồm lịch sử Git, được lưu tại `archives/FabLoop_before_merge_2026-09-08.zip`; [manifest migration](docs/migration/FabLoop_migration_manifest.json) ghi hash và mapping file. Tài liệu `docs/migration/FabLoop_original_*` chỉ lưu lịch sử; dùng các lệnh trong README chính.

Thư mục `FabLoop` không còn ở gốc dự án: bản gốc được chuyển nguyên vẹn vào `archives/FabLoop_retired` vì công cụ chặn lệnh xóa đệ quy (`blocked by policy`). Đã xác minh hash cả bản ZIP và 42 file được chuyển. [Kiểm tra migration](docs/preflight/photometric_migration_validation.json) ghi nhận **40 tests PASS**, pip check sạch và smoke test L2/L1 tổng hợp tại thời điểm migration. Chưa chạy DiLiGenT thật hoặc training.

## Chuẩn bị EfficientAD-S slim

Đã hoàn thành Bước 2 với **Slim-0.5 candidate** trong `src/models/efficientad_slim/slim_model.py`: Student `3 → 64 → 128 → 128 → 768`; AE encoder `3 → 16 → 16 → 32 → 32 → 32 → 32`, decoder dùng 32 channels nội bộ trước output 384. Teacher giữ nguyên; đầu ra vẫn **Teacher 384, Student 768 = 384 + 384, AE 384**. Số layer, geometry, activation, dropout, loss và anomaly-map logic giữ nguyên từ snapshot Anomalib 2.6.0. `torch_model.py`, `SOURCE.json` và source trong `site-packages` không đổi.

Kiểm tra candidate và đo baseline/slim bằng cùng tensor tổng hợp, không cần dataset hoặc checkpoint pretrained:

```powershell
python -m unittest discover -s tests -p test_efficientad_slim.py -v
python scripts/profile_efficientad_slim.py
```

Bảy test đã đạt, gồm shape/geometry, loss backward, Teacher frozen, map parity và strict state-dict roundtrip; không có optimizer step. Với input CPU float32 `1×3×256×256`, Student còn **1.855.552 params / 7,625 GFLOPs được fvcore hỗ trợ**, giảm **56,52% / 62,33%**, đạt mốc Student ≥40% / ≥30%. Tổng **EfficientAD core (Teacher + Student + AE)** còn 4.879.936 params / 23,717 GFLOPs, giảm 39,44% / 37,51%. Tổng này chỉ tính forward của ba mạng feature, chưa gồm anomaly-map postprocessing, SAM3 hoặc 772 phần tử calibration; operator coverage và quy ước một multiply-add = một FLOP nằm trong [kết quả profile](docs/preflight/efficientad_slim_05_profile.json).

Slim-0.5 chưa phải kiến trúc cuối cùng. Các kiểm tra dùng trọng số khởi tạo ngẫu nhiên, chưa nạp Teacher pretrained, chưa huấn luyện hoặc đo chất lượng/latency. Runtime `src/model.py`, trainer và config vẫn dùng baseline. Xem [cách khởi tạo candidate độc lập](src/models/efficientad_slim/README.md) và [các phần tích hợp còn lại trước khi train](docs/preflight/efficientad_slim_readiness.md).

## Bước 3 — Shape gate bắt buộc trước training

Chạy gate riêng cho Slim-0.5, không cần dataset hay checkpoint pretrained:

```powershell
python scripts/check_efficientad_shapes.py
python scripts/check_efficientad_shapes.py --device cuda:0 --output docs/preflight/efficientad_slim_05_shapes_cuda.json
```

Với dummy `[1,3,256,256]` và `padding=false`, CPU và CUDA đều PASS:

```text
Teacher       [1,384,56,56]
Slim Student  [1,768,56,56]
Student[:384] [1,384,56,56]
Student[384:] [1,384,56,56]
Slim AE       [1,384,56,56]
T - S_T       [1,384,56,56] PASS
T - A         [1,384,56,56] PASS
A - S_A       [1,384,56,56] PASS
```

Gate yêu cầu shape bằng nhau chính xác trước khi trừ, không chấp nhận broadcasting. `src/trainer.py` gọi gate bắt buộc trước nạp Teacher pretrained, chuẩn bị Imagenette, thống kê feature và tạo optimizer. Kết quả mỗi lần chuẩn bị train được ghi vào `checkpoints/shape_check_seed_<seed>.json`; lỗi shape ghi FAIL rồi dừng. Gate bảo toàn state, mode và RNG; không sửa loss.

[Báo cáo CPU](docs/preflight/efficientad_slim_05_shapes.json) và [báo cáo CUDA](docs/preflight/efficientad_slim_05_shapes_cuda.json) chỉ xác nhận tính tương thích shape, chưa phải bằng chứng chất lượng sau train. Factory training hiện vẫn chọn baseline; khi nối Slim vào factory, cùng gate sẽ kiểm tra core đó. Kiểm tra toàn bộ tests, gồm cả compression gate ở Bước 4:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Bước 4 — Params và FLOPs trước training

Đã đo lại baseline và Slim-0.5 bằng **cùng tensor `[1,3,256,256]`**, CPU float32, eval mode, `padding=false`, với fvcore trong `.venv`. Một multiply-add được tính là một FLOP; GFLOPs = FLOPs / 10⁹.

| Thành phần | Params gốc | Params Slim | Giảm Params | GFLOPs gốc | GFLOPs Slim | Giảm FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Student | 4.267.392 | 1.855.552 | **56,52%** | 20,2434 | 7,6254 | **62,33%** |
| Autoencoder | 1.096.320 | 330.240 | 69,88% | 2,3993 | 0,7811 | 67,44% |
| Teacher + Student + AE | 8.057.856 | 4.879.936 | 39,44% | 37,9536 | 23,7174 | 37,51% |

Teacher giữ nguyên **2.694.144 params / 15,3109 GFLOPs**. **Student PASS cả hai mục tiêu ≥40% Params và ≥30% FLOPs**, tính bằng tỷ lệ chưa làm tròn; chưa cần sửa architecture để đạt hai mục tiêu này. AE và tổng ba mạng chỉ báo cáo mức giảm thực tế.

```powershell
.\.venv\Scripts\python.exe scripts\profile_efficientad_slim.py
```

Lệnh lưu [báo cáo đầy đủ](docs/preflight/efficientad_slim_05_profile.json), trả exit code **0 khi PASS**, **1 nếu một trong hai mục tiêu Student thất bại**. Khi FAIL, phải sửa architecture trước train. PASS cho phép chuyển sang bước tích hợp training; lệnh này không chạy train, không chọn Slim trong factory và không thay thế các gate Teacher/dữ liệu.

**19 tests đã PASS**, gồm 5 tests mới cho ngưỡng Student và exit code: đạt đúng ngưỡng được PASS; thiếu một ngưỡng vẫn FAIL dù tổng pipeline giảm mạnh; cả PASS/FAIL đều lưu báo cáo. Xem [kết quả kiểm tra gate](docs/preflight/efficientad_compression_gate_validation.json).

FLOPs trong bảng là **các operator được fvcore hỗ trợ** ở forward của ba mạng feature. `avg_pool2d`, phép trừ/chia chuẩn hóa chưa được bộ đếm mặc định tính; các operator bỏ qua và breakdown từng module đều được ghi trong JSON. Tổng ba mạng chưa gồm anomaly-map postprocessing, SAM3 và 772 phần tử calibration; tính cả calibration thì Params gốc/Slim là 8.058.628 / 4.880.708. Đây chưa phải phép đo latency hoặc chất lượng sau train.

Không dùng test để tạo validation, tính quantile hoặc chọn threshold. Với VisA và MVTec AD, `data.calibration_ratio` giữ lại một phần deterministic của **official train-normal** làm validation-normal. MVTec LOCO không dùng ratio mà lấy nguyên official `validation/good`; official test rows luôn được giữ nguyên.

## Chạy từng bước

Ví dụ cho `pcb1`, seed 42:

```powershell
. .\.venv\Scripts\Activate.ps1
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
