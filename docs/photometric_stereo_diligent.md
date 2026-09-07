# Photometric stereo validation on DiLiGenT

Mục tiêu của nhánh này là validate lớp hình học photometric stereo bằng dữ liệu chuẩn công khai, trước khi có hộp chụp 4 đèn thật. Nhánh này không dùng SAM, không dùng EfficientAD, và không liên quan trực tiếp tới anomaly detection.

## Dependency

FabLoop không vendor code GPL của `yasumat/RobustPhotometricStereo`. Clone nó như dependency local:

```bash
git clone https://github.com/yasumat/RobustPhotometricStereo.git third_party/RobustPhotometricStereo
```

Cài Python packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Trên Windows PowerShell, activate bằng:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
```

Trên WSL/Linux/macOS:

```bash
export PYTHONPATH=src
```

## Dataset layout

Tải DiLiGenT single-view dataset từ trang chính thức:

```text
https://sites.google.com/site/photometricstereodata/single
```

Đặt từng object vào `data/diligent/<object>/`. Script kỳ vọng các file chính:

```text
data/diligent/BallPNG/
  filenames.txt              # optional, dùng nếu có
  light_directions.txt
  light_intensities.txt      # optional, dùng nếu có
  mask.png
  Normal_gt.mat
  *.png / *.tif / ...
```

`light_directions.txt` và `light_intensities.txt` được chấp nhận ở cả dạng `Nx3` hoặc `3xN`. Nếu có `light_intensities.txt`, ảnh RGB sẽ được chia theo cường độ đèn từng kênh trước khi chuyển về grayscale.

## Run the 4-light smoke test

Chạy đúng 4 ảnh đầu tiên để mô phỏng điều kiện hộp 4 đèn:

```bash
PYTHONPATH=src python -m fabloop.photometric_stereo.diligent_validate \
  --object-dir data/diligent/BallPNG \
  --rps-root third_party/RobustPhotometricStereo \
  --image-counts 4 \
  --solvers l2 l1
```

Muốn chọn 4 đèn cụ thể thay vì 4 ảnh đầu, truyền zero-based indices:

```bash
PYTHONPATH=src python -m fabloop.photometric_stereo.diligent_validate \
  --object-dir data/diligent/BallPNG \
  --rps-root third_party/RobustPhotometricStereo \
  --indices 0,24,48,72 \
  --solvers l2 l1
```

## Run published-baseline comparison

Để so với baseline DiLiGenT công bố, chạy full object với 96 ảnh và solver L2:

```bash
PYTHONPATH=src python -m fabloop.photometric_stereo.diligent_validate \
  --object-dir data/diligent/BallPNG \
  --rps-root third_party/RobustPhotometricStereo \
  --image-counts all \
  --solvers l2
```

Script đọc baseline từ `references/diligent_main_l2_baseline.csv`. File này lấy dòng `BASELINE` của bảng "Main dataset" trên trang DiLiGenT official summary. Script attach:

```text
baseline_mae_deg
baseline_delta_deg
baseline_ratio
baseline_source_url
baseline_status
```

Không có ngưỡng pass/fail hard-code. Với 4 ảnh hoặc solver robust L1, script vẫn ghi MAE để so tương đối giữa L2/L1, nhưng không gán baseline công bố nếu điều kiện không khớp.

## Outputs

Mặc định output ở:

```text
outputs/photometric_stereo/<object>/
```

Mỗi run có:

```text
normal_gt_rgb.png
summary.csv
summary.json
lights_004/l2/normal_est.npy
lights_004/l2/normal_est_rgb.png
lights_004/l2/normal_gt_rgb.png
lights_004/l2/angular_error_deg.npy
lights_004/l2/angular_error_heatmap.png
lights_004/l2/height_frankot_chellappa.npy
lights_004/l2/height_frankot_chellappa.png
lights_004/l2/height_mesh.vtk
lights_004/l2/height_mesh.png
```

Nếu máy không render được PyVista off-screen, script vẫn hoàn thành phần metric và ghi cảnh báo mesh trong output metadata.

## References

- DiLiGenT single-view dataset: https://sites.google.com/site/photometricstereodata/single
- DiLiGenT benchmarking summary: https://sites.google.com/site/photometricstereodata/single/summary-of-benchmarking-results
- DiLiGenT CVPR 2016 paper: https://openaccess.thecvf.com/content_cvpr_2016/papers/Shi_A_Benchmark_Dataset_CVPR_2016_paper.pdf
- RobustPhotometricStereo: https://github.com/yasumat/RobustPhotometricStereo
