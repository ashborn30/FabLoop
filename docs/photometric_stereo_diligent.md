# Kiểm định photometric stereo trên DiLiGenT

Code đã chuyển từ thư mục con `FabLoop` sang `src/fabloop/photometric_stereo/` trong dự án này. Nhánh này kiểm định normal bằng ground truth DiLiGenT; bộ PCBA không có ground truth nên dùng [quy trình PCBA riêng](photometric_stereo_pcba.md). Chưa có ảnh DiLiGenT thật trong workspace, vì vậy chưa có kết quả tái lập baseline công bố.

## Môi trường và dependency

Dùng `.venv` Python 3.12 hiện có. NumPy, SciPy, OpenCV headless và scikit-learn đã nằm trong requirements chính; không cài chồng `opencv-python` hoặc thay requirements bằng file của repo cũ.

Dependency nghiên cứu đặt tại `third-party/RobustPhotometricStereo`, commit `f03aa95b57e746a7d31df76b1c0fa0a83584a3c1`. [Provenance](../references/robust_photometric_stereo_source.json) ghi hash source và điều kiện sử dụng upstream. Để dựng lại checkout:

```powershell
git clone https://github.com/yasumat/RobustPhotometricStereo.git third-party/RobustPhotometricStereo
git -C third-party/RobustPhotometricStereo checkout f03aa95b57e746a7d31df76b1c0fa0a83584a3c1
.\.venv\Scripts\python.exe scripts\validate_diligent.py --help
```

`--skip-mesh` bỏ phần xuất mesh PyVista; normal, MAE và height vẫn chạy đầy đủ. PyVista là dependency hiển thị tùy chọn, chưa cài vào môi trường training.

## Dataset và baseline

Mỗi object của [DiLiGenT chính thức](https://sites.google.com/site/photometricstereodata/single) cần `mask.png`, `Normal_gt.mat`, `light_directions.txt` và các ảnh. `filenames.txt` quy định thứ tự nếu có; `light_intensities.txt` được dùng để chuẩn hóa từng kênh RGB và bắt buộc khi so baseline công bố.

[CSV baseline](../references/diligent_main_l2_baseline.csv) có đủ 10 đối tượng, solver L2, 96 ảnh. Ngày 2026-09-08 đã đối chiếu và khớp đủ 10 số với dòng BASELINE của [bảng chính thức](https://sites.google.com/site/photometricstereodata/single/summary-of-benchmarking-results); [bằng chứng](../references/diligent_baseline_verification.json) lưu giá trị và hash CSV. Không sửa số liệu, không tạo baseline giả cho 4 ảnh hoặc L1.

```powershell
.\.venv\Scripts\python.exe scripts\validate_diligent.py --object-dir data/diligent/BallPNG --image-counts all --solvers l2 --require-baseline --skip-mesh
```

CSV cấu hình thiếu/rỗng/sai sẽ báo lỗi. `reference_available` nghĩa là đã gắn số tham chiếu; `not_applicable` nghĩa là protocol không có tham chiếu phù hợp. L2/96 ảnh của object DiLiGenT đã biết mà thiếu dòng CSV sẽ bị chặn trước khi solve. `--require-baseline` buộc mọi case yêu cầu phải có reference. Không đặt ngưỡng MAE pass/fail tùy ý.

## Chọn bộ 4 đèn

Không xem 4 ảnh đầu hoặc các index cách đều trong danh sách là mô phỏng hộp thật. Đọc hướng sáng trước:

```powershell
.\.venv\Scripts\python.exe scripts\validate_diligent.py --object-dir data/diligent/BallPNG --inspect-lights
```

Lệnh in index zero-based, tên ảnh, vector, azimuth, elevation, rank và condition number. Chọn 4 index theo hướng đèn thực đã hiệu chuẩn, rồi truyền chuỗi bốn số vào `--indices`. Độ trải azimuth khoảng 90° chưa đủ để khẳng định tương đương hộp: elevation, cường độ, frame camera và điều kiện chụp cũng phải tương ứng. Hiện không có `light_directions.txt` DiLiGenT hay calibration hộp, nên chưa chọn bộ index cụ thể.

Nếu chỉ cần smoke test tuần tự, phải khai báo rõ:

```powershell
.\.venv\Scripts\python.exe scripts\validate_diligent.py --object-dir data/diligent/BallPNG --image-counts 4 --selection sequential --solvers l2 l1 --skip-mesh
```

Index trùng, ít hơn 3 quan sát, vector không hữu hạn và ma trận hướng sáng rank khác 3 đều bị từ chối. Condition number được báo cáo, không tự đặt một ngưỡng để gọi hình học là tương đương hộp.

## Các điều chỉnh sau review

- Upstream có `psutil.py` nội bộ trùng tên package hệ thống. Adapter dùng phạm vi import tạm và khôi phục `sys.path`/`sys.modules`, tránh để Lightning/joblib nhận nhầm module sau khi solve. L2/L1 được kiểm tra với upstream thật; L1-multicore có smoke test riêng trên Windows.
- FC vẫn là FFT tuần hoàn trên toàn ảnh. Gradient ngoài mask bằng 0 vẫn tham gia lời giải; có thể có artifact ở biên và mất độ nghiêng trung bình. Height là tương đối theo pixel, không phải mm hoặc ground truth 3D.
- Trục Y của normal phải khai báo đúng khi dựng height. CLI DiLiGenT mặc định `--normal-y-axis up`; helper tổng quát/PCBA dùng Y đi xuống hàng ảnh. Chỉ phép đổi normal sang gradient chiều hàng thay đổi; normal lưu ra và MAE giữ nguyên. [Một implementation nghiên cứu đối chiếu frame DiLiGenT](https://raw.githubusercontent.com/facebookresearch/surface_normal_integration/main/datasets/diligent.py) cũng chuyển frame trước tích phân.

Output gồm normal, angular error, height, ảnh hiển thị và `summary.json`/`summary.csv`; metadata ghi điều kiện hình học, protocol baseline, quy ước trục và giới hạn của height. Kiểm tra tổng hợp đã chạy bằng ảnh Lambertian sinh nhân tạo; không dùng kết quả đó làm MAE DiLiGenT thật.
