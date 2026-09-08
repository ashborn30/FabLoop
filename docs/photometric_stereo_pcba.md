# Chuẩn bị PCBA 4 đèn cho photometric stereo và training

## Dữ liệu thực tế

`data/PCBA_4Light` và `data/PCBA_4Light_edited` mỗi thư mục có **34 nhóm PCB1–PCB34, 170 ảnh**. Đã đọc đầy đủ và tính SHA-256 cho cả 340 ảnh; không có ảnh hỏng. Mỗi nhóm có đủ `light_F.jpg`, `light_B.jpg`, `light_L.jpg`, `light_R.jpg`.

**33/34 nhóm edited có kích thước khác nhau giữa các đèn**. PCB15 có cùng kích thước 484×2560 nhưng chưa chứng minh đã căn chỉnh. Ảnh gốc cùng kích thước trong từng nhóm, song cũng cần kiểm tra cùng viewpoint. Resize bốn ảnh về một size không thay thế registration: một pixel trong stack phải chỉ cùng điểm trên board. Thành phần cao trên PCB còn có thể gây parallax khi camera/board di chuyển; một homography phẳng không đảm bảo toàn bộ linh kiện thẳng hàng.

Không tìm thấy nhãn normal/anomaly, mask board/lỗi, split, quan hệ board vật lý hoặc hướng/cường độ đèn. `PCB8/03f712c14b09ca579318.jpg` được giữ là ảnh chưa phân loại, không tự coi là ambient. F/B/L/R chỉ là tên hướng; chưa thể suy ra vector sáng 3D, góc nâng hoặc exposure từ tên file.

## Bước đã làm: inventory và metadata

```powershell
.\.venv\Scripts\python.exe scripts\prepare_pcba_4light.py
.\.venv\Scripts\python.exe scripts\prepare_pcba_4light.py --require-ready
```

Lệnh thứ nhất hoàn thành kiểm kê với exit 0. Lệnh thứ hai hiện trả exit 1 vì thiếu căn chỉnh và metadata. Đây là gate kiểm kê, chưa phải bộ xuất dataset training. Các file:

- `data/processed/pcba_4light/inventory.json`: đường dẫn/hình dạng/hash từng ảnh, liên kết raw/edited, thứ tự F/B/L/R và lỗi theo từng board.
- `data/processed/pcba_4light/labels.json`: mẫu nhãn giữ `unknown`; chạy lại không ghi đè chỉnh sửa của người dùng.
- [Báo cáo hiện tại](preflight/pcba_4light_readiness.json): `ready_for_training=false`.

Điền `physical_board_id`, `category`, `label` (`normal`/`anomaly`) và `split` (`train`/`calibration`/`test`) bằng thông tin đã xác minh. Mọi ảnh đèn, crop, normal và patch của cùng board vật lý phải ở cùng split. Không chia bốn đèn thành bốn mẫu độc lập rồi rải train/test. Train và calibration chỉ dùng normal; không coi thiếu nhãn là normal.

## Xuất bản thử trực tiếp từ PCBA_4Light_edited

Khi chưa có calibration, dùng lệnh riêng `scripts/export_pcba_preview.py` để xuất **normal/height định tính**. Cờ `--nominal-lights` lựa chọn rõ chế độ hướng đèn giả định. Đây là dữ liệu để xem và khảo sát; các file không được đánh dấu đã hiệu chuẩn hoặc đủ điều kiện train.

Mở PowerShell tại dự án:

```powershell
Set-Location "D:\Visual studio code\FabLoop_VJSS26"
```

Chạy thử một board trước:

```powershell
.\.venv\Scripts\python.exe scripts\export_pcba_preview.py --nominal-lights --boards PCB1 --output-root outputs/pcba_preview_PCB1
```

Sau khi xem ảnh căn chỉnh và normal/pseudo-3D, chạy toàn bộ thư mục:

```powershell
.\.venv\Scripts\python.exe scripts\export_pcba_preview.py --nominal-lights --input-root data/PCBA_4Light_edited --output-root outputs/pcba_photometric_preview --max-edge 1024
```

Không cần kích hoạt venv vì lệnh gọi trực tiếp Python trong `.venv`. `--boards PCB1 PCB2` giới hạn nhóm cần chạy. `--max-edge` là cạnh dài tối đa khi xử lý, giữ tỷ lệ ảnh; mặc định 1024. Xử lý tuần tự trên CPU, không huấn luyện EfficientAD.

Mỗi board xuất dưới `outputs/pcba_photometric_preview/PCB<n>/`:

```text
reference_rgb.png    Ảnh F làm hệ tọa độ tham chiếu
alignment_preview.png Ảnh ghép F/B/L/R sau căn chỉnh ước lượng
normal_est.npy       Vector normal ước lượng dưới giả định ánh sáng
normal_rgb.png       Ảnh normal RGB
relativeheight.npy   Height tương đối, NaN ngoài vùng hợp lệ
height_preview.png   Ảnh hiển thị height
pseudo3d.png         Hình bề mặt pseudo-3D
valid_mask.png       Vùng hợp lệ số học, không phải mask board ground truth
report.json          Giả định, chất lượng căn chỉnh và provenance
```

## Pseudo-3D Viewer realtime bằng normal mapping

Viewer nhẹ nằm ở `src/fabloop/photometric_stereo/pseudo3d_viewer.py`. Nó đọc từng folder PCB có `normal_est.npy` hoặc `normal_l2_nominal.npy` cùng `albedo.npy`/`albedo_l2_nominal.npy`/ảnh albedo tương đương, rồi render ảnh relight 2D:

```text
shading(x,y) = max(0, normal(x,y) . L_virtual) * albedo(x,y)
```

`L_virtual` lấy từ hai slider:

- `azimuth`: 0-360 độ trong mặt phẳng ảnh, frame x sang phải/y xuống.
- `elevation`: 0-90 độ, 90 là chiếu từ phía camera.

Chạy viewer độc lập:

```powershell
.\.venv\Scripts\python.exe scripts\launch_pseudo3d_viewer.py --results-root outputs/pcba_photometric_preview
```

Liệt kê board mà không mở GUI:

```powershell
.\.venv\Scripts\python.exe scripts\launch_pseudo3d_viewer.py --results-root outputs/pcba_photometric_preview --list-only
```

Xuất một ảnh relight cho từng board để review hàng loạt:

```powershell
.\.venv\Scripts\python.exe scripts\export_pseudo3d_relight_previews.py --results-root outputs/pcba_photometric_preview --output-dir outputs/pcba_relight_preview --azimuth 315 --elevation 45
```

Nhúng vào dashboard Qt đã có:

```python
from fabloop.photometric_stereo.pseudo3d_viewer import add_pseudo3d_tab

add_pseudo3d_tab(tabs, "outputs/pcba_photometric_preview")
```

Viewer chỉ là relight định tính để xem nổi khối từ normal map; nó không thay thế calibration, MAE, mask ground truth hoặc phép đo 3D metric.

`summary.json` tại thư mục output ghi số board xuất được/thất bại và lý do. Exit 0 nghĩa là mọi board đã xuất bản thử; không phải gate chất lượng hình học. Exit 1 nếu có lỗi. Lệnh giữ output cũ: muốn chạy lại, chọn tên `--output-root` mới.

Đã chạy thử thực tế PCB1 ở cạnh dài 1024, xuất đủ 9 file tại `outputs/pcba_preview_smoke_PCB1/PCB1/`; toàn bộ 50 tests đã PASS. Phép căn chỉnh ảnh R ước lượng xoay gần 180° và height hiện có thành phần cong quy mô lớn. Với ánh sáng chưa hiệu chuẩn, không diễn giải dạng cong đó thành độ cong thật của board. Chưa chạy batch 34 board; lệnh batch phía trên để người dùng tự chạy.

Các giả định/giới hạn được ghi rõ trong report:

- F/B/L/R lần lượt dùng `(0,-1,1)`, `(0,1,1)`, `(-1,0,1)`, `(1,0,1)`, chuẩn hóa độ dài; frame X sang phải, Y xuống hàng ảnh, Z về camera. Đây là hướng danh định theo log cũ, chưa được đo từ hộp.
- Giả định response sRGB và cường độ đèn bằng nhau. Không tự ước lượng calibration hoặc trừ ảnh ambient crop khác; ánh sáng môi trường có thể ảnh hưởng kết quả.
- Dùng feature matching và homography để ước lượng căn chỉnh về ảnh F, có vùng giao hợp lệ. Các ảnh khác kích thước được thu nhỏ giữ tỷ lệ và warp theo phép biến đổi ước lượng; không resize méo về cùng shape. Homography trên board phẳng không đảm bảo linh kiện cao hết parallax.
- Các ngưỡng lọc tín hiệu/normal chỉ là heuristic để dựng bản thử. Mask không phải nhãn board/lỗi. Height FC vẫn tương đối, có giả định FFT tuần hoàn và artifact ở biên; không phải số đo mm.
- Report giữ `calibrated=false`, `registration_verified=false`, `training_ready=false`. Chưa có MAE trên normal ground truth hoặc nhãn anomaly để chứng minh chất lượng training.

## Bridge dùng capture đã hiệu chuẩn

`scripts/process_pcba_photometric.py` dùng solver của adapter đã chuyển, không yêu cầu `Normal_gt.mat`. Đầu vào là config của **một capture đã căn chỉnh và hiệu chuẩn**, theo [mẫu](../configs/pcba_photometric_capture.example.json). Mẫu có đường dẫn/vector rỗng và gate false nên chủ động từ chối chạy; không điền vector sáng giả để vượt gate.

```powershell
.\.venv\Scripts\python.exe scripts\process_pcba_photometric.py --help
```

Config thực phải ghi đường dẫn bốn ảnh và mask cùng hệ tọa độ, vector/cường độ RGB theo F/B/L/R, response ảnh `linear` hoặc `srgb`, `registration_verified=true`, `calibrated=true`, frame `x_right_y_down_z_towards_camera`. Đường dẫn ảnh tính tương đối với file config. Response phải dựa trên quy trình chụp, không mặc nhiên xem JPEG camera là irradiance tuyến tính. Bridge không tự resize, registration hay đo calibration.

Bridge xuất normal float, normal RGB, height tương đối và provenance. Nó chưa tự chia split hoặc tạo nhãn anomaly. Quan sát cần đã xử lý ambient hoặc có ambient không đáng kể; bản đầu ghi rõ chưa hiệu chỉnh ambient, không tự trừ ảnh ambient có crop khác. Không dùng height làm ground truth mm. Các vùng shadow, specular, occlusion và biên mask cần được kiểm tra trên PCB thật.

## Điều kiện còn thiếu để train

Cần xác nhận board bình thường/có lỗi, nhóm loại board và định danh vật lý; sau đó khóa split theo board. Cần căn chỉnh, mask và calibration nếu chọn normal/height làm representation. Nếu chọn RGB thì không cần calibration PS, nhưng vẫn cần nhãn/split đúng và chính sách crop/patch giữ tỷ lệ ảnh. Resize trực tiếp board rất dài về 256×256 có thể làm méo linh kiện.

Runtime EfficientAD hiện vẫn dùng factory baseline và routing VisA/MVTec. Chưa nối dataset PCBA hoặc Slim vào trainer; không đổi `data.root` của config VisA sang PCBA để chạy. Việc tạo input 3 kênh từ normal RGB cũng cần thí nghiệm riêng, không chứng minh rằng Teacher pretrained trên ảnh tự nhiên giữ nguyên chất lượng.

Không trộn ảnh DiLiGenT vào tập PCB để thay thế nhãn hoặc bổ sung normal train. DiLiGenT dùng kiểm định thuật toán hình học; PCBA dùng dữ liệu thí nghiệm thực của bài toán board.
