# EfficientAD-S slimming: kiểm tra mức độ sẵn sàng

Ngày kiểm tra: 2026-09-08. Phạm vi: source, cấu hình, môi trường và các điểm tích hợp cần thiết trước khi huấn luyện. Chưa huấn luyện Student/AE, chưa đánh giá dataset và chưa có kết quả độ chính xác của biến thể slim.

**Trạng thái hiện tại:** Bước 1–4 đã hoàn thành. **Slim-0.5 candidate** đã có kiến trúc Student/AE giảm width nội bộ, vượt mục tiêu Student ≥40% ít params / ≥30% ít FLOPs được fvcore hỗ trợ, và vượt qua 19 kiểm tra kiến trúc/gradient/map/reload/shape/compression gate. Đây chưa phải kiến trúc cuối cùng; chưa nạp Teacher pretrained, chưa có optimizer step hoặc kết quả chất lượng/latency. Runtime hiện tại vẫn dùng baseline chính thức; factory/config, provenance checkpoint, output riêng và dữ liệu thí nghiệm còn cần tích hợp trước khi train.

## Bước 3: shape gate đã đạt

Dummy input `[1,3,256,256]` đã PASS trên CPU và CUDA. Với `padding=False`, Teacher, AE và hai nhánh Student đều có shape `[1,384,56,56]`; Student đầy đủ là `[1,768,56,56]`. Gate kiểm tra shape bằng nhau chính xác trước khi thực sự tính `T - S_T`, `T - A`, `A - S_A`; cả ba phép trừ đều cho kết quả hữu hạn. Không sửa công thức loss.

Kết quả: [Slim CPU](efficientad_slim_05_shapes.json), [Slim CUDA](efficientad_slim_05_shapes_cuda.json), [baseline CPU](efficientad_baseline_shapes.json), [tổng hợp kiểm tra](efficientad_shape_gate_validation.json).

Chạy lại bằng `.\.venv\Scripts\python.exe scripts\check_efficientad_shapes.py`; thêm `--device cuda:0` để kiểm tra CUDA. Lệnh ghi báo cáo PASS/FAIL và trả exit code 1 khi gate thất bại.

`src/trainer.py` bắt buộc chạy gate trước khi nạp Teacher pretrained, chuẩn bị Imagenette, tính thống kê feature hoặc tạo optimizer. Báo cáo từng seed được ghi vào thư mục checkpoint với tên `shape_check_seed_<seed>.json`. Gate giữ nguyên trạng thái model, gradient có sẵn, chế độ train/eval và RNG PyTorch. Toàn bộ 14 tests đã đạt, gồm kiểm tra trainer dừng khi gate thất bại. Đây là kiểm tra feature thô; lựa chọn Slim trong factory training vẫn thuộc bước tích hợp tiếp theo.

## Bước 4: compression gate đã đạt

Đo lại bằng `scripts/profile_efficientad_slim.py` trong `.venv`, dùng cùng tensor `[1,3,256,256]`, CPU float32, eval mode, `padding=False`. Student giảm **56,52% Params / 62,33% FLOPs**, đạt cả hai mục tiêu. AE giảm **69,88% / 67,44%**; Teacher + Student + AE giảm **39,44% / 37,51%**. Teacher giữ nguyên; chưa cần sửa Slim-0.5 để đạt hai mục tiêu thiết kế.

[Profile](efficientad_slim_05_profile.json) ghi đầy đủ số gốc/slim, tỷ lệ chưa làm tròn, breakdown từng module/operator, hash source/config/input và quyết định `PROCEED_TO_TRAINING_INTEGRATION`. FLOPs theo operator được fvcore hỗ trợ, một multiply-add = một FLOP; tổng ba mạng chỉ gồm forward feature, chưa gồm anomaly-map postprocessing, SAM3 hay 772 phần tử calibration.

CLI trả 0 khi cả hai mục tiêu Student đạt, trả 1 và lưu FAIL nếu thiếu bất kỳ mục tiêu nào. Khi FAIL phải sửa architecture trước train. [Kiểm tra gate](efficientad_compression_gate_validation.json) ghi nhận 19 tests PASS, gồm 5 tests mới cho ngưỡng inclusive, từng ngưỡng thất bại bất chấp mức giảm whole pipeline, và báo cáo/exit code của CLI. Chưa chạy training; factory hiện tại vẫn chọn baseline, và bước tích hợp Slim phải dùng kết quả compression gate cùng các gate shape/Teacher/dữ liệu.

## 1. Hợp đồng kiến trúc phải giữ nguyên

| Thành phần | Output cuối | Phần được thay đổi |
| --- | --- | --- |
| Teacher-S pretrained | 384 channels | Không thay đổi kiến trúc hoặc trọng số |
| Student-S | 768 channels = 384 ST + 384 SA | Chỉ channel nội bộ |
| Autoencoder | 384 channels | Chỉ channel nội bộ |

Giữ nguyên các phép tính `dTS`, `dTA`, `dSA`, hard-feature loss, penalty từ Imagenette, phép chuẩn hóa và phối hợp anomaly maps của source đã đóng băng. Đây là **architecture slimming + re-distillation từ Teacher frozen**, không phải distillation từ Student-S gốc sang Student mới.

Baseline hiện đã đúng với hợp đồng Student 768: `src/model.py:73` kiểm tra output Student bằng `teacher_out_channels * 2`. `configs/efficientad.yaml:21` đặt `size: small`, `teacher_out_channels: 384`; ba cấu hình diagnostic cũng dùng cùng thiết lập này. Không sửa `teacher_out_channels` để giảm kích thước Student.

Nguồn bài EfficientAD gốc là [EfficientAD, arXiv:2303.14535](https://arxiv.org/abs/2303.14535). [arXiv:2407.17909](https://arxiv.org/abs/2407.17909) là bài DFSC có phần trình bày lại EfficientAD, không phải paper EfficientAD gốc. Source of truth cho triển khai FabLoop là **Anomalib 2.6.0 đang cài trong `.venv`**, không tự thay bằng source mới nhất trên mạng.

## 2. Bước 1 đã chuẩn bị

- Đã sao chép nguyên byte `anomalib/models/image/efficient_ad/torch_model.py` vào [src/models/efficientad_slim/torch_model.py](../../src/models/efficientad_slim/torch_model.py), giữ copyright/SPDX và [LICENSE](../../src/models/efficientad_slim/LICENSE).
- [SOURCE.json](../../src/models/efficientad_slim/SOURCE.json) ghi phiên bản, đường dẫn gốc, trạng thái `unmodified_baseline_snapshot_not_slimmed`, đối chiếu wheel RECORD và SHA-256:

  ```text
  8f80c2db5e546e5209f105c757d8c88c70c48e86bcf1d036b797fff88be3bd08
  ```

- Đã cài và pin `fvcore==0.1.5.post20221221` trong [requirements.txt](../../requirements.txt).
- Không chỉnh `site-packages`. [src/model.py](../../src/model.py) vẫn khởi tạo `EfficientAd` chính thức. Bản copy `torch_model.py` giữ nguyên width baseline; candidate giảm width được đặt riêng trong `slim_model.py` và chưa nối vào training/evaluation runtime.

SHA-256 trên là checksum **file source**, không phải checksum trọng số Teacher pretrained. Teacher gate hiện tại vẫn cần kiểm tra state dict trước/sau huấn luyện và sau reload.

Git repository hiện tồn tại, nhưng `git ls-files` tại thời điểm audit chỉ có `README.md`; `src`, `configs`, `requirements.txt`, `scripts` và `docs` còn untracked. Snapshot hash xác nhận nội dung source được copy, chưa thay thế một commit/tag có thể tái tạo toàn thí nghiệm. Việc commit/tag baseline cần thực hiện riêng khi nội dung dự án đã được rà soát; phiên chuẩn bị này không commit các thay đổi hiện có của người dùng.

Số đo kiến trúc baseline được lưu riêng tại [efficientad_baseline_profile.json](efficientad_baseline_profile.json), tạo bởi [scripts/profile_efficientad_baseline.py](../../scripts/profile_efficientad_baseline.py). Đọc trực tiếp artifact này để lấy số params/FLOPs và giới hạn tracing; báo cáo readiness không chép các con số chưa đo. Forward/tracing bằng tensor tổng hợp phục vụ kiểm tra kiến trúc không cung cấp bằng chứng AUROC hoặc chất lượng anomaly maps trên dữ liệu.

Kết quả đã đo với cùng tensor CPU float32 `[1, 3, 256, 256]`, eval mode, `padding=false`:

| Thành phần | Params | FLOPs được fvcore hỗ trợ | Shape đầu ra |
| --- | ---: | ---: | --- |
| Teacher-S | 2,694,144 | 15,310,903,296 | `[1,384,56,56]` |
| Student-S | 4,267,392 | 20,243,404,800 | `[1,768,56,56]` |
| AE | 1,096,320 | 2,399,267,840 | `[1,384,56,56]` |
| Tổng T + S + AE | 8,057,856 | 37,953,575,936 | — |

Tổng trên không tính thêm 772 phần tử parameter lưu mean/std/quantiles. FLOPs dùng một multiply-add = một FLOP, gồm convolution và bilinear interpolation được hỗ trợ; `avg_pool2d`, phép trừ/chia của ImageNet normalization chưa được đếm. ReLU và các toán tử được fvcore chủ động bỏ qua cũng không có trong tổng. Không dùng các con số này như all-op FLOPs, full FabLoop cost hoặc latency Jetson.

Theo cùng phép đếm, mục tiêu Student ≥40% ít params hơn tương ứng **không quá 2,560,435 params**; ≥30% ít FLOPs hơn tương ứng **không quá 14,170,383,360 supported FLOPs**. Candidate Slim-0.5 ở Bước 2 đã đạt cả hai ngân sách kiến trúc này.

## Bước 2 — Slim Student và Slim Autoencoder đã triển khai

[slim_model.py](../../src/models/efficientad_slim/slim_model.py) thêm ba lớp `SlimStudent`, `SlimAutoEncoder`, `SlimEfficientAdModel`, được export từ package. `torch_model.py` và `SOURCE.json` tiếp tục là snapshot baseline nguyên trạng của Bước 1; không thay source trong `site-packages`.

| Thành phần | Slim-0.5 candidate |
| --- | --- |
| Teacher-S | Giữ nguyên `3 → 128 → 256 → 256 → 384` |
| Student | `3 → 64 → 128 → 128 → 768`, chia 384 ST + 384 SA |
| AE encoder | `3 → 16 → 16 → 32 → 32 → 32 → 32` |
| AE decoder | Bảy convolution 32 channels nội bộ, convolution cuối `32 → 384` |

Chỉ thay `in_channels`/`out_channels` của convolution Student/AE. Số layer, kernel, stride, dilation, pooling, padding, spatial geometry, activation, interpolation và dropout giữ nguyên. Các forward, loss `dTS`/`dTA`/`dSA` và anomaly-map methods kế thừa trực tiếp từ snapshot. Teacher có `requires_grad=False` và luôn ở eval mode sau `model.train()`; Student/AE cùng dropout vẫn đổi mode bình thường. Constructor khóa output 384/768/384.

Khởi tạo độc lập tại thư mục repository, trong `.venv`:

```python
from src.models.efficientad_slim import SlimAutoEncoder, SlimEfficientAdModel, SlimStudent

student = SlimStudent()
autoencoder = SlimAutoEncoder()
model = SlimEfficientAdModel(padding=False, pad_maps=True)
model.eval()
```

Constructor chỉ khởi tạo trọng số; chưa nạp Teacher pretrained hoặc checkpoint đã train. Khi tích hợp runtime phải tiếp tục dùng luồng nạp Teacher đã xác minh và checksum/gate hiện tại.

Kiểm tra và tạo lại số đo:

```powershell
python -m unittest discover -s tests -p test_efficientad_slim.py -v
python scripts/profile_efficientad_slim.py
```

[Bảy unittest](../../tests/test_efficientad_slim.py) đã PASS trong 7,277 giây: source hash/geometry và các method kế thừa; shape input vuông 256 và không vuông với cả hai chế độ padding; finite loss/backward với gradient hữu hạn ở cả hai nhóm output Student và AE, Teacher không có gradient; Teacher khởi tạo cùng seed không đổi; raw/normalized map parity với core gốc dùng cùng các module; strict state-dict roundtrip; từ chối output hoặc state dict không tương thích. Backward dùng tensor tổng hợp, không có optimizer step. Map parity ở đây kiểm tra logic khi dùng cùng module/trọng số, không có nghĩa output của mạng baseline rộng và mạng slim bằng nhau.

Kết quả kiểm tra được lưu tại [efficientad_slim_05_validation.json](efficientad_slim_05_validation.json).

Kết quả [efficientad_slim_05_profile.json](efficientad_slim_05_profile.json), tạo bởi [profile_efficientad_slim.py](../../scripts/profile_efficientad_slim.py), dùng **cùng tensor CPU float32 `[1,3,256,256]`**, eval mode, `padding=false` cho baseline và candidate:

| Phạm vi | Params candidate | FLOPs được fvcore hỗ trợ | Giảm Params | Giảm FLOPs |
| --- | ---: | ---: | ---: | ---: |
| Student-only | 1,855,552 | 7,625,419,776 | 56,52% | 62,33% |
| AE | 330,240 | 781,088,256 | 69,88% | 67,44% |
| EfficientAD core T + S + AE | 4,879,936 | 23,717,411,328 | 39,44% | 37,51% |

Teacher giữ nguyên 2,694,144 params / 15,310,903,296 supported FLOPs. Compression gate **Student-only** đã PASS; mức giảm core được báo riêng. Core ở bảng này chỉ gồm forward của Teacher/Student/AE, không gồm anomaly-map postprocessing, SAM3 hoặc 772 phần tử parameter mean/std/quantiles. Quy ước fvcore là một multiply-add = một FLOP; operator chưa hỗ trợ hoặc được profiler chủ động bỏ qua vẫn không nằm trong tổng, với đầy đủ coverage trong JSON.

Số đo chỉ xác nhận độ nhỏ của kiến trúc **Slim-0.5 candidate**. Chưa huấn luyện, chưa đánh giá AUROC/anomaly-map quality, latency hoặc Jetson; chưa chọn kiến trúc cuối cùng. `src/model.py`, `src/trainer.py` và các config tiếp tục sử dụng baseline.

## 3. Các phần baseline có thể tái sử dụng

| Chức năng | Vị trí hiện tại | Yêu cầu khi tích hợp slim |
| --- | --- | --- |
| Teacher frozen và eval | `src/model.py:67`, `:89` | Giữ nguyên kiểm tra |
| Optimizer chỉ Student + AE | `src/model.py:86`, `src/trainer.py:165` | Giữ nguyên phạm vi |
| Loss do EfficientAD core xử lý | `src/trainer.py:218` | Không viết lại công thức loss |
| Teacher checksum khi lưu/reload | `src/trainer.py:87`, `:105`, `:282` | Giữ nguyên gate |
| Calibration bằng normal-only | `src/calibrator.py:29`, `:41` | Chạy lại riêng cho mỗi kiến trúc/checkpoint |
| Teacher checksum sau calibration | `src/calibrator.py:94` | Giữ nguyên kiểm tra |
| Local/global map và phép trộn 50/50 | `src/evaluator.py:318` | Giữ API và toán học |
| Checksum và reproducible reload | `src/validator.py:95`, `:122` | Áp dụng cho cả baseline/slim |
| Reproduction gate hiện có | `src/reporter.py:156` | Giữ các tiêu chí hiện hành |

## 4. Phần còn thiếu trước khi train slim

### Factory và cấu hình width

`src/model.py:40` hiện chỉ tạo model chính thức; chưa có selector kiến trúc hoặc Student/AE internal widths. Cần thêm cấu hình variant và width có kiểm tra đầu vào, đồng thời giữ baseline làm giá trị mặc định tương thích.

Khi thay core, phải giữ `lightning_model.model` và `wrapper.core` cùng trỏ tới model được chọn. Các helper của Lightning vẫn được dùng để nạp Teacher và chuẩn bị Imagenette/mean-std (`src/trainer.py:158`, `:161`, `:162`) cũng như map quantiles (`src/calibrator.py:41`). Chỉ thay `wrapper.core` sẽ khiến các helper hoạt động trên một model khác.

Candidate đã khóa output 384/768/384 và giữ spatial geometry, kernels, strides, padding, interpolation cùng cách chia nhánh. Factory/config cần tiếp tục kiểm tra hợp đồng này khi chọn variant. Width 0.5 đã đạt mục tiêu nén Student, nhưng vẫn là candidate cần đánh giá chất lượng trước khi chốt kiến trúc.

### Checkpoint và định danh kiến trúc

`src/trainer.py:85` dựng model theo config đang truyền vào; mặc dù payload lưu config tại `:111`, loader chưa dùng nó để dựng/đối chiếu kiến trúc. Payload hiện có `format_version: 1` (`:107`) và mới kiểm tra category/seed (`:83`).

Cần lưu và kiểm tra architecture ID, source version/hash, Student/AE widths cùng hợp đồng output trước khi nạp state dict. Giữ `strict=True` tại `src/model.py:97`; không bỏ strict để làm checkpoint baseline nạp được vào slim. Checkpoint baseline có thể tiếp tục dùng default architecture cũ nếu metadata thiếu, nhưng phải có quy tắc tương thích rõ ràng.

`src/evaluator.py:270` cũng chỉ xác nhận category/seed cho calibration JSON. Artifact mới nên được gắn với kiến trúc và checkpoint cụ thể để tránh dùng nhầm phân phối score/threshold của một model khác.

### Output riêng cho từng biến thể

`src/trainer.py:26` hiện đặt checkpoint theo output root/category/seed, không có variant. Calibration (`src/calibrator.py:17`), predictions (`src/evaluator.py:278`), benchmark (`src/benchmarker.py:75`) và reporter (`src/reporter.py:30`) dùng cùng quy ước.

Mỗi biến thể cần output root riêng, ví dụ `outputs/efficientad` cho baseline và `outputs/efficientad_slim/<variant>` cho slim. Không chạy hai kiến trúc vào cùng output root với cùng category/seed. Giữ cùng split seed, danh sách mẫu, input resolution và quy trình calibration để so sánh công bằng.

### Kiểm tra kiến trúc và reload

Dry-run của runtime hiện tại chỉ kiểm tra Student có hai nhóm channels (`src/model.py:73`). Bảy test độc lập của Bước 2 đã kiểm tra hợp đồng output/spatial shapes, loss/gradient, map parity, Teacher frozen và strict state-dict roundtrip. Bước tích hợp tiếp theo cần đưa các hợp đồng này vào factory/runtime và kiểm tra checkpoint roundtrip theo architecture metadata, Teacher checksum sau nạp pretrained, calibration provenance và output routing. Candidate chỉ hỗ trợ width 0.5; baseline vẫn dùng lớp gốc riêng.

## 5. Params/FLOPs và cách báo cáo

Benchmark runtime hiện tại yêu cầu checkpoint đã calibrated và normal-loader (`src/benchmarker.py:22`), sau đó đo toàn `wrapper.core` (`:36`, `:48`). Artifact tại `:58` chỉ có latency/FPS/VRAM, chưa có params, FLOPs hoặc so sánh nén. Reporter mới tổng hợp latency/memory (`src/reporter.py:76`, `:108`, `:241`).

Profiler kiến trúc độc lập đã báo hai nhóm số liệu dưới đây bằng cùng tensor input, batch size, precision, resolution và convention. Cần nối cách báo cáo này vào artifact runtime/report khi tích hợp slim:

| Nhóm | Params | FLOPs | Mục đích |
| --- | --- | --- | --- |
| Student-only | Student | Student | Áp dụng mục tiêu giảm ít nhất 40% params, 30% FLOPs |
| EfficientAD core deployment | Teacher + Student + AE | Teacher + Student + AE | Báo cáo mức giảm thực tế của core triển khai |

Nên lưu riêng số đo Teacher và AE để giải thích tổng. Reduction cho mỗi nhóm là `1 - slim / baseline`, dùng baseline có cùng không gian feature cuối. Teacher cố định làm giảm lợi ích khi chuyển từ Student+AE sang tổng core; tuy nhiên so sánh **tổng core với riêng Student** còn phụ thuộc mức giảm AE, nên phải dựa trên số đo, không mặc định một quan hệ lớn/nhỏ.

[fvcore](https://github.com/facebookresearch/fvcore) tính một multiply-add là một FLOP. Ghi rõ convention, phiên bản thư viện, unsupported operators và phạm vi operator được tính. Không so trực tiếp kết quả fvcore với con số của paper nếu profiler/convention khác nhau; không diễn giải tổng FLOPs có operator chưa được hỗ trợ thành phép đếm đầy đủ tuyệt đối.

Giữ reproduction gate hiện hành. Compression gate Student của profiler Slim-0.5 đã PASS và cần được đưa vào báo cáo thí nghiệm riêng với gate chất lượng. Các cấu hình hiện có `visual_review_pass: true` (ví dụ `configs/efficientad.yaml:74`); thí nghiệm slim cần review riêng trước khi được đánh dấu đạt.

Config VisA đang bật SAM (`configs/efficientad.yaml:53`) và evaluator khởi tạo SAM tại `src/evaluator.py:290`, còn benchmark chỉ đo EfficientAD core. Vì vậy **Teacher+Student+AE phải ghi rõ là EfficientAD core**, không phải toàn bộ FabLoop có conditional SAM. Nếu báo cáo FabLoop end-to-end trên Jetson sau này, cần đo thêm tác động của SAM được gọi theo điều kiện và các bước nằm ngoài core.

## 6. Dữ liệu hiện tại và điểm chặn chạy thí nghiệm

| Đường dẫn | Trạng thái tại thời điểm audit |
| --- | --- |
| `data/PCBA_4Light` | Có dữ liệu ảnh theo PCB và hướng chiếu sáng |
| `data/PCBA_4Light_edited` | Có bản ảnh đã chỉnh sửa tương ứng |
| `data/visa` | Chưa có |
| `data/imagenette` | Chưa có; helper hiện có thể tải khi train |
| `data/mvtec_ad` | Chưa có |
| `data/mvtec_loco` | Chưa có |
| `third-party/spot-diff/split_csv/1cls.csv` | Có |
| `third-party/SAM3/checkpoints/sam3.pt` | Có; chỉ xác nhận sự tồn tại trong audit này |
| `outputs/efficientad` | Chưa có kết quả/checkpoint |

Routing hiện chỉ hỗ trợ Visa, MVTecAD và MVTecLoco (`src/data.py:24`). VisA chỉ nhận `pcb1`–`pcb4` với official one-class split (`:249`, `:252`). **PCBA_4Light chưa có dataset adapter/config và không thể chạy bằng config VisA hiện tại.**

PCBA có nhiều ảnh cùng board (`light_B/F/L/R.jpg`, ambient hoặc tên ảnh gốc), đồng thời có thư mục bản edited. Trước khi dùng cần xác định rõ nhãn normal/anomaly, masks, train/calibration/test và chính sách giữ các ảnh liên quan của cùng board trong một split để tránh leakage. Không tự suy luận nhãn chất lượng từ tên thư mục hoặc coi bản edited là sample độc lập cho split.

`src/anomaly.py:92` preflight thông thường dựng dataset splits và ghi manifest; nó sẽ chưa qua được khi thiếu VisA được cấu hình. Kiểm tra source/shape/complexity không cần dataset nên có thể thực hiện trước, còn train/calibrate/evaluate phải chờ dữ liệu và routing phù hợp.

## 7. Trình tự triển khai tiếp theo

1. Giữ snapshot/provenance Bước 1 và candidate/test/profile Bước 2 đã hoàn thành.
2. Thêm factory/config chọn baseline hoặc Slim-0.5, giữ hợp đồng output/maps/loss và luồng nạp Teacher pretrained.
3. Bổ sung architecture-aware checkpoints, provenance calibration và output root riêng; kiểm tra reload qua runtime cùng Teacher checksum.
4. Nối báo cáo Student-only và EfficientAD core cùng compression gate vào artifact/report thí nghiệm.
5. Chuẩn bị official VisA hoặc adapter/split PCBA có nhãn được xác định; giữ split giống nhau giữa các biến thể.
6. Huấn luyện từ cùng Teacher pretrained frozen, calibration riêng từng checkpoint, đánh giá bằng evaluator hiện có và giữ Teacher/reproduction gate trước khi chọn kiến trúc cuối cùng.
