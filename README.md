# Multiple-Pill Recognition and Interaction Safety

Project xây dựng hệ thống nhận diện nhiều viên thuốc trong một ảnh và cảnh báo tương tác thuốc dựa trên dữ liệu có kiểm chứng.

## Tổ Chức Folder

```text
src/          -> code lõi có thể import lại
training/     -> script để train/evaluate, gọi code trong src/
inference/    -> script để chạy dự đoán ảnh mới, gọi code trong src/
experiments/  -> kết quả từng lần train/evaluate: logs, metrics, checkpoints
models/       -> weight chính thức được chọn để chạy inference
outputs/      -> kết quả sinh ra khi chạy inference/demo
docs/         -> tài liệu đồ án: report, slide, paper summary, specification
scripts/      -> các công cụ hỗ trợ độc lập (ví dụ: chia dataset)
```

Phân biệt nhanh các folder dễ nhầm:

| Cặp folder | Khác nhau thế nào? |
|---|---|
| `src/` vs `training/` | `src/` chứa logic/model/pipeline dùng lại được; `training/` chỉ chứa script chạy train/evaluate. |
| `src/` vs `inference/` | `src/` là code lõi; `inference/` là entrypoint để load model và chạy ảnh mới. |
| `src/.../trainers/` vs `training/` | `trainers/` là code train import được; `training/` là script chạy experiment cụ thể. |
| `src/.../predictors/` vs `inference/` | `predictors/` là logic dự đoán import được; `inference/` là script chạy predict cho ảnh mới/demo. |
| `training/.../evaluation/` vs `experiments/.../metrics/` | `evaluation/` là code tính metric; `metrics/` là kết quả metric đã sinh ra. |
| `experiments/.../checkpoints/` vs `models/` | `checkpoints/` là checkpoint thô của từng lần train; `models/` chỉ chứa weight tốt nhất dùng chính thức. |
| `experiments/` vs `outputs/` | `experiments/` lưu kết quả train/test; `outputs/` lưu kết quả khi chạy inference/demo. |
| `data/processed/` vs `data/augmented/` | `processed/` là dữ liệu đã chuẩn hóa; `augmented/` là dữ liệu sinh thêm từ train split. |
| `docs/` vs `src/` | `docs/` chứa tài liệu đồ án; `src/` chứa code hệ thống. |

Frontend/Demo UI chưa được tạo ở giai đoạn này.

## Môi Trường

Phiên bản Python chốt cho project:

```text
Python 3.11.9
```

Khi chạy local, ưu tiên đúng `Python 3.11.9` để đồng nhất với `.python-version`. Khi chạy trên Colab/Kaggle, chấp nhận runtime `Python 3.11.x` hoặc `Python 3.12.x`, nhưng cần kiểm tra lại version trước khi train.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Các thư viện đã được pin trong `requirements.txt`. Nếu máy dùng GPU NVIDIA, có thể cần cài PyTorch theo CUDA tương ứng từ hướng dẫn chính thức của PyTorch, sau đó cài các thư viện còn lại trong `requirements.txt`.

### Colab/Kaggle

Trước khi chạy notebook trên Colab hoặc Kaggle, kiểm tra runtime:

```python
import sys
import torch

print(sys.version)
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Khuyến nghị:

| Nền tảng | Cách dùng |
|---|---|
| Local | Dùng `Python 3.11.9` và `pip install -r requirements.txt`. |
| Colab | Có thể dùng runtime mới, nhưng nếu cần khớp project thì cài lại package theo `requirements.txt`. |
| Kaggle | Ưu tiên GPU T4/P100-compatible runtime; nếu PyTorch mới không nhận P100 thì cài lại PyTorch pinned trong `requirements.txt` hoặc đổi accelerator. |

Nếu Colab/Kaggle đã có sẵn PyTorch mới hơn và gây xung đột, cài lại PyTorch trước rồi mới cài các thư viện còn lại.

## Cây Thư Mục

```text
Multiple-Pill-Recognition-And-Interaction-Safety/
├── configs/                                      # Cấu hình chạy hệ thống, KHÔNG chứa code model
│   ├── training/                                 # Config cho train: epoch, batch size, lr, augmentation, split path
│   └── inference/                                # Config cho chạy thật: weight path, threshold, OCR/RAG setting
│
├── data/                                         # Dữ liệu của project, thường KHÔNG commit toàn bộ lên Git
│   ├── raw/                                      # Dữ liệu gốc tải về, giữ nguyên để có thể tái xử lý
│   │   ├── mediseg/                              # MEDISEG gốc cho Module 1 segmentation
│   │   └── nih_rximage/                          # NIH/RxImage gốc cho Module 2 attribute recognition
│   ├── processed/                                # Dữ liệu đã làm sạch/convert/crop/normalize label
│   │   ├── mediseg_yolo_segmentation/            # MEDISEG đã convert sang YOLO segmentation format
│   │   └── nih_attribute/                        # NIH/RxImage đã crop và chuẩn hóa label attribute
│   ├── splits/                                   # File train/val/test split, dùng để tái lập thí nghiệm
│   │   ├── mediseg/                              # Split cho segmentation
│   │   └── nih_attribute/                        # Split chống leakage cho attribute theo rxcui/ndc11
│   ├── augmented/                                # Dữ liệu sinh thêm, chỉ nên sinh từ train split
│   │   ├── mediseg/                              # Augmentation/copy-paste cho YOLOv11-Seg
│   │   └── nih_attribute/                        # Augmentation/sim2real cho ResNet18 attribute
│   └── benchmark/                                # Dữ liệu đánh giá ngoài dataset chính
│       └── real_world/                           # Ảnh tự chụp để kiểm tra domain gap thực tế
│
├── database/                                     # Tài nguyên database cho thuốc, appearance và DDI
│   ├── migrations/                               # SQL/migration tạo hoặc cập nhật schema
│   └── seeds/                                    # Dữ liệu/script seed cho drug, appearance, ingredient, DDI
│
├── docs/                                         # Tài liệu đồ án và contract giữa các module
│   ├── Overview.md                               # Tổng quan đề tài, mục tiêu và ứng dụng
│   ├── CV_Module.md                              # Đặc tả kiến trúc Computer Vision module
│   ├── RAG_Module.md                             # Đặc tả Retrieval/RAG, database, DDI và report
│   ├── metric_CV.md                              # Metric đánh giá các module CV
│   ├── metric_LLM.md                             # Metric đánh giá retrieval/ranking/safety
│   ├── train_request.md                          # Yêu cầu log và artifact khi train model
│   └── schema.md                                 # Input/output contract giữa các module
│
├── experiments/                                  # Kết quả của các lần train/evaluate, có thể rất lớn
│   ├── segmentation_yolov11_full_finetune/       # Experiment cho Module 1: YOLOv11-Seg full fine-tune
│   │   ├── checkpoints/                          # Checkpoint từng lần train: best.pt, last.pt, epoch_x.pt
│   │   ├── logs/                                 # Log train/evaluate: loss, warning, runtime
│   │   ├── metrics/                              # Kết quả metric đã tính: Instance Recall, Merge Error Rate
│   │   ├── plots/                                # Biểu đồ loss/metric/threshold để đưa vào báo cáo
│   │   └── predictions/                          # Ảnh/mask dự đoán mẫu để phân tích lỗi
│   ├── attribute_resnet18_head_tune/             # Experiment cho Module 2a: train head ResNet18
│   │   ├── checkpoints/                          # Checkpoint trong giai đoạn freeze backbone, train head
│   │   ├── logs/                                 # Log train/evaluate head-tune
│   │   ├── metrics/                              # Macro F1, loss curve, confusion matrix nếu cần
│   │   ├── plots/                                # Biểu đồ loss, Macro F1, confusion matrix
│   │   └── predictions/                          # Sample dự đoán shape/color của Module 2
│   ├── attribute_resnet18_last_blocks_finetune/  # Experiment cho Module 2b: fine-tune last blocks ResNet18
│   │   ├── checkpoints/                          # Checkpoint sau khi unfreeze last blocks
│   │   ├── logs/                                 # Log fine-tune với learning rate nhỏ
│   │   ├── metrics/                              # So sánh metric với bản head-tune
│   │   ├── plots/                                # Biểu đồ so sánh head-tune và last-blocks
│   │   └── predictions/                          # Sample dự đoán sau fine-tune
│   └── ocr_paddleocr_baseline/                   # Experiment cho Module 3: PaddleOCR baseline
│       ├── logs/                                 # Log chạy OCR/evaluate OCR
│       ├── metrics/                              # Candidate Recall@5, Character Error Rate
│       ├── predictions/                          # OCR raw text và normalized imprint candidates
│       └── error_cases/                          # Các case OCR sai để xem lỗi 0/O, 1/I, 5/S...
│
├── inference/                                    # Script chạy dự đoán thật, KHÔNG chứa training logic
│   ├── cv_segmentation/                          # Entrypoint load YOLOv11-Seg và xuất bbox/mask/crop
│   ├── cv_attribute/                             # Entrypoint load ResNet18 và dự đoán attribute
│   ├── cv_ocr/                                   # Entrypoint chạy PaddleOCR và OCR normalization
│   ├── cv_pipeline/                              # run_full_cv_pipeline.py chạy toàn bộ CV; run_cv_pipeline.py chỉ ghép JSON có sẵn
│   └── rag_retrieval/                            # Entrypoint retrieval, ranking, safety gate và DDI lookup
│
├── models/                                       # Weight chính thức dùng cho inference, copy từ checkpoint tốt nhất
│   ├── segmentation_yolov11_full_finetune/       # Weight YOLOv11-Seg được chọn để chạy inference
│   ├── attribute_resnet18_head_tune/             # Weight ResNet18 head-tune nếu dùng baseline
│   ├── attribute_resnet18_last_blocks_finetune/  # Weight ResNet18 fine-tune cuối cùng nếu tốt hơn baseline
│   └── ocr_paddleocr/                            # OCR model/config nếu có custom hoặc fine-tune
│
├── outputs/                                      # Kết quả sinh ra khi chạy inference/demo, có thể xóa và tạo lại
│   ├── crops/                                    # Crop từng viên thuốc từ ảnh người dùng
│   ├── masks/                                    # Mask segmentation dự đoán
│   ├── predictions/                              # JSON/ảnh minh họa prediction cuối
│   └── reports/                                  # Báo cáo nhận diện thuốc và cảnh báo tương tác thuốc
│
├── src/                                          # Source code chính, nơi nên viết logic thật của hệ thống
│   └── pill_safety/                              # Python package chính của project
│       ├── cv/                                   # Logic thị giác máy tính
│       │   ├── segmentation/                     # Module 1: YOLOv11-Seg segmentation
│       │   │   ├── datasets/                     # Dataset loader/format adapter cho MEDISEG
│       │   │   ├── transforms/                   # Augmentation/resize/normalize dùng khi train/val
│       │   │   ├── models/                       # Wrapper load YOLOv11-Seg pretrained/fine-tuned
│       │   │   ├── trainers/                     # Logic train/full fine-tune, optimizer, scheduler
│       │   │   ├── evaluators/                   # Logic tính metric segmentation trên val/test
│       │   │   ├── predictors/                   # Logic predict mask/bbox cho ảnh mới
│       │   │   ├── postprocessing/               # Lọc mask, tách instance, crop từng viên thuốc
│       │   │   └── utils/                        # Helper riêng cho segmentation
│       │   ├── attribute/                        # Module 2: ResNet18 attribute recognition
│       │   │   ├── datasets/                     # Dataset loader cho NIH/RxImage crop
│       │   │   ├── transforms/                   # Augmentation/sim2real và normalize input
│       │   │   ├── models/                       # ResNet18 hai head cho shape/color; scoreline do OCR phụ trách
│       │   │   ├── trainers/                     # Logic head-tune và last-blocks fine-tune
│       │   │   ├── evaluators/                   # Logic tính Macro F1/accuracy theo từng attribute
│       │   │   ├── predictors/                   # Logic predict attribute cho một crop thuốc
│       │   │   ├── postprocessing/               # Chuẩn hóa top-k attribute output sang JSON
│       │   │   ├── labels/                       # Label mapping và label normalization
│       │   │   └── utils/                        # Helper riêng cho attribute
│       │   ├── ocr/                              # Module 3: imprint OCR
│       │   │   ├── preprocessing/                # CLAHE, threshold, deglare, rotate/multi-angle crop
│       │   │   ├── engines/                      # PaddleOCR/LLM OCR wrapper nếu thử nghiệm
│       │   │   ├── correction/                   # OCR candidate correction: O->0, I->1, fuzzy rules
│       │   │   ├── evaluators/                   # Logic tính CER và Candidate Recall@K
│       │   │   ├── predictors/                   # Logic OCR một crop hoặc nhiều rotation
│       │   │   ├── postprocessing/               # Gộp OCR observations thành candidate list
│       │   │   └── utils/                        # Helper riêng cho OCR
│       │   └── pipeline/                         # CV pipeline xuất structured visual metadata JSON
│       │       ├── orchestration/                 # Điều phối segmentation -> attribute -> OCR
│       │       ├── fusion/                        # Gộp evidence từ mask, attribute, OCR
│       │       ├── calibration/                   # Calibrate score/confidence trước khi đưa sang RAG
│       │       └── quality/                       # Blur/glare/occlusion quality flags
│       ├── rag/                                  # Logic retrieval, ranking, DDI và report
│       │   ├── retrieval/                        # Imprint-first search, fuzzy matching, candidate query
│       │   ├── ranking/                          # Feature scoring và final candidate ranking
│       │   ├── safety/                           # Safety gate: identified/ambiguous/unknown
│       │   ├── ddi/                              # Ingredient mapping, DDI lookup, duplicate ingredient check
│       │   └── reporting/                        # Context builder và grounded report formatter
│       ├── database/                             # DB connection, repository/query layer
│       ├── schemas/                              # Pydantic schemas cho CV output, RAG input/output, report
│       └── utils/                                # Logging, path utils, image utils, common helpers
│
├── scripts/                                      # Các công cụ hỗ trợ độc lập, tác vụ chuẩn bị dữ liệu một lần
│   └── resplit_by_ndc.py                         # Tool chia lại tập data NIH theo NDC để chống leakage
│
├── tests/                                        # Unit test và integration test
│   ├── cv/                                       # Test segmentation/attribute/OCR/CV schema
│   ├── rag/                                      # Test retrieval, ranking, safety gate, DDI
│   └── integration/                              # Test end-to-end từ ảnh đầu vào đến report
│
├── training/                                     # Script train/evaluate, gọi lại logic trong src/
│   ├── segmentation_yolov11_full_finetune/       # Module 1: full fine-tune YOLOv11-Seg trên MEDISEG
│   │   ├── data_preparation/                     # Script convert/check/split data, KHÔNG train model
│   │   ├── augmentation/                         # Script sinh augmentation, chỉ áp dụng train split
│   │   ├── train/                                # Script chạy full fine-tune YOLOv11-Seg
│   │   └── evaluation/                           # Code tính metric segmentation, output lưu ở experiments/.../metrics/
│   ├── attribute_resnet18_head_tune/             # Module 2a: freeze backbone, train head ResNet18
│   │   ├── data_preparation/                     # Script normalize label, crop, anti-leakage split NIH/RxImage
│   │   ├── augmentation/                         # Script augmentation nhẹ cho attribute
│   │   ├── train/                                # Script train classification heads
│   │   └── evaluation/                           # Code tính Macro F1, output lưu ở experiments/.../metrics/
│   ├── attribute_resnet18_last_blocks_finetune/  # Module 2b: unfreeze last blocks ResNet18
│   │   ├── train/                                # Script fine-tune last blocks với learning rate nhỏ
│   │   └── evaluation/                           # Code so sánh head-tune vs last-blocks fine-tune
│   └── ocr_paddleocr_baseline/                   # Module 3: OCR baseline bằng PaddleOCR
│       ├── preprocessing/                        # Script thử CLAHE, gamma, threshold, multi-angle OCR
│       └── evaluation/                           # Code tính Recall@5/CER, output lưu ở experiments/.../metrics/
│
├── .gitignore                                    # Rule ignore data lớn, model weight, log và output runtime
├── README.md                                     # Tài liệu cấu trúc project
└── requirements.txt                              # Python dependencies
```

## Tài Liệu Chính

| File | Nội dung |
|---|---|
| `docs/Overview.md` | Tổng quan bài toán, mục tiêu, phạm vi và giá trị ứng dụng của đề tài. |
| `docs/CV_Module.md` | Thiết kế CV pipeline: segmentation, attribute recognition, imprint OCR và CV output. |
| `docs/RAG_Module.md` | Thiết kế retrieval/ranking, database thuốc, mapping hoạt chất, DDI lookup và report. |
| `docs/metric_CV.md` | Metric cho segmentation, attribute recognition, OCR và CV output. |
| `docs/metric_LLM.md` | Metric cho identification, unknown rejection, ranking và safety gate. |
| `docs/train_request.md` | Quy định log, metric, checkpoint và evidence cần lưu cho 3 training job hiện tại. |
| `docs/schema.md` | Contract input/output giữa các module để các team làm song song không conflict. |

## Quy Ước Làm Việc

| Thành phần | Quy ước |
|---|---|
| `src/` | Viết logic chính ở đây để training, inference và tests import chung. |
| `training/` | Chỉ viết script chạy một task cụ thể: prepare data, augment, train, evaluate. |
| `inference/` | Chỉ viết script chạy dự đoán ảnh mới; không viết lại logic đã có trong `src/`. |
| `scripts/` | Chứa code công cụ chạy một lần, dọn dẹp hệ thống, chia data... không tham gia logic chính. |
| `evaluation/` | Là code tính điểm, không phải nơi lưu điểm. |
| `experiments/.../metrics/` | Là nơi lưu điểm đã tính ra từ evaluation. |
| `experiments/.../checkpoints/` | Lưu checkpoint theo từng lần train, có thể có nhiều file. |
| `models/` | Chỉ lưu model/weight tốt nhất đã được chọn để dùng inference. |
| `outputs/` | Lưu kết quả runtime khi chạy inference/demo. |
| `docs/` | Lưu tài liệu thiết kế, paper summary, metric, report, slide và tài liệu nộp đồ án. |

## Luồng Làm Việc Dự Kiến

```text
training/
  -> gọi logic trong src/
  -> chuẩn bị dữ liệu
  -> train/evaluate model
  -> lưu logs/checkpoints/metrics vào experiments/
  -> copy weight tốt nhất sang models/

inference/
  -> gọi logic trong src/
  -> load weight từ models/
  -> xử lý ảnh mới
  -> xuất crops/masks/predictions/reports vào outputs/

tests/
  -> gọi logic trong src/
  -> kiểm tra từng module và end-to-end pipeline
```

## Ghi Chú Về Checkpoint

Không tạo folder `checkpoint/` ở root. Checkpoint phải nằm trong đúng experiment tương ứng:

```text
experiments/<module_name>/checkpoints/
```

Ví dụ:

```text
experiments/segmentation_yolov11_full_finetune/checkpoints/best.pt
experiments/attribute_resnet18_head_tune/checkpoints/best.pt
```

Sau khi chọn được weight tốt nhất, copy weight đó sang `models/` để inference dùng.

## Ghi Chú Về Dữ Liệu Và Model

Các thư mục `data/`, `models/`, `experiments/` và `outputs/` có thể rất lớn. Khi dùng Git, nên chỉ commit cấu trúc, config mẫu hoặc sample nhỏ; dữ liệu, checkpoint và weight lớn nên được quản lý riêng.

`.gitkeep` được dùng để giữ các folder rỗng trên Git. Không xóa `.gitkeep` cho đến khi folder đó đã có file thật cần commit.

`.gitignore` ở root project đang ignore dữ liệu lớn, checkpoint, model weight, log và output runtime; code, config, schema, docs và test vẫn nên commit bình thường.

# Ghi chú về chỉnh sửa folder data
Thêm folder image_all vào để tiện trong việc train model

```
Multiple-Pill-Recognition-And-Interaction-Safety/
│
└── data/                                 
    │
    ├── image_all/                         # Toàn bộ ảnh dùng cho train, test, validation
    │   └── nih_attribute/
    │                   
    ├── augmented/                       
    │
    ├── benchmark/                       
    │
    ├── processed/                       
    │
    ├── raw/                              
    │
    └── splits/                                     
```
