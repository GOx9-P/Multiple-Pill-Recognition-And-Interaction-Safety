# Deployment Notebook

Tài liệu này mô tả notebook `merger-fe-be.ipynb` dùng để chạy demo end-to-end trên Colab hoặc Kaggle sau khi đã merge phần RAG tuned parameters và UI.

Notebook là runner triển khai/demo. Nó không phải nơi định nghĩa lại thuật toán hoặc tune lại Safety Gate; nguồn sự thật của tham số RAG vẫn nằm trong `src/pill_safety/rag/ranking/safety_gate.py`.

---

## 1. Mục tiêu

Notebook tự động hóa 4 bước:

1. Clone source từ nhánh `tune_rag`.
2. Cài môi trường AI/CV/OCR với Paddle GPU và retry khi mạng không ổn định.
3. Tải model/database artifacts từ Kaggle và seed SQLite.
4. Chạy Streamlit UI ở port `8501` và mở Cloudflare public tunnel.

---

## 2. Runtime mục tiêu

Notebook tự nhận diện môi trường:

```python
work_dir = "/content" if os.path.exists("/content") else "/kaggle/working"
repo_dir = os.path.join(work_dir, "repo")
```

Runtime khuyến nghị:

| Nền tảng | Ghi chú |
|---|---|
| Google Colab | Ưu tiên GPU T4. |
| Kaggle Notebook | Ưu tiên accelerator GPU tương thích CUDA 11.8. |

Nếu Paddle GPU smoke test fail, notebook dừng rõ ràng thay vì tiếp tục chạy UI với OCR hỏng.

---

## 3. Clone source

Notebook clone đúng nhánh:

```text
tune_rag
```

Lệnh trong notebook:

```python
!git clone --depth 1 -b tune_rag https://github.com/GOx9-P/Multiple-Pill-Recognition-And-Interaction-Safety.git {repo_dir}
```

Ý nghĩa:

- Giữ các tuned parameters của RAG/Safety Gate.
- Đảm bảo demo dùng code mới nhất của nhánh đang chứa thuật toán định danh đã tune.
- Không clone `FE_Final` trực tiếp trong notebook, vì UI đã được merge vào working tree/nhánh đích khi resolve conflict.

---

## 4. Cài dependency

Notebook dùng helper:

```python
install_with_retry(args, label, attempts=3)
```

Hành vi:

- Gọi `python -m pip install`.
- Dùng `--no-cache-dir`, `--timeout 120`, `--retries 4`.
- Retry theo backoff đơn giản `10 * attempt` giây.
- Sau số lần thử tối đa, raise `RuntimeError` với label rõ ràng.

Trình tự cài:

1. Gỡ các package Paddle cũ:

```text
paddlepaddle
paddlepaddle-gpu
paddleocr
paddlex
```

2. Đọc `requirements.txt`, loại `paddleocr` và `paddlex` khỏi file tạm `/tmp/requirements_without_paddle.txt`.

3. Cài requirements còn lại với `attempts=2`.

4. Cài Paddle GPU:

```text
paddlepaddle-gpu==3.0.0
--index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

5. Cài OCR stack:

```text
paddleocr==3.0.3
paddlex==3.0.3
numpy==1.26.4
opencv-python-headless==4.10.0.84
```

6. Chạy smoke test:

```python
import paddle, paddleocr, paddlex
paddle.set_device("gpu:0")
print("Paddle GPU ready:", paddle.get_device())
```

---

## 5. Artifacts tải từ Kaggle

Notebook tải 3 nhóm artifact:

| Nhóm | Kaggle dataset | Đích trong repo |
|---|---|---|
| YOLOv11 segmentation model | `nnphuchcmus/pill-segmentation-model` | `models/segmentation_yolov11_full_finetune/yolov11m_seg_mediseg_full_finetune_v1.pt` |
| ResNet-18 attribute artifacts | `nnphuchcmus/attrubute-artifact` | `models/attribute_resnet18_last_blocks_finetune/` |
| Database seed / SQLite | `trannhattruong19691/database-mliotlab` | `database_seed/*.json` hoặc `medication.db` |

Nếu tải segmentation model qua `kagglehub` lỗi, notebook fallback sang scan `/kaggle/input/**/*.pt` và copy file có tên chứa `seg` hoặc `yolo`.

Sau khi tải database, notebook ghi `.env`:

```text
DATABASE_URL=sqlite:///./medication.db
LLM_PROVIDER=fallback
```

---

## 6. Seed database

Notebook thêm `src/` và repo root vào `sys.path`, set `PYTHONPATH`, rồi chạy:

```python
!python scripts/seed_database.py
```

Sau seed, notebook mở `SessionLocal()` và kiểm tra:

```python
total_drugs = db.query(DrugProduct).count()
total_ddi = db.query(DrugInteraction).count()
```

Hai số này là sanity check để biết database đã có sản phẩm thuốc và cặp DDI trước khi mở UI.

---

## 7. Chạy Web & Mobile UI

Notebook dọn port cũ:

```bash
fuser -k 8501/tcp
```

Sau đó cài `cloudflared` nếu chưa có và chạy Streamlit:

```bash
streamlit run app.py \
  --server.port 8501 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.enableWebsocketCompression false \
  --server.fileWatcherType none
```

Cuối cùng mở tunnel:

```bash
cloudflared tunnel --url http://localhost:8501
```

Đường dẫn public có dạng:

```text
https://*.trycloudflare.com
```

---

## 8. Những điểm cần giữ khi cập nhật notebook

- Không đổi branch clone khỏi `tune_rag` nếu mục tiêu là giữ tuned RAG parameters.
- Không cài PaddleOCR trực tiếp từ `requirements.txt` trước Paddle GPU; notebook cố ý loại `paddleocr/paddlex` khỏi requirements tạm để tránh xung đột.
- Không bỏ GPU smoke test. Nếu OCR GPU không sẵn sàng, demo có thể nhìn như chạy được nhưng kết quả OCR sai hoặc rất chậm.
- Không dùng kết quả `no_interaction_found` như kết luận an toàn tuyệt đối. UI/report chỉ nói không tìm thấy tương tác trong database hiện tại giữa các thuốc đã định danh.
- Nếu cập nhật tên Kaggle dataset hoặc tên weight, phải cập nhật cả file này và các path trong notebook.
