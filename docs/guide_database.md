# Hướng Dẫn Thiết Lập và Quản Trị Database (PostgreSQL + Alembic + SQLAlchemy)

Tài liệu này cung cấp hướng dẫn chi tiết từng bước để thiết lập, chạy migration, nạp dữ liệu (seed) và kiểm tra cơ sở dữ liệu của dự án. Hướng dẫn được thiết kế để bất kỳ ai cũng có thể thực hiện dễ dàng thông qua mọi loại terminal (Command Prompt, PowerShell, Bash).

---

## 1. Thành Phần Hệ Thống và Vị Trí File

| Thành phần | Đường dẫn / Vị trí | Vai trò |
|---|---|---|
| **Hạ tầng Container** | [`compose.yaml`](file:///c:/TranNhatTruong_2026/other/MLIOT_Lab/final_project/code_main/Multiple-Pill-Recognition-And-Interaction-Safety/compose.yaml) | Chạy PostgreSQL 16 Alpine trong Docker |
| **Cấu hình môi trường** | [`.env.example`](file:///c:/TranNhatTruong_2026/other/MLIOT_Lab/final_project/code_main/Multiple-Pill-Recognition-And-Interaction-Safety/.env.example) | Mẫu khai báo các biến môi trường |
| **Cấu hình Alembic** | [`alembic.ini`](file:///c:/TranNhatTruong_2026/other/MLIOT_Lab/final_project/code_main/Multiple-Pill-Recognition-And-Interaction-Safety/alembic.ini) | Chỉ định kết nối và cài đặt tool migration |
| **Các bản Migration** | [`database/migrations/`](file:///c:/TranNhatTruong_2026/other/MLIOT_Lab/final_project/code_main/Multiple-Pill-Recognition-And-Interaction-Safety/database/migrations) | Chứa lịch sử các file thay đổi cấu trúc bảng |
| **SQLAlchemy models** | [`src/pill_safety/database/models/`](file:///c:/TranNhatTruong_2026/other/MLIOT_Lab/final_project/code_main/Multiple-Pill-Recognition-And-Interaction-Safety/src/pill_safety/database/models) | Định nghĩa cấu trúc bảng dưới dạng Class Python |
| **DB Session** | [`src/pill_safety/database/session.py`](file:///c:/TranNhatTruong_2026/other/MLIOT_Lab/final_project/code_main/Multiple-Pill-Recognition-And-Interaction-Safety/src/pill_safety/database/session.py) | Quản lý kết nối và dependency cho FastAPI |
| **Script Seed** | [`src/pill_safety/database/scripts/seed.py`](file:///c:/TranNhatTruong_2026/other/MLIOT_Lab/final_project/code_main/Multiple-Pill-Recognition-And-Interaction-Safety/src/pill_safety/database/scripts/seed.py) | Đọc file JSON nạp dữ liệu thông qua SQLAlchemy |
| **Dữ liệu Seed (JSON)**| [`database_seed/`](file:///c:/TranNhatTruong_2026/other/MLIOT_Lab/final_project/code_main/Multiple-Pill-Recognition-And-Interaction-Safety/database_seed) | Thư mục chứa các file JSON dữ liệu mẫu của hệ thống |
| **Tài liệu Schema** | [`database_seed/db.md`](file:///c:/TranNhatTruong_2026/other/MLIOT_Lab/final_project/code_main/Multiple-Pill-Recognition-And-Interaction-Safety/database_seed/db.md) | Từ điển dữ liệu, lược đồ quan hệ thực thể (ERD) |
| **FastAPI App** | [`src/pill_safety/api/main.py`](file:///c:/TranNhatTruong_2026/other/MLIOT_Lab/final_project/code_main/Multiple-Pill-Recognition-And-Interaction-Safety/src/pill_safety/api/main.py) | Entrypoint của Web API |

---

## 2. Quy Trình Thiết Lập Từng Bước (Step-by-Step)

Mở terminal tại thư mục gốc của dự án (`Multiple-Pill-Recognition-And-Interaction-Safety`) và thực hiện các bước sau:

### Bước 1: Khởi tạo và Kích hoạt Môi trường Ảo (Virtual Environment)
Đảm bảo bạn đang sử dụng Python phiên bản **3.11.9** (hoặc 3.11.x).

* **Trên Windows (Command Prompt - CMD):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **Trên Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **Trên macOS / Linux / Git Bash:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Bước 2: Cài đặt các thư viện cần thiết
Nâng cấp `pip` và cài đặt các dependencies từ file `requirements.txt`:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 3: Thiết lập File biến môi trường `.env`
Sao chép cấu hình mẫu từ `.env.example` thành `.env`:

* **Trên Windows (CMD):**
  ```cmd
  copy .env.example .env
  ```
* **Trên Windows (PowerShell):**
  ```powershell
  Copy-Item .env.example .env
  ```
* **Trên macOS / Linux / Git Bash:**
  ```bash
  cp .env.example .env
  ```

> [!NOTE]
> Mặc định, file `.env` chứa mật khẩu kết nối `change_me` và cổng kết nối mặc định `5432`. Bạn có thể chỉnh sửa file `.env` này nếu cần cấu hình cổng kết nối khác (ví dụ: `5433` nếu máy của bạn đã chạy sẵn một dịch vụ Postgres khác).

### Bước 4: Khởi động PostgreSQL qua Docker
Đảm bảo phần mềm **Docker Desktop** đã được mở trước khi chạy lệnh:
```bash
docker compose up -d
```
Kiểm tra trạng thái container:
```bash
docker compose ps
```
> [!IMPORTANT]
> Chờ khoảng 5-10 giây để cơ sở dữ liệu PostgreSQL khởi động hoàn tất và hiển thị trạng thái `healthy` trước khi thực hiện bước tiếp theo.

### Bước 5: Chạy các bản Migration (Tạo bảng tự động)
Để Alembic hiểu được cấu trúc code trong thư mục `src/`, ta phải thiết lập biến môi trường `PYTHONPATH` trỏ vào thư mục `src` trước khi chạy lệnh:

* **Trên Windows (CMD):**
  ```cmd
  set PYTHONPATH=src
  alembic upgrade head
  ```
* **Trên Windows (PowerShell):**
  ```powershell
  $env:PYTHONPATH = "src"
  alembic upgrade head
  ```
* **Trên macOS / Linux / Git Bash:**
  ```bash
  export PYTHONPATH=src
  alembic upgrade head
  ```

### Bước 6: Nạp dữ liệu mẫu (Database Seeding)
Do cấu trúc import của script seed phụ thuộc vào package `src/`, ta cần chạy script này dưới dạng module của Python:

* **Trên Windows (CMD):**
  ```cmd
  set PYTHONPATH=src
  python -m pill_safety.database.scripts.seed
  ```
* **Trên Windows (PowerShell):**
  ```powershell
  $env:PYTHONPATH = "src"
  python -m pill_safety.database.scripts.seed
  ```
* **Trên macOS / Linux / Git Bash:**
  ```bash
  export PYTHONPATH=src
  python3 -m pill_safety.database.scripts.seed
  ```

> [!TIP]
> **Giải pháp một dòng lệnh (Shell-Agnostic):** Nếu bạn không muốn thiết lập biến môi trường `PYTHONPATH` thủ công, bạn có thể chạy lệnh sau (hoạt động tốt trên tất cả các loại terminal):
> ```bash
> python -c "import sys; sys.path.append('src'); from pill_safety.database.scripts.seed import seed_database; seed_database()"
> ```

---

## 3. Khởi Chạy Web API (FastAPI)

Sau khi thiết lập và nạp dữ liệu DB thành công, khởi động FastAPI server bằng lệnh:

* **Trên Windows (CMD):**
  ```cmd
  set PYTHONPATH=src
  uvicorn pill_safety.api.main:app --reload
  ```
* **Trên Windows (PowerShell):**
  ```powershell
  $env:PYTHONPATH = "src"
  uvicorn pill_safety.api.main:app --reload
  ```
* **Trên macOS / Linux / Git Bash:**
  ```bash
  export PYTHONPATH=src
  uvicorn pill_safety.api.main:app --reload
  ```

Sau khi server chạy, truy cập các đường dẫn sau trên trình duyệt:
* Trang chủ API: [http://localhost:8000/](http://localhost:8000/)
* Tài liệu tương tác Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
* Kiểm tra kết nối Database: [http://localhost:8000/health/database](http://localhost:8000/health/database) (Kỳ vọng trả về `{"database": "connected"}`)

---

## 4. Chạy nhanh bằng File Script Tự Động (Tùy chọn)

Nếu bạn sử dụng **PowerShell** trên Windows, bạn có thể sử dụng các file script tự động hóa được thiết kế sẵn:

1. **Thiết lập nhanh toàn bộ hệ thống lần đầu:**
   ```powershell
   .\scripts\setup.ps1
   ```
   *Lệnh này sẽ tự động: Chạy Docker Compose $\rightarrow$ Thiết lập PYTHONPATH $\rightarrow$ Chạy migration nâng cấp $\rightarrow$ Thực hiện nạp dữ liệu mẫu.*

2. **Xóa sạch và đặt lại (Reset) Database:**
   Khi muốn reset cơ sở dữ liệu về trạng thái ban đầu (xóa sạch toàn bộ dữ liệu hiện tại, bao gồm cả dữ liệu scan và bệnh nhân):
   ```powershell
   .\scripts\reset_database.ps1
   ```
   *Nhập chữ `RESET` viết hoa khi script yêu cầu xác nhận.*

---

## 5. Truy Vấn Trực Tiếp Dữ Liệu Trong PostgreSQL (psql CLI)

Bạn có thể đăng nhập trực tiếp vào hệ quản trị cơ sở dữ liệu PostgreSQL bên trong Docker để xem cấu trúc và dữ liệu của các bảng:

1. **Đăng nhập vào psql:**
   ```bash
   docker compose exec postgres psql -U medication_user -d medication_db
   ```
2. **Một số lệnh psql thông dụng:**
   * Liệt kê danh sách bảng: `\dt`
   * Xem cấu trúc chi tiết của một bảng: `\d drug_products` hoặc `\d drug_interactions`
   * Đếm số lượng bản ghi của tất cả các bảng (Copy & Paste vào terminal):
     ```sql
     SELECT 'drug_products' AS table, count(*) FROM drug_products
     UNION ALL SELECT 'ingredients', count(*) FROM ingredients
     UNION ALL SELECT 'drug_appearances', count(*) FROM drug_appearances
     UNION ALL SELECT 'product_ingredients', count(*) FROM product_ingredients
     UNION ALL SELECT 'drug_interactions', count(*) FROM drug_interactions
     UNION ALL SELECT 'patient_profiles', count(*) FROM patient_profiles
     UNION ALL SELECT 'scan_sessions', count(*) FROM scan_sessions
     UNION ALL SELECT 'scan_items', count(*) FROM scan_items
     UNION ALL SELECT 'scan_interaction_results', count(*) FROM scan_interaction_results
     ORDER BY table;
     ```
   * Thoát giao diện psql: `\q`

---

## 6. Phát Triển Schema (Thêm/Sửa Bảng và Cột)

Khi bạn cần chỉnh sửa hoặc thêm mới các trường thông tin trong cơ sở dữ liệu:

1. Cập nhật các Class Model tương ứng tại thư mục [`src/pill_safety/database/models/`](file:///c:/TranNhatTruong_2026/other/MLIOT_Lab/final_project/code_main/Multiple-Pill-Recognition-And-Interaction-Safety/src/pill_safety/database/models).
2. Tạo file migration tự động sinh bằng Alembic (Nhớ cấu hình `PYTHONPATH` trước):
   ```bash
   # Ví dụ trên CMD
   set PYTHONPATH=src
   python -m alembic revision --autogenerate -m "them_cot_abc_vao_bang_xyz"
   ```
3. Mở file migration vừa sinh ra tại [`database/migrations/versions/`](file:///c:/TranNhatTruong_2026/other/MLIOT_Lab/final_project/code_main/Multiple-Pill-Recognition-And-Interaction-Safety/database/migrations/versions) để kiểm tra lại code xem đã chính xác chưa.
4. Thực thi cập nhật cơ sở dữ liệu:
   ```bash
   alembic upgrade head
   ```
5. Cập nhật sơ đồ lược đồ trong tài liệu [`database_seed/db.md`](file:///c:/TranNhatTruong_2026/other/MLIOT_Lab/final_project/code_main/Multiple-Pill-Recognition-And-Interaction-Safety/database_seed/db.md).

---

## 7. Xử Lý Sự Cố (Troubleshooting)

### Lỗi: `ModuleNotFoundError: No module named 'pill_safety'`
* **Nguyên nhân**: Python không tìm thấy package của dự án do biến môi trường `PYTHONPATH` chưa được thiết lập chính xác trên terminal hiện tại của bạn.
* **Cách xử lý**: Đảm bảo bạn đã gõ đúng lệnh thiết lập `PYTHONPATH` tương ứng với shell bạn đang dùng (Xem lại mục 2, Bước 5 & 6) hoặc dùng lệnh một dòng ở mục TIP.

### Lỗi: `failed to connect to the docker API`
* **Nguyên nhân**: Docker Desktop chưa chạy hoặc engine của Docker chưa sẵn sàng.
* **Cách xử lý**: Hãy mở Docker Desktop và đợi đến khi biểu tượng góc trái màn hình chuyển sang màu xanh lá báo hiệu "Engine Running", sau đó chạy lại lệnh `docker compose up -d`.

### Lỗi: `Port 5432 is already in use` (Cổng 5432 bị chiếm dụng)
* **Nguyên nhân**: Máy tính của bạn đã cài đặt và chạy sẵn một cơ sở dữ liệu PostgreSQL local ngoài Docker.
* **Cách xử lý**: 
  1. Mở file `.env`.
  2. Thay đổi cổng kết nối `POSTGRES_PORT` sang `5433` (hoặc cổng bất kỳ còn trống):
     ```env
     POSTGRES_PORT=5433
     DATABASE_URL=postgresql+psycopg://medication_user:change_me@localhost:5433/medication_db
     ```
  3. Khởi động lại docker compose: `docker compose up -d`.
