# Data Directory

Thư mục này chứa dữ liệu huấn luyện và đánh giá. **Dữ liệu không được push lên GitHub** do kích thước quá lớn.

## Tải dữ liệu

### NIH/RxImage Dataset (dùng cho attribute recognition)

**Kaggle Dataset:** [rximage_new](https://www.kaggle.com/datasets/) *(cập nhật link chính xác)*

```bash
# Cách 1: Tải từ Kaggle CLI
pip install kaggle
kaggle datasets download -d <username>/rximage-new -p data/raw/nih_rximage/

# Cách 2: Trên Kaggle Notebook
# Dataset đã được mount tự động tại: /kaggle/input/rximage-new/rximage/
```

### Cấu trúc dữ liệu sau khi tải

```
data/raw/nih_rximage/
└── rximage/
    ├── combined/
    │   ├── train_combined_crop.csv
    │   ├── val_combined_crop.csv
    │   └── test_combined_crop.csv
    └── image_all/
        ├── <image_id>.png
        └── ...
```

## Lưu ý

- **KHÔNG commit dữ liệu lên Git** — file `.gitignore` đã chặn `data/raw/**`, `data/processed/**`
- Trên Kaggle, dữ liệu được mount tự động qua **Add Input** → không cần tải thủ công
- Script training (`run_head_train.py`) tự động tìm đường dẫn data trên cả máy local và Kaggle
