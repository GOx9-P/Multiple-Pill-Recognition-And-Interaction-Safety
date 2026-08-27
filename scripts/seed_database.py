#!/usr/bin/env python3
"""Script tiện ích chạy seed database SQLite từ database_seed/."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pill_safety.database.base import Base
from pill_safety.database.session import SessionLocal, engine
from pill_safety.database.scripts.seed import seed_database

if __name__ == "__main__":
    print(f"-> Đang khởi tạo bảng SQLite trên: {engine.url}...")
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        seed_database(db)
    db_file_name = engine.url.database or "pill_safety.db"
    db_path = PROJECT_ROOT / db_file_name
    print(f"✓ ĐÃ SEED DATABASE THÀNH CÔNG!")
    print(f"✓ File database sẵn sàng tại: {db_path}")
