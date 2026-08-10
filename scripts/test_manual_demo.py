from __future__ import annotations

import glob
import io
import json
import sys
from pathlib import Path

# Configure UTF-8 encoding for Windows stdout
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Add src to sys.path
project_root = Path(__file__).resolve().parents[1]
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import BigInteger

# Map BigInteger to INTEGER in SQLite so autoincrement works natively
@compiles(BigInteger, "sqlite")
def compile_big_int_sqlite(type_, compiler, **kw):
    return "INTEGER"


from pill_safety.api.main import app
from pill_safety.database.models import (
    DrugAppearance,
    DrugInteraction,
    DrugProduct,
    Ingredient,
    ProductIngredient,
)
from pill_safety.database.scripts.seed import (
    seed_drug_appearances,
    seed_drug_interactions,
    seed_drug_products,
    seed_ingredients,
    seed_product_ingredients,
)
from pill_safety.database.session import get_db


# Setup in-memory SQLite database populated with real JSON seed data using StaticPool
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
DrugProduct.__table__.create(engine)
DrugAppearance.__table__.create(engine)
Ingredient.__table__.create(engine)
ProductIngredient.__table__.create(engine)
DrugInteraction.__table__.create(engine)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
init_session = TestingSessionLocal()
seed_ingredients(init_session)
seed_drug_products(init_session)
init_session.flush()
seed_drug_appearances(init_session)
seed_product_ingredients(init_session)
seed_drug_interactions(init_session)
init_session.commit()
init_session.close()


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)



def run_demo() -> None:
    print("=" * 80)
    print("   DEMO KIỂM THỬ TỰ ĐỘNG CÁC TÍNH NĂNG PILL SAFETY QUA TERMINAL")
    print("=" * 80)

    # 1. Health check
    res = client.get("/")
    print(f"\n[1] Check Server Status: {res.json()}")

    # 2. Search Drug
    res = client.get("/drugs/search?imprint=TV5056")
    drugs = res.json()
    print(f"\n[2] Search Thuốc với Imprint 'TV5056': Tìm thấy {len(drugs)} sản phẩm.")
    if drugs:
        print(f"    -> Tên sản phẩm: {drugs[0].get('name')}")

    # 3. Test Pair Interaction
    res = client.get("/interactions/pair?ingredient_a_id=1&ingredient_b_id=2")
    if res.status_code == 200:
        inter = res.json()
        print(f"\n[3] Tra cứu tương tác cặp (Ingredient 1 vs 2): Severity = {inter.get('severity')}")

    # 4. Run all 4 scenario files in tests/rag/fakeoutputCV
    scenario_files = sorted(glob.glob("tests/rag/fakeoutputCV/*.json"))

    for filepath in scenario_files:
        print("\n" + "=" * 80)
        print(f"📌 ĐANG CHẠY KỊCH BẢN: {filepath}")
        print("=" * 80)
        with open(filepath, encoding="utf-8") as f:
            cv_data = json.load(f)

        res = client.post("/rag/report", json={"cv_output": cv_data})
        if res.status_code == 200:
            output = res.json()
            print(f"Overall Severity: {output.get('overall_severity')}")
            print(f"Provider Used: {output.get('provider_used')}\n")
            print(output.get("formatted_report_text"))
        else:
            print(f"Lỗi: {res.status_code} - {res.text}")

    # 5. Test Manual Override
    print("\n" + "=" * 80)
    print("📌 TEST TÍNH NĂNG NHẬP THUỐC THỦ CÔNG (/rag/manual-identify)")
    print("=" * 80)
    res = client.post("/rag/manual-identify", json={
        "session_id": "sess_scenario_unresolved_03",
        "instance_id": "pill_instance_202_unresolved",
        "manual_drug_name": "TV5056"
    })
    print(f"Kết quả ghép nối thủ công: {json.dumps(res.json(), ensure_ascii=False, indent=2)}")

    print("\n" + "=" * 80)
    print("   HOÀN THÀNH DEMO KIỂM THỬ TRÊN TERMINAL!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
