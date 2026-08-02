from __future__ import annotations

import importlib


def test_fastapi_app_imports_with_database_url(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")

    module = importlib.import_module("pill_safety.api.main")

    assert module.app.title == "Medication Safety API"
