$ErrorActionPreference = "Stop"

$env:PYTHONPATH = "src"

docker compose up -d
alembic upgrade head
python -m pill_safety.database.scripts.seed

Write-Host ""
Write-Host "Database da san sang."
Write-Host "Chay FastAPI bang lenh:"
Write-Host '$env:PYTHONPATH = "src"; uvicorn pill_safety.api.main:app --reload'

