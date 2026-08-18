$ErrorActionPreference = "Stop"

Write-Host "CANH BAO: lenh nay se xoa toan bo Docker volume va du lieu PostgreSQL local."
$confirmation = Read-Host "Nhap RESET de tiep tuc"

if ($confirmation -ne "RESET") {
    Write-Host "Da huy reset database."
    exit 0
}

$env:PYTHONPATH = "src"

docker compose down -v
docker compose up -d
alembic upgrade head
python -m pill_safety.database.scripts.seed

Write-Host "Da reset va seed lai database local."

