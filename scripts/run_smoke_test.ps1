# PowerShell script to run smoke test for generating fresh artifacts
# This addresses Error 4 (stale artifacts) and Error 5 (missing last.pt) in failed.md

$ErrorActionPreference = "Stop"

# Ensure we are in the project root
$PROJECT_ROOT = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location -Path $PROJECT_ROOT

Write-Host "Starting Head-Tune Smoke Test (1 epoch) to generate attr_head_v2 artifacts..." -ForegroundColor Cyan

# Run head tune with 1 epoch
python training/attribute_resnet18_head_tune/train/run_head_train.py --run_id attr_head_v2 --epochs 1

if ($LASTEXITCODE -eq 0) {
    Write-Host "Head-tune completed successfully." -ForegroundColor Green
    Write-Host "Check experiments/attribute_resnet18_head_tune/checkpoints/ for best.pt and last.pt" -ForegroundColor Green
} else {
    Write-Host "Head-tune failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Starting Last-Blocks Smoke Test (1 epoch) to generate attr_last_v2 artifacts..." -ForegroundColor Cyan

# Run last blocks with 1 epoch, depending on the head-tune we just created
python training/attribute_resnet18_last_blocks_finetune/train/train_last_blocks.py --run_id attr_last_v2 --head_run_id attr_head_v2 --epochs 1

if ($LASTEXITCODE -eq 0) {
    Write-Host "Last-blocks finetune completed successfully." -ForegroundColor Green
    Write-Host "All new artifacts have been generated!" -ForegroundColor Green
} else {
    Write-Host "Last-blocks finetune failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
