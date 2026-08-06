"""
Script đẩy kết quả training từ Kaggle lên GitHub.

Chạy script này trong một cell của Kaggle Notebook SAU KHI train xong.
Yêu cầu: Thêm GitHub Token vào Kaggle Secrets với key "GITHUB_TOKEN".

Hướng dẫn tạo GitHub Token:
1. Vào https://github.com/settings/tokens → "Generate new token (classic)"
2. Chọn scope: repo (full control)
3. Copy token
4. Trong Kaggle notebook: Add-ons → Secrets → Add → Key: GITHUB_TOKEN, Value: <paste token>
"""

import os
import subprocess
import shutil
from pathlib import Path

# ==============================================================================
# CẤU HÌNH — SỬA CÁC BIẾN NÀY CHO PHÙ HỢP
# ==============================================================================
GITHUB_USERNAME = "GOx9-P"
GITHUB_REPO = "Multiple-Pill-Recognition-And-Interaction-Safety"
BRANCH = "CV-attribute_resnet18_head_finetune-NguyenQuocBao"
COMMIT_MESSAGE = "feat: add head-tune experiment results from Kaggle"

# Thư mục chứa kết quả training trên Kaggle
# (Sửa nếu EXPERIMENT_DIR trong run_head_train.py khác)
KAGGLE_EXPERIMENT_DIR = Path("/kaggle/working/experiments/attribute_resnet18_head_tune")

# ==============================================================================
# BƯỚC 1: Lấy GitHub Token từ Kaggle Secrets
# ==============================================================================
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    GITHUB_TOKEN = secrets.get_secret("GITHUB_TOKEN")
    print("✅ Đã lấy GITHUB_TOKEN từ Kaggle Secrets")
except Exception as e:
    print(f"❌ Không lấy được token: {e}")
    print("→ Vào Add-ons → Secrets → Add: Key='GITHUB_TOKEN', Value=<your_token>")
    raise

# ==============================================================================
# BƯỚC 2: Clone repo
# ==============================================================================
CLONE_DIR = Path("/kaggle/working/repo_push")
if CLONE_DIR.exists():
    shutil.rmtree(CLONE_DIR)

repo_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"

print(f"\n📥 Cloning branch '{BRANCH}'...")
subprocess.run(
    ["git", "clone", "--branch", BRANCH, "--single-branch", "--depth", "1", repo_url, str(CLONE_DIR)],
    check=True,
)
print("✅ Clone thành công")

# ==============================================================================
# BƯỚC 3: Copy kết quả training vào repo
# ==============================================================================
DEST_EXPERIMENT = CLONE_DIR / "experiments" / "attribute_resnet18_head_tune"

# Copy từng thư mục con
for subfolder in ["logs", "metrics", "plots", "predictions", "checkpoints"]:
    src = KAGGLE_EXPERIMENT_DIR / subfolder
    dst = DEST_EXPERIMENT / subfolder

    if not src.exists():
        print(f"  ⚠️  Bỏ qua {subfolder}/ (không tồn tại)")
        continue

    # Copy tất cả file (không đè .gitkeep)
    for f in src.rglob("*"):
        if f.is_file() and f.name != ".gitkeep":
            rel = f.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            print(f"  📄 {subfolder}/{rel}")

# Copy best checkpoint vào models/ (nếu muốn)
best_ckpt = KAGGLE_EXPERIMENT_DIR / "checkpoints" / "attr_head_v1_best.pt"
if best_ckpt.exists():
    models_dst = CLONE_DIR / "models" / "attribute_resnet18_head_tune"
    models_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_ckpt, models_dst / "attr_head_v1_best.pt")
    print(f"  📄 models/attribute_resnet18_head_tune/attr_head_v1_best.pt")

print("\n✅ Copy kết quả xong")

# ==============================================================================
# BƯỚC 4: Git add, commit, push
# ==============================================================================
os.chdir(CLONE_DIR)

# Config git
subprocess.run(["git", "config", "user.email", "kaggle-runner@users.noreply.github.com"], check=True)
subprocess.run(["git", "config", "user.name", "Kaggle Runner"], check=True)

# Force add (vượt qua .gitignore cho .pt và .log)
subprocess.run(["git", "add", "--force", "experiments/attribute_resnet18_head_tune/"], check=True)
subprocess.run(["git", "add", "--force", "models/attribute_resnet18_head_tune/"], check=True)

# Check status
result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
print(f"\n📋 Staged files:\n{result.stdout}")

if not result.stdout.strip():
    print("ℹ️  Không có thay đổi nào mới để commit.")
else:
    # Commit & Push
    subprocess.run(["git", "commit", "-m", COMMIT_MESSAGE], check=True)
    subprocess.run(["git", "push", "origin", BRANCH], check=True)
    print(f"\n🎉 Push thành công lên branch: {BRANCH}")
    print(f"🔗 https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}/tree/{BRANCH}")

# Cleanup
os.chdir("/kaggle/working")
print("\n✅ Hoàn tất!")
