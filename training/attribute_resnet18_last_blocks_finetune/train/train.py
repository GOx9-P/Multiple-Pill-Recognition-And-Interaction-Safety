"""Entry point last-block fine-tune; bat buoc truyen head checkpoint."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pill_safety.cv.attribute.training.cli import main


if __name__ == "__main__":
    raise SystemExit(main(default_command="train", default_config=PROJECT_ROOT / "configs" / "training" / "attribute_resnet18_last_blocks_finetune" / "config.yaml"))
