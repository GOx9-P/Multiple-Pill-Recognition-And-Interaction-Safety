"""
Shared fixtures for CV attribute tests.
"""

import sys
from pathlib import Path

# Ensure src/ is always importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
