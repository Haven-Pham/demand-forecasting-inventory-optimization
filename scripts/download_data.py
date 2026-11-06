"""Downloads and verifies the M5 files into data/raw, which is not committed.

    python scripts/download_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from demand_inventory.data import download  # noqa: E402

print(f"M5 files ready in {download()}")
