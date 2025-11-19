"""
Convenience script to train model from scratch with optimal settings.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Just run the improved training script
os.chdir(PROJECT_ROOT)
exec(open(PROJECT_ROOT / "pretraining" / "train_improved.py").read())

