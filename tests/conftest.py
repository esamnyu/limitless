"""Shared test configuration — adds project root to sys.path."""

import sys
from pathlib import Path

# Ensure project root is importable from all test files
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
