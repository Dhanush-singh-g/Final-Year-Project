#!/usr/bin/env python3
"""
Wrapper for evaluation/evaluate_benchmarks.py to match required repo structure:

NeuroCompiler/
├── scripts/
│   ├── extract_features.py
│   ├── generate_sl_dataset.py
│   ├── collect_rl_transitions.py
│   ├── process_dataset.py
│   └── evaluate.py  <- this file

Delegates to evaluation module.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_benchmarks import main

if __name__ == "__main__":
    raise SystemExit(main())
