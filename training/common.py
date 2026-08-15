#!/usr/bin/env python3
"""
Common utilities for SL and RL training.

- Feature column selection
- Loading processed CSV
- Normalization handling
- Action vocabulary building
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from scripts.extract_features import AUTOPHASE_FEATURE_NAMES  # type: ignore
except ImportError:
    AUTOPHASE_FEATURE_NAMES = (
        "BBNumArgsHi","BBNumArgsLo","onePred","onePredOneSuc","onePredTwoSuc","oneSuccessor",
        "twoPred","twoPredOneSuc","twoEach","twoSuccessor","morePreds","BB03Phi","BBHiPhi",
        "BBNoPhi","BeginPhi","BranchCount","returnInt","CriticalCount","NumEdges","const32Bit",
        "const64Bit","numConstZeroes","numConstOnes","UncondBranches","binaryConstArg","NumAShrInst",
        "NumAddInst","NumAllocaInst","NumAndInst","BlockMid","BlockLow","NumBitCastInst","NumBrInst",
        "NumCallInst","NumGetElementPtrInst","NumICmpInst","NumLShrInst","NumLoadInst","NumMulInst",
        "NumOrInst","NumPHIInst","NumRetInst","NumSExtInst","NumSelectInst","NumShlInst","NumStoreInst",
        "NumSubInst","NumTruncInst","NumXorInst","NumZExtInst","TotalBlocks","TotalInsts","TotalMemInst",
        "TotalFuncs","ArgsPhi","testUnary",
    )

LOGGER = logging.getLogger(__name__)

CORE_NUMERIC_FEATURES = [
    "pre_ir_instruction_count",
    "pre_object_text_size_bytes",
    "pre_total_basic_blocks",
    "pre_total_functions",
    "pre_total_instructions",
    "pre_total_memory_instructions",
]

def get_pre_autophase_cols() -> List[str]:
    return [f"pre_autophase_{name}" for name in AUTOPHASE_FEATURE_NAMES]

def get_feature_cols(available: List[str], use_norm: bool = False) -> List[str]:
    """
    Choose feature columns present in CSV.

    Defaults to RAW pre-state features: raw values are safer for tree models
    (train_sl.py trains on raw by default). Normalized features are only used
    when explicitly requested AND present, because online inference has no
    normalization statistics to apply (inference.py rejects norm_* models).
    """
    cols = []
    # Try norm_pre_*
    if use_norm:
        norm_core = [f"norm_{c}" for c in CORE_NUMERIC_FEATURES]
        norm_auto = [f"norm_pre_autophase_{n}" for n in AUTOPHASE_FEATURE_NAMES]
        for c in norm_core + norm_auto:
            if c in available:
                cols.append(c)
        if cols:
            return cols
    # Fallback raw
    for c in CORE_NUMERIC_FEATURES:
        if c in available:
            cols.append(c)
    for c in get_pre_autophase_cols():
        if c in available:
            cols.append(c)
    return cols

def safe_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None

def load_csv_rows(path: Path, max_rows: Optional[int] = None) -> Tuple[List[Dict[str, str]], List[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for i, r in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append(r)
    return rows, fieldnames

def build_action_vocab(rows: List[Dict[str, str]]) -> Dict[str, int]:
    """Map pass_flag or pass_name to contiguous ids for model output"""
    vocab = {}
    for r in rows:
        key = r.get("pass_flag") or r.get("pass_name") or r.get("pass_id")
        if key and key not in vocab:
            vocab[key] = len(vocab)
    return vocab

def split_by_column(rows: List[Dict[str, str]], split_col: str = "dataset_split") -> Dict[str, List[Dict[str, str]]]:
    out = {"train": [], "validation": [], "test": []}
    for r in rows:
        split = r.get(split_col, "train")
        if split not in out:
            split = "train"
        out[split].append(r)
    return out


# Scale-free state features: ratios of the absolute IR/size/block/function
# counts. They are benchmark-size agnostic (a big program and a small program
# with the same shape get the same values), so a Q-function trained on one
# benchmark family does not depend on the absolute program size of the
# training family. Derived per-state, so they are computable online at
# inference for unseen benchmarks without any normalization statistics.
SCALE_FREE_RATIO_COLS: List[str] = [
    "pre_ir_per_func",
    "pre_mem_frac",
    "pre_size_per_inst",
    "pre_blocks_per_func",
    "pre_insts_per_block",
]


def derive_ratio_features(row: Dict[str, object], prefix: str = "pre_") -> Dict[str, float]:
    """Compute the 5 scale-free ratio features from a row (or flattened state
    dict) that carries ``{prefix}ir_instruction_count`` etc."""
    def val(name: str) -> float:
        try:
            return float(row.get(f"{prefix}{name}", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    ir = val("ir_instruction_count")
    funcs = val("total_functions")
    mem = val("total_memory_instructions")
    insts = val("total_instructions")
    size = val("object_text_size_bytes")
    blocks = val("total_basic_blocks")

    def ratio(a: float, b: float) -> float:
        return (a / b) if b else 0.0

    return {
        f"{prefix}ir_per_func": ratio(ir, funcs),
        f"{prefix}mem_frac": ratio(mem, insts),
        f"{prefix}size_per_inst": ratio(size, insts),
        f"{prefix}blocks_per_func": ratio(blocks, funcs),
        f"{prefix}insts_per_block": ratio(insts, blocks),
    }
