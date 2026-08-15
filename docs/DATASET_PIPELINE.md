# NeuroCompiler Dataset Pipeline

> **Legacy documentation (pre-Aug 2026).** This describes the original
> IR-oriented dataset pipeline. The current runtime-based multi-state pipeline
> is `scripts/generate_multistate_dataset.py` (see `README.md`).

The master entry point is `scripts/generate_dataset.py`. It enumerates multiple
benchmark programs and LLVM actions, records independent pre/action/post
transitions incrementally, resumes interrupted runs, and can invoke processing
at the end.

## Outputs

- `datasets/raw/pass_runtime_dataset.csv` — complete raw transition log.
- `datasets/raw/pass_runtime_dataset.manifest.json` — run configuration.
- `datasets/processed/hybrid_dataset.csv` — cleaned, split, normalized dataset.
- `datasets/processed/hybrid_dataset_rejected.csv` — failed/invalid audit rows.
- `datasets/processed/normalization.json` — train-only z-score parameters.
- `datasets/processed/dataset_manifest.json` — final dataset report.

## 1. Fast end-to-end smoke test

```bash
cd ~/NeuroCompiler
conda activate neurocompiler
python scripts/generate_dataset.py \
  --dataset cbench-v1 \
  --max-benchmarks 2 \
  --max-passes 5 \
  --skip-object-text-size \
  --no-resume \
  --process
```

This creates 10 attempted transitions and immediately produces the processed
CSV. Inspect it with:

```bash
python - <<'PY'
import csv
from pathlib import Path
p = Path('datasets/processed/hybrid_dataset.csv')
with p.open() as f:
    r = csv.reader(f)
    header = next(r)
    rows = list(r)
print('rows:', len(rows))
print('columns:', len(header))
print('first 20 columns:', header[:20])
PY
```

## 2. Full deterministic cBench census

```bash
python scripts/generate_dataset.py \
  --dataset cbench-v1 \
  --reward-space IrInstructionCountO3 \
  --process
```

By default this selects all benchmark URIs and all available LLVM actions. Each
pass is independently applied from the benchmark's initial state. Existing
completed transition keys are skipped, so the command can be rerun after an
interruption.

## 3. Select specific benchmark programs and passes

```bash
python scripts/generate_dataset.py \
  --dataset cbench-v1 \
  --benchmark qsort \
  --benchmark dijkstra \
  --benchmark bitcount \
  --passes=-adce,-sroa,-mem2reg \
  --process
```

`--passes` accepts comma-separated action IDs, names, or LLVM flags. Use the
`--passes=...` form when values start with a hyphen.

## 4. Runtime measurements

Runtime is expensive and experimental. Test it on a small subset first and use
a separate raw file:

```bash
python scripts/generate_dataset.py \
  --dataset cbench-v1 \
  --max-benchmarks 2 \
  --max-passes 5 \
  --measure-runtime \
  --require-runtime \
  --runtime-warmup-count 1 \
  --runtime-count 5 \
  --output datasets/raw/pass_runtime_measured_dataset.csv \
  --processed-output datasets/processed/hybrid_runtime_dataset.csv \
  --process
```

Non-runnable programs are retained with empty runtime fields. Runtime samples,
median, mean, standard deviation, and measurement count are recorded.

## 5. Run processing separately

```bash
python scripts/process_dataset.py \
  --input datasets/raw/pass_runtime_dataset.csv \
  --output datasets/processed/hybrid_dataset.csv
```

## 6. Tests

```bash
python -m pytest -q
```

## Useful controls

```text
--max-benchmarks N       Limit benchmark programs.
--max-passes N           Limit LLVM actions.
--benchmark VALUE        Select one program; repeat the option for more.
--passes=VALUE,...       Select actions by ID, name, or flag.
--reward-space VALUE     Default: IrInstructionCountO3.
--measure-runtime        Collect repeated execution times.
--measure-buildtime      Collect Buildtime observations.
--skip-object-text-size  Skip platform-dependent object .TEXT size.
--no-resume              Replace the selected raw CSV.
--process                Produce the processed CSV after raw generation.
--fsync                   Force each row to stable storage; safer but slower.
```
