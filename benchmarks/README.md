# Benchmark Collection — Phase 1

Goal: Diversity

## Dataset 1: cBench (default)
Contains:
- embedded software
- compression
- encryption
- image processing
- networking

23 programs. Excellent starting point.

CompilerGym URI: `benchmark://cbench-v1/*`

To list:
```python
import compiler_gym
env = compiler_gym.make("llvm-v0")
print(list(env.datasets["benchmark://cbench-v1"].benchmark_uris())[:5])
```

## Dataset 2: PolyBench / NPB / CHStone ?
Contains:
- matrix multiplication
- stencil kernels
- numerical loops

Useful for:
- loop optimizations
- vectorization
- LICM

In CompilerGym: often `benchmark://npb-v1/` or `benchmark://polybench-v0/` if installed, or `benchmark://chstone-v0/`

## Dataset 3: LLVM Test Suite
General compiler benchmarks.

`benchmark://llvm-v0/` special? Actually LLVM test suite available via `benchmark://cbench-v1` plus others. For full llvm test suite you can install custom datasets.

## Dataset 4: AnghaBench
Thousands of automatically extracted C functions.

Huge diversity.

In CompilerGym: `benchmark://anghabench-v1/`

Size:
- AnghaBench 1M functions. We typically sample 5000 × 30 = 150k SL rows.

## Optional: SPEC CPU
Excellent benchmark. Commercial license.

Not included in open-source pipeline.

## How to use in NeuroCompiler

All dataset URIs are passed via `--dataset` flag:

```bash
# cBench only
python scripts/generate_sl_dataset.py --dataset cbench-v1 --process

# AnghaBench sample 100 benchmarks, 20 episodes each for RL
python scripts/collect_rl_transitions.py --dataset anghabench-v1 --max-benchmarks 100 --episodes-per-benchmark 20
```

## Expected total supervised samples

```
cBench 23 * 30 passes = 690
LLVM Test Suite 500 * 30 = 15000
PolyBench 30 * 30 = 900
AnghaBench 5000 * 30 = 150000
Total ≈166k supervised samples
```

## Expected RL replay buffer

```
500 benchmarks × 200 random episodes = 100k episodes
Each episode ~10 transitions
Replay buffer ≈1 million transitions
```

This matches the design doc from 2026-07-31.
