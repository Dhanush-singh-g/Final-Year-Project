#!/usr/bin/env python3
"""NeuroCompiler scale-up driver: parallel, resumable SL/RL dataset generation.

The base scripts (`scripts/generate_dataset.py`, `scripts/collect_rl_transitions.py`)
are single-process and therefore impractical for the design targets
(150k+ SL samples, 1M RL transitions). This driver shards benchmark URIs across
N worker processes, runs the *same* generation functions per shard (no logic
duplication), and then merges + processes the shards into the canonical
datasets. It is fully resumable: rerunning a command skips completed work using
the existing per-row transition keys / deterministic episode IDs.

Example — scaled SL census across many datasets (runtime labeled):

    python scripts/scale_census.py sl \\
      --workdir datasets/raw/scale_sl \\
      --datasets cbench-v1,chstone-v0,blas-v0,clgen-v0,poj104-v1 \\
      --csmith-count 30 --sample 30 \\
      --workers 24 --measure-runtime --runtime-warmup-count 1 --runtime-count 3

    python scripts/scale_census.py merge-sl --workdir datasets/raw/scale_sl \\
      --output datasets/raw/scale_sl_combined.csv --process \\
      --processed-output datasets/processed/hybrid_dataset_scaled.csv

Example — scaled RL replay buffer (episodes only from train-split benchmarks):

    python scripts/scale_census.py rl \\
      --workdir datasets/raw/scale_rl \\
      --processed-csv datasets/processed/hybrid_dataset_scaled.csv \\
      --workers 24 --episodes-per-benchmark 10 --max-steps-per-episode 8

    python scripts/scale_census.py merge-rl --workdir datasets/raw/scale_rl \\
      --output datasets/replay_buffer/rl_experiences_scaled.csv

Progress:

    python scripts/scale_census.py status --workdir datasets/raw/scale_sl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import multiprocessing
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger("scale_census")


# --------------------------------------------------------------------------
# Planning helpers
# --------------------------------------------------------------------------

def _stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_suite(benchmark_uri: str) -> str:
    """Derive the dataset URI (benchmark_suite provenance) from a benchmark URI."""
    if benchmark_uri.startswith("benchmark://"):
        rest = benchmark_uri[len("benchmark://"):]
        suite = "benchmark://" + rest.split("/", 1)[0]
    elif benchmark_uri.startswith("generator://"):
        rest = benchmark_uri[len("generator://"):]
        suite = "generator://" + rest.split("/", 1)[0]
    else:
        suite = benchmark_uri
    return suite


def enumerate_dataset_benchmarks(env: Any, dataset_uri: str, sample: Optional[int]) -> List[str]:
    """List benchmark URIs for one dataset, optionally a stable hash sample."""
    dataset = env.datasets[dataset_uri]
    uris = sorted(str(uri) for uri in dataset.benchmark_uris())
    if not uris:
        raise RuntimeError(f"No benchmarks found in {dataset_uri}")
    if sample is not None and 0 < sample < len(uris):
        uris = sorted(uris, key=_stable_hash)[:sample]
    return uris


def csmith_uris(count: int, seed_offset: int = 0) -> List[str]:
    """Deterministic synthetic-program URIs from the csmith generator."""
    return [f"generator://csmith-v0/{seed_offset + i}" for i in range(count)]


def shard_assignments(uris: Sequence[str], workers: int) -> List[List[str]]:
    """Stable, hash-balanced sharding of benchmark URIs across workers."""
    buckets: List[List[str]] = [[] for _ in range(workers)]
    for uri in sorted(uris):
        buckets[_stable_hash(uri) % workers].append(uri)
    return buckets


def _seed_worker_csv(
    worker_csv: Path, resume_from: Path, shard_uris: Sequence[str]
) -> int:
    """Copy already-completed rows belonging to this shard into the worker CSV.

    generate()/collect_rl() treat rows already present as completed and skip
    them on resume, so seeding a canonical merged file makes re-runs cheap
    instead of re-walking every benchmark.
    """

    if not resume_from.exists() or not shard_uris:
        return 0
    wanted = set(shard_uris)
    fields: List[str] = []
    rows: List[Dict[str, str]] = []
    with resume_from.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            if row.get("benchmark_uri", "") in wanted:
                rows.append(row)
    if not rows:
        return 0
    worker_csv.parent.mkdir(parents=True, exist_ok=True)
    with worker_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def csv_row_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def csv_fieldnames(path: Path) -> List[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


# --------------------------------------------------------------------------
# Worker entry points (module-level so multiprocessing spawn can pickle them)
# --------------------------------------------------------------------------

def _setup_worker_logging(log_path: Path, level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)


def _run_sl_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    from generate_dataset import generate  # imported inside the worker process

    worker_index = payload["worker_index"]
    _setup_worker_logging(
        Path(payload["workdir"]) / f"worker_{worker_index}.log", payload["log_level"]
    )
    args = argparse.Namespace(**payload["sl_args"])
    if not args.benchmark:
        raise RuntimeError(
            "Empty benchmark shard; refusing to enumerate the full dataset "
            "(generate_dataset treats an empty --benchmark list as 'all')."
        )
    started = time.perf_counter()
    try:
        generate(args)
        status = "ok"
    except Exception as error:  # noqa: BLE001 - worker isolation
        LOGGER.exception("SL worker %s failed", worker_index)
        status = f"error: {type(error).__name__}: {error}"
    output = Path(payload["output"])
    return {
        "worker": worker_index,
        "status": status,
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "rows": csv_row_count(output),
    }


def _run_rl_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    from collect_rl_transitions import collect_rl  # imported inside the worker

    worker_index = payload["worker_index"]
    _setup_worker_logging(
        Path(payload["workdir"]) / f"rl_worker_{worker_index}.log", payload["log_level"]
    )
    args = argparse.Namespace(**payload["rl_args"])
    if not args.benchmark:
        raise RuntimeError(
            "Empty benchmark shard; refusing to enumerate the full dataset."
        )
    started = time.perf_counter()
    try:
        collect_rl(args)
        status = "ok"
    except Exception as error:  # noqa: BLE001
        LOGGER.exception("RL worker %s failed", worker_index)
        status = f"error: {type(error).__name__}: {error}"
    output = Path(payload["output"])
    return {
        "worker": worker_index,
        "status": status,
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "rows": csv_row_count(output),
    }


# --------------------------------------------------------------------------
# SL command
# --------------------------------------------------------------------------

def _sl_worker_args(
    args: argparse.Namespace, output: Path, benchmarks: Sequence[str]
) -> argparse.Namespace:
    """Build the argparse.Namespace expected by generate_dataset.generate()."""
    return argparse.Namespace(
        dataset="benchmark://cbench-v1",  # provenance fixed at merge time
        benchmark=list(benchmarks),
        max_benchmarks=None,
        passes=",".join(args.passes),
        max_passes=None,
        reward_space=args.reward_space,
        measure_runtime=args.measure_runtime,
        require_runtime=False,
        runtime_count=args.runtime_count,
        runtime_warmup_count=args.runtime_warmup_count,
        measure_buildtime=args.measure_buildtime,
        skip_object_text_size=args.skip_object_text_size,
        timeout=args.timeout,
        output=str(output),
        resume=True,
        fsync=False,
        process=False,
        processed_output="",
        log_level=args.log_level,
    )


def cmd_sl(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    import compiler_gym

    env = compiler_gym.make("llvm-v0")
    plan: Dict[str, Any] = {"mode": "sl", "created_at_utc": _utc_now()}
    all_uris: List[str] = []
    try:
        for dataset_uri in args.datasets:
            uris = enumerate_dataset_benchmarks(env, dataset_uri, args.sample)
            plan[dataset_uri] = len(uris)
            all_uris.extend(uris)
            LOGGER.info("Dataset %s: %d benchmarks", dataset_uri, len(uris))
    finally:
        env.close()

    if args.csmith_count > 0:
        csm = csmith_uris(args.csmith_count, seed_offset=args.csmith_seed)
        plan["generator://csmith-v0"] = len(csm)
        all_uris.extend(csm)

    if not all_uris:
        raise RuntimeError("No benchmarks selected")

    n_shards = args.shards if args.shards else len(all_uris)
    buckets = shard_assignments(all_uris, n_shards)
    expected_per_worker = [len(bucket) * len(args.passes) for bucket in buckets]
    plan["benchmark_count"] = len(all_uris)
    plan["transition_count_target"] = sum(expected_per_worker)
    plan["shards"] = n_shards
    plan["passes"] = args.passes
    plan["workers"] = args.workers
    plan["config"] = {
        "measure_runtime": args.measure_runtime,
        "runtime_count": args.runtime_count,
        "runtime_warmup_count": args.runtime_warmup_count,
        "reward_space": args.reward_space,
    }
    (workdir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )

    payloads: List[Dict[str, Any]] = []
    for worker_index, bucket in enumerate(buckets):
        output = workdir / f"worker_{worker_index}.csv"
        if args.resume_from and bucket and not output.exists():
            seeded = _seed_worker_csv(
                output, Path(args.resume_from).expanduser().resolve(), bucket
            )
            if seeded:
                LOGGER.info(
                    "worker %d: seeded %d completed rows from %s",
                    worker_index,
                    seeded,
                    args.resume_from,
                )
        payloads.append(
            {
                "worker_index": worker_index,
                "workdir": str(workdir),
                "output": str(output),
                "log_level": args.log_level,
                "sl_args": vars(_sl_worker_args(args, output, bucket)),
            }
        )

    return _run_workers(payloads, expected_per_worker, _run_sl_worker, args.workers)


# --------------------------------------------------------------------------
# RL command
# --------------------------------------------------------------------------

def _rl_worker_args(
    args: argparse.Namespace, output: Path, benchmarks: Sequence[str], seed: int
) -> argparse.Namespace:
    """Build the argparse.Namespace expected by collect_rl_transitions.collect_rl()."""
    return argparse.Namespace(
        dataset="benchmark://cbench-v1",  # provenance fixed at merge time
        benchmark=list(benchmarks),
        max_benchmarks=None,
        passes=None,
        max_passes=None,
        reward_space=args.reward_space,
        episodes_per_benchmark=args.episodes_per_benchmark,
        max_steps_per_episode=args.max_steps_per_episode,
        seed=seed,
        measure_runtime=args.measure_runtime,
        runtime_count=args.runtime_count,
        runtime_warmup_count=args.runtime_warmup_count,
        skip_object_text_size=args.skip_object_text_size,
        reward_weight_runtime=args.reward_weight_runtime,
        reward_weight_ir=args.reward_weight_ir,
        reward_weight_size=args.reward_weight_size,
        allow_no_effect=args.allow_no_effect,
        terminate_on_zero_reward=args.terminate_on_zero_reward,
        timeout=args.timeout,
        output=str(output),
        resume=True,
        log_level=args.log_level,
    )


def load_train_benchmarks(processed_csv: Path) -> List[str]:
    """Return benchmark URIs assigned to the train split (no test leakage)."""
    uris = set()
    with processed_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("dataset_split") == "train":
                uri = row.get("benchmark_uri", "")
                if uri:
                    uris.add(uri)
    return sorted(uris)


def cmd_rl(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    if args.benchmark:
        uris = list(args.benchmark)
    elif args.processed_csv and Path(args.processed_csv).exists():
        uris = load_train_benchmarks(Path(args.processed_csv))
        if not uris:
            raise RuntimeError(
                f"No train-split benchmarks found in {args.processed_csv}"
            )
    else:
        raise RuntimeError(
            "Provide --processed-csv (train-split benchmarks) or --benchmark list"
        )

    LOGGER.info("RL collection: %d train-split benchmarks", len(uris))
    n_shards = args.shards if args.shards else len(uris)
    buckets = shard_assignments(uris, n_shards)
    expected_per_worker = [
        len(bucket) * args.episodes_per_benchmark for bucket in buckets
    ]
    plan = {
        "mode": "rl",
        "created_at_utc": _utc_now(),
        "benchmark_count": len(uris),
        "episodes_per_benchmark": args.episodes_per_benchmark,
        "max_steps_per_episode": args.max_steps_per_episode,
        "workers": args.workers,
        "shards": n_shards,
        "seed": args.seed,
    }
    (workdir / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n"
    )

    payloads: List[Dict[str, Any]] = []
    for worker_index, bucket in enumerate(buckets):
        output = workdir / f"rl_worker_{worker_index}.csv"
        if args.resume_from and bucket and not output.exists():
            seeded = _seed_worker_csv(
                output, Path(args.resume_from).expanduser().resolve(), bucket
            )
            if seeded:
                LOGGER.info(
                    "rl worker %d: seeded %d completed rows from %s",
                    worker_index,
                    seeded,
                    args.resume_from,
                )
        payloads.append(
            {
                "worker_index": worker_index,
                "workdir": str(workdir),
                "output": str(output),
                "log_level": args.log_level,
                "rl_args": vars(
                    _rl_worker_args(
                        args, output, bucket, seed=args.seed + worker_index
                    )
                ),
            }
        )

    return _run_workers(payloads, expected_per_worker, _run_rl_worker, args.workers)


# --------------------------------------------------------------------------
# Shared worker launcher
# --------------------------------------------------------------------------

def _run_workers(
    payloads: List[Dict[str, Any]],
    expected_per_worker: Sequence[int],
    worker_fn: Any,
    max_workers: int,
) -> int:
    started = time.perf_counter()
    results: List[Dict[str, Any]] = []
    pending = [p for p in payloads if not _worker_complete(p, expected_per_worker)]
    empty = [p for p in pending if expected_per_worker[p["worker_index"]] == 0]
    for payload in empty:
        print(
            f"[scale] worker {payload['worker_index']}: empty shard, skipped "
            f"(use fewer workers than benchmarks)"
        )
    pending = [p for p in pending if expected_per_worker[p["worker_index"]] > 0]
    if not pending:
        print(
            f"[scale] All {len(payloads)} shards already complete "
            f"(rows match targets); nothing to do."
        )
        return 0
    print(
        f"[scale] Launching {len(pending)}/{len(payloads)} workers "
        f"(max {max_workers} in parallel) ..."
    )
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=max_workers) as pool:
        for result in pool.imap_unordered(worker_fn, pending):
            results.append(result)
            done = sum(1 for p in payloads if _worker_complete(p, expected_per_worker))
            print(
                f"[scale] worker {result['worker']}: {result['status']} "
                f"rows={result['rows']} elapsed={result['elapsed_sec']}s "
                f"| total done {done}/{len(payloads)}"
            )
    print(
        f"[scale] All workers finished in {time.perf_counter() - started:.1f}s. "
        f"Run 'merge-sl'/'merge-rl' to combine shards."
    )
    return 0


def _worker_complete(payload: Dict[str, Any], expected_per_worker: Sequence[int]) -> bool:
    worker_index = payload["worker_index"]
    output = Path(payload["output"])
    expected = expected_per_worker[worker_index]
    # +1 header line. Transitions that failed are still written as rows, so a
    # matching row count means this shard is finished.
    return csv_row_count(output) >= expected + 1


# --------------------------------------------------------------------------
# Merge commands
# --------------------------------------------------------------------------

def _merge_shards(
    workdir: Path,
    pattern: str,
    output: Path,
    dedup_col: str,
) -> Dict[str, Any]:
    shards = sorted(workdir.glob(pattern))
    if not shards:
        raise RuntimeError(f"No shard CSVs found in {workdir} matching {pattern!r}")
    fields: List[str] = []
    seen = set()
    merged: List[Dict[str, str]] = []
    per_shard: Dict[str, int] = {}
    for shard in shards:
        with shard.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not fields:
                fields = list(reader.fieldnames or [])
            shard_rows = 0
            for row in reader:
                shard_rows += 1
                key = row.get(dedup_col, "")
                if not key or key in seen:
                    continue
                seen.add(key)
                # Fix benchmark_suite provenance for mixed-dataset runs.
                if "benchmark_uri" in row and "benchmark_suite" in row:
                    row["benchmark_suite"] = _dataset_suite(row["benchmark_uri"])
                merged.append(row)
            per_shard[shard.name] = shard_rows
    if not fields:
        raise RuntimeError(f"No columns found in {shards[0]}")
    if not merged:
        raise RuntimeError("No rows found after merging (nothing generated yet?)")

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)

    report = {
        "merged_at_utc": _utc_now(),
        "shards": per_shard,
        "rows_before_dedup": sum(per_shard.values()),
        "rows_after_dedup": len(merged),
        "output": str(output),
    }
    (output.with_suffix(".merge.json")).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def cmd_merge_sl(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).expanduser().resolve()
    report = _merge_shards(workdir, "worker_*.csv", args.output, "transition_key")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.process:
        from process_dataset import process_dataset

        process_dataset(
            Path(args.output),
            Path(args.processed_output).expanduser().resolve(),
            require_runtime=args.require_runtime,
        )
    return 0


def cmd_merge_rl(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).expanduser().resolve()
    report = _merge_shards(workdir, "rl_worker_*.csv", args.output, "transition_key")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).expanduser().resolve()
    plan_path = workdir / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        print(json.dumps(plan, indent=2, sort_keys=True))
    pattern = "worker_*.csv" if plan_path.exists() and plan.get("mode") == "sl" else "rl_worker_*.csv"
    for shard in sorted(workdir.glob(pattern)):
        print(f"{shard.name}: {csv_row_count(shard)} lines")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _add_sl_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workdir", required=True, help="Directory for plan + shards")
    parser.add_argument(
        "--datasets",
        default="cbench-v1,chstone-v0",
        help="Comma-separated CompilerGym dataset URIs/names.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Stable per-dataset sample size (hash-ordered). Default: all.",
    )
    parser.add_argument(
        "--csmith-count", type=int, default=0, help="Number of csmith generator programs."
    )
    parser.add_argument("--csmith-seed", type=int, default=1)
    parser.add_argument(
        "--passes",
        default=None,
        help="Comma-separated pass flags. Default: curated 31-pass set.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--shards",
        type=int,
        default=None,
        help="Number of shards (default: one per benchmark). Fewer benchmarks per "
        "shard means faster resume after an interrupted run.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Canonical merged CSV to seed already-completed rows from.",
    )
    parser.add_argument("--reward-space", default="IrInstructionCountO3")
    parser.add_argument("--measure-runtime", action="store_true")
    parser.add_argument("--runtime-count", type=int, default=3)
    parser.add_argument("--runtime-warmup-count", type=int, default=1)
    parser.add_argument("--measure-buildtime", action="store_true")
    parser.add_argument("--skip-object-text-size", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--log-level", default="INFO")


def _add_rl_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workdir", required=True)
    parser.add_argument(
        "--processed-csv",
        default=str(PROJECT_ROOT / "datasets" / "processed" / "hybrid_dataset.csv"),
        help="Processed SL dataset; its train-split benchmarks get episodes.",
    )
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--shards",
        type=int,
        default=None,
        help="Number of shards (default: one per benchmark).",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Canonical merged replay buffer to seed completed episodes from.",
    )
    parser.add_argument("--episodes-per-benchmark", type=int, default=10)
    parser.add_argument("--max-steps-per-episode", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-space", default="IrInstructionCountO3")
    parser.add_argument("--measure-runtime", action="store_true")
    parser.add_argument("--runtime-count", type=int, default=3)
    parser.add_argument("--runtime-warmup-count", type=int, default=1)
    parser.add_argument("--skip-object-text-size", action="store_true")
    parser.add_argument("--reward-weight-runtime", type=float, default=0.6)
    parser.add_argument("--reward-weight-ir", type=float, default=0.3)
    parser.add_argument("--reward-weight-size", type=float, default=0.1)
    parser.add_argument("--allow-no-effect", action="store_true")
    parser.add_argument("--terminate-on-zero-reward", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--log-level", default="INFO")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parallel, resumable SL/RL dataset scale-up driver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sl = sub.add_parser("sl", help="Generate scaled SL census via parallel workers")
    _add_sl_args(p_sl)

    p_rl = sub.add_parser("rl", help="Generate scaled RL replay buffer via workers")
    _add_rl_args(p_rl)

    p_ms = sub.add_parser("merge-sl", help="Merge SL shards into one raw CSV")
    p_ms.add_argument("--workdir", required=True)
    p_ms.add_argument("--output", required=True)
    p_ms.add_argument("--process", action="store_true")
    p_ms.add_argument(
        "--processed-output",
        default=str(PROJECT_ROOT / "datasets" / "processed" / "hybrid_dataset.csv"),
    )
    p_ms.add_argument("--require-runtime", action="store_true")

    p_mr = sub.add_parser("merge-rl", help="Merge RL shards into one replay buffer")
    p_mr.add_argument("--workdir", required=True)
    p_mr.add_argument("--output", required=True)

    p_st = sub.add_parser("status", help="Show plan + per-shard progress")
    p_st.add_argument("--workdir", required=True)

    args = parser.parse_args()
    log_level = getattr(args, "log_level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    LOGGER.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if args.command == "sl":
        if args.passes:
            args.passes = [p.strip() for p in args.passes.split(",") if p.strip()]
        else:
            from curated_passes import get_curated_flags
            args.passes = get_curated_flags()
        args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        return cmd_sl(args)
    if args.command == "rl":
        return cmd_rl(args)
    if args.command == "merge-sl":
        return cmd_merge_sl(args)
    if args.command == "merge-rl":
        return cmd_merge_rl(args)
    if args.command == "status":
        return cmd_status(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
