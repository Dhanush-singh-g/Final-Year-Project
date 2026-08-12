#!/usr/bin/env python3
"""
Evaluation: Compare Hybrid vs -O1/-O2/-O3

Objective from design: Given unseen C/C++ program, produce better code than default -O1/-O2/-O3.

We evaluate on held-out test split from process_dataset.py (benchmark-wise split prevents leakage).

Metrics:
 - IR instruction count reduction % vs O0
 - Object .TEXT size reduction %
 - Runtime speedup if available (requires measure_runtime)
 - Win rate vs -O3 (hybrid beats O3)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOGGER = logging.getLogger("evaluate")

def load_test_benchmarks(processed_csv: Path) -> List[str]:
    benchmarks = set()
    with processed_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset_split") == "test":
                benchmarks.add(row.get("benchmark_uri",""))
    return sorted(b for b in benchmarks if b)

def evaluate_csv_metrics(processed_csv: Path, target: str):
    from collections import defaultdict
    per_pass_reward = defaultdict(list)
    with processed_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                flag = row.get("pass_flag","")
                reward = float(row.get(target, ""))
                per_pass_reward[flag].append(reward)
            except (TypeError, ValueError):
                continue
    avg = {k: sum(v)/len(v) for k,v in per_pass_reward.items() if v}
    top = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print(f"[Eval] Processed CSV: {processed_csv}")
    print(f"  Unique passes: {len(per_pass_reward)}")
    print(f"  Top 10 passes by average {target}:")
    for flag, score in top:
        print(f"    {flag}: {score:.4f}")

def run_hybrid_on_testset(args: argparse.Namespace):
    # Import inference
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from training.inference import hybrid_optimize_benchmark

    test_benchmarks = []
    if Path(args.processed_csv).exists():
        test_benchmarks = load_test_benchmarks(Path(args.processed_csv))
        print(f"[Eval] Test set benchmarks: {len(test_benchmarks)}")

    if args.benchmarks:
        test_benchmarks = args.benchmarks

    if not test_benchmarks:
        test_benchmarks = ["benchmark://cbench-v1/qsort", "benchmark://cbench-v1/dijkstra", "benchmark://cbench-v1/bitcount"]

    results = []
    for uri in test_benchmarks[: args.max_benchmarks]:
        try:
            res = hybrid_optimize_benchmark(
                benchmark_uri=uri,
                max_steps=args.max_steps,
                sl_dir=Path(args.sl_model_dir),
                rl_dir=Path(args.rl_model_dir),
                measure_runtime=args.measure_runtime,
                verbose=False,
            )
            results.append(res)
            runtime_text = (
                f" runtime={res['runtime_speedup']:.4f}x"
                if res.get("runtime_speedup") is not None else " runtime=N/A"
            )
            print(f"{uri}: {res['initial_ir']} -> {res['final_ir']} ({res['ir_reduction_pct']:.1f}% IR reduction){runtime_text} seq={res['pass_sequence']}")
        except Exception as e:
            print(f"{uri} failed: {e}")

    if results:
        avg_reduction = sum(r["ir_reduction_pct"] for r in results)/len(results)
        print(f"\n[Eval] Average IR reduction on test set: {avg_reduction:.2f}% over {len(results)} benchmarks")
        o3_results = [r for r in results if r.get("hybrid_vs_o3_ir_pct") is not None]
        if o3_results:
            avg_vs_o3 = sum(r["hybrid_vs_o3_ir_pct"] for r in o3_results) / len(o3_results)
            o3_wins = sum(r["hybrid_vs_o3_ir_pct"] > 0 for r in o3_results)
            print(
                f"[Eval] Hybrid vs -O3 IR: mean {avg_vs_o3:+.2f}%; "
                f"wins={o3_wins}/{len(o3_results)}"
            )
        runtime_results = [r for r in results if r.get("runtime_speedup") is not None]
        if runtime_results:
            import math
            geo_speedup = math.exp(
                sum(math.log(r["runtime_speedup"]) for r in runtime_results)
                / len(runtime_results)
            )
            wins = sum(r["runtime_speedup"] > 1.0 for r in runtime_results)
            print(
                f"[Eval] Runtime geometric-mean speedup vs initial state: "
                f"{geo_speedup:.4f}x; wins={wins}/{len(runtime_results)}"
            )
            print(
                "[Eval] NOTE: this is runtime vs the initial no-pass state, not "
                "runtime vs -O3. CompilerGym exposes the -O3 IR cost directly but "
                "not an -O3 Runtime observation."
            )
            # Guardrail (runbook section 10): do not claim runtime superiority
            # over -O3 from the numbers above. Only the external O3 executable
            # baseline harness may back such a claim; surface it when measured.
            o3_summary = PROJECT_ROOT / "results" / "o3_runtime_vs_o3_summary.json"
            if o3_summary.exists():
                try:
                    data = json.loads(o3_summary.read_text())
                    gm = data.get("geo_mean_speedup")
                    n = data.get("benchmarks_evaluated")
                    wins = data.get("wins")
                    if gm is not None:
                        print(
                            "[Eval] O3 executable baseline (external harness): "
                            f"geo-mean speedup vs opt -O3 = {gm:.4f}x over {n} "
                            f"benchmarks (wins {wins}/{n}). "
                            f"See {o3_summary.relative_to(PROJECT_ROOT)}."
                        )
                except Exception as error:
                    print(f"[Eval] WARNING: could not read O3 baseline summary: {error}")
            else:
                print(
                    "[Eval] GUARDRAIL: no measured O3 baseline found "
                    f"({o3_summary.relative_to(PROJECT_ROOT)}). Do not claim runtime "
                    "superiority over -O3; run evaluation/o3_runtime_harness.py first."
                )
        # Save
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2, default=str))
            print(f"Saved results to {args.output}")

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Hybrid vs baselines")
    # Canonical dataset is the scaled one; the pilot (hybrid_dataset.csv) is
    # only used when explicitly requested.
    p.add_argument("--processed-csv", default=str(PROJECT_ROOT / "datasets" / "processed" / "hybrid_dataset_scaled.csv"))
    p.add_argument("--benchmarks", action="append", default=[], help="Benchmark URIs to evaluate (overrides test split)")
    p.add_argument("--max-benchmarks", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--measure-runtime", action="store_true")
    p.add_argument(
        "--target", default="runtime_improvement_pct",
        help="CSV target used for offline pass statistics.",
    )
    p.add_argument("--sl-model-dir", default=str(PROJECT_ROOT / "models" / "supervised"))
    p.add_argument("--rl-model-dir", default=str(PROJECT_ROOT / "models" / "reinforcement"))
    p.add_argument("--output", default=str(PROJECT_ROOT / "results" / "hybrid_test_results.json"))
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    if Path(args.processed_csv).exists():
        evaluate_csv_metrics(Path(args.processed_csv), args.target)
    run_hybrid_on_testset(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
