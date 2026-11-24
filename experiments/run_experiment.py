"""Command-line entry point for reproducible operating-characteristic runs.

Produces the two numbers the brief asks for -- false-acceptance risk and
average inspection count -- across the fixed risk presets, using paired
seeds so the same fictional lots drive every configuration.

Usage:
    python -m experiments.run_experiment --replications 200 --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import simulate, stats, storage  # noqa: E402

# True defective rates spanning good, indifferent and clearly bad lots.
DEFAULT_TRUE_PS = [0.0, 0.02, 0.03, 0.06, 0.10, 0.15, 0.25]
DEFAULT_ROUND_SIZE = 8


def run(replications: int, base_seed: int,
        true_ps: list[float], round_size: int, cap: int) -> dict:
    plans = [stats.make_plan(**p) for p in storage.RISK_PRESETS]
    grids = simulate.compare_plans(
        plans, true_ps, replications, base_seed, round_size, cap)
    summary = []
    for grid in grids:
        plan = grid["plan"]
        rows = []
        for cell in grid["cells"]:
            rows.append({
                "true_p": cell["true_p"],
                "accept_rate": round(cell["accept_rate"], 4),
                "reject_rate": round(cell["reject_rate"], 4),
                "censor_rate": round(cell["censor_rate"], 4),
                "avg_inspection_count": round(cell["avg_n"], 2),
                "n_range": [cell["min_n"], cell["max_n"]],
            })
        # headline guarantees at the two hypothesis points
        at_p0 = next((r for r in rows if abs(r["true_p"] - plan["p0"]) < 1e-9),
                     None)
        at_p1 = next((r for r in rows if abs(r["true_p"] - plan["p1"]) < 1e-9),
                     None)
        summary.append({
            "plan": plan,
            "false_reject_at_p0": at_p0["reject_rate"] if at_p0 else None,
            "false_accept_at_p1": at_p1["accept_rate"] if at_p1 else None,
            "cells": rows,
        })
    return {
        "base_seed": base_seed,
        "replications": replications,
        "round_size": round_size,
        "cap": cap,
        "true_ps": true_ps,
        "note": "paired seeds: every preset inspects the same fictional lot",
        "results": summary,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=20261124)
    parser.add_argument("--round-size", type=int, default=DEFAULT_ROUND_SIZE)
    parser.add_argument("--cap", type=int, default=1500,
                        help="max items per run; undecided runs are censored")
    parser.add_argument("--true-ps", default=",".join(map(str, DEFAULT_TRUE_PS)))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    true_ps = [float(x) for x in args.true_ps.split(",")]
    report = run(args.replications, args.base_seed, true_ps,
                 args.round_size, args.cap)
    text = json.dumps(report, ensure_ascii=False, indent=2)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"report written to {out}")
    else:
        print(text)

    # compact human table
    print("\n方案 / 真实p / 接受率 / 拒收率 / 截尾率 / 平均检验量",
          file=sys.stderr)
    for block in report["results"]:
        name = block["plan"]["name"]
        for row in block["cells"]:
            print(f"{name:24s} p={row['true_p']:5.2f}  "
                  f"accept={row['accept_rate']:6.3f}  "
                  f"reject={row['reject_rate']:6.3f}  "
                  f"censor={row['censor_rate']:6.3f}  "
                  f"ASN={row['avg_inspection_count']:8.2f}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
