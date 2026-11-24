"""Reproducible simulation experiments on fictional lots.

The simulator drives the *real* state machine (state.create_batch /
record_items / rebuild), never a re-implemented decision loop, so the
reported error rates and average inspection counts describe exactly what the
application does.  Every run is seeded and every generated item gets a
deterministic id.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import state, stats


@dataclass
class SimResult:
    seed: int
    true_p: float
    decision: str
    n: int
    k: int
    rounds: int
    batch_id: str
    timeline: list

    def to_dict(self) -> dict:
        return {
            "seed": self.seed, "true_p": self.true_p,
            "decision": self.decision, "n": self.n, "k": self.k,
            "rounds": self.rounds, "batch_id": self.batch_id,
            "timeline": self.timeline,
        }


def _generate_lot(rng: random.Random, true_p: float, plan: stats.PlanParams,
                  seed: int) -> list[dict]:
    """Fictional lot; without replacement uses a fixed shuffled finite lot."""
    if plan.sampling == "without_replacement":
        if plan.N is None:
            raise stats.PlanError("N required for without-replacement lot")
        defects = round(true_p * plan.N)
        lot = ([True] * defects + [False] * (plan.N - defects))
        rng.shuffle(lot)
        return [{"item_id": f"seed{seed}-item{idx:05d}",
                 "defective": d} for idx, d in enumerate(lot)]
    # Infinite stream; ids remain unique so the state machine's no-repeated-
    # information rule is exercised exactly as in production.
    out = []
    for idx in range(10_000):
        out.append({"item_id": f"seed{seed}-item{idx:05d}",
                    "defective": rng.random() < true_p})
    return out


def run_single(plan: stats.PlanParams, true_p: float, seed: int,
               round_size: int = 1, batch_id: str | None = None,
               cap: int | None = None) -> SimResult:
    """Run one sequential inspection until accept/reject.

    The simulation builds the *same* append-only event log the web UI would
    produce (round open/close, per-item events with the engine's LLR
    increments, a decision event) and then runs one full ``state.rebuild``
    as an end-of-run audit.  Keeping the running (n, k) tally instead of
    replaying the whole log after every item avoids an O(n^2) hot path,
    while the final rebuild proves the generated log is accepted by the
    production state machine.

    A run that reaches ``cap`` without deciding is returned as *censored*
    (decision ``continue``) rather than raised: a conservative confidence
    sequence can legitimately need more than the cap to separate an
    indifferent lot.
    """
    rng = random.Random(seed)
    lot = _generate_lot(rng, true_p, plan, seed)
    at = f"2026-01-01T00:00:{seed % 60:02d}+00:00"
    cap = cap if cap is not None else (plan.N if plan.N is not None else 5_000)
    batch_id = batch_id or f"SIM-{seed}"

    events = [{"type": "batch_created", "at": at},
              {"type": "round_opened", "at": at, "round": 1}]
    n = k = 0
    seq = 0
    round_no = 1
    since_round = 0
    status = state.CONTINUE
    decision_snapshot = None

    for item in lot[:cap]:
        if status != state.CONTINUE:
            break
        item_id = item["item_id"]
        defective = item["defective"]
        if plan.method == "cs_beta_mixture":
            inc = None
        else:
            inc = stats.llr_increment(plan, n, k, defective)
        seq += 1
        events.append({
            "type": "item_recorded", "at": at, "seq": seq,
            "item_id": item_id, "defective": defective,
            "round": round_no, "llr_increment": inc,
        })
        n += 1
        k += 1 if defective else 0
        since_round += 1

        snap = stats.snapshot(plan, n, k)
        if snap["decision"] != state.CONTINUE:
            events.append({
                "type": "decision", "at": at,
                "decision": snap["decision"], "n": n, "k": k,
                "round": round_no, "snapshot": state._jsonable(snap),
            })
            events.append({"type": "round_closed", "at": at,
                           "round": round_no})
            status = snap["decision"]
            decision_snapshot = snap
            break

        # close and reopen a sampling round at the requested granularity
        if round_size > 1 and since_round >= round_size and n < cap:
            events.append({"type": "round_closed", "at": at,
                           "round": round_no})
            round_no += 1
            events.append({"type": "round_opened", "at": at,
                           "round": round_no})
            since_round = 0

    if status == state.CONTINUE and events[-1]["type"] != "round_closed":
        events.append({"type": "round_closed", "at": at, "round": round_no})

    batch = {
        "batch_id": batch_id,
        "created_at": at,
        "plan": plan.to_dict(),
        "assumptions": stats.assumptions_text(plan),
        "lot_note": f"simulated p={true_p}",
        "events": events,
    }
    # Single full replay: this is the production consistency gate.
    derived = state.rebuild(batch)
    if decision_snapshot is not None and derived["status"] != status:
        raise AssertionError("simulation log failed the production replay gate")

    timeline = [{
        "seq": h["seq"], "item_id": h["item_id"],
        "defective": next(i["defective"] for i in derived["items"]
                          if i["item_id"] == h["item_id"] and i["active"]),
        "n": h["n"], "k": h["k"],
        "statistic": h.get("statistic"),
        "cs_lower": h.get("cs_lower"), "cs_upper": h.get("cs_upper"),
        "decision": h["decision"],
    } for h in derived["history"]]

    return SimResult(
        seed=seed, true_p=true_p, decision=derived["status"],
        n=derived["current"]["n"], k=derived["current"]["k"],
        rounds=derived["round"], batch_id=derived["batch_id"],
        timeline=timeline,
    )


def run_grid(plan: stats.PlanParams, true_ps: list[float],
             replications: int, base_seed: int = 20261124,
             round_size: int = 1, cap: int | None = None) -> dict:
    """Monte Carlo operating-characteristic grid.

    Seeds are deterministic and shared by position: plan A and plan B at the
    same (true_p, replication) draw identical fictional lots, which makes
    between-plan comparisons paired and reproducible bit-for-bit.

    ``censor_rate`` reports runs still undecided at the cap (relevant for
    conservative confidence sequences on indifferent lots).
    """
    cells = []
    for true_p in true_ps:
        decisions, ns, ks = [], [], []
        for rep in range(replications):
            seed = base_seed + rep * 10_003 + int(round(true_p * 1_000_000))
            res = run_single(plan, true_p, seed,
                             round_size=round_size, cap=cap)
            decisions.append(res.decision)
            ns.append(res.n)
            ks.append(res.k)
        n_accept = decisions.count(stats.ACCEPT)
        n_reject = decisions.count(stats.REJECT)
        n_censor = decisions.count(stats.CONTINUE)
        cells.append({
            "true_p": true_p,
            "replications": replications,
            "accept_rate": n_accept / replications,
            "reject_rate": n_reject / replications,
            "censor_rate": n_censor / replications,
            "avg_n": sum(ns) / replications,
            "min_n": min(ns), "max_n": max(ns),
            "avg_defectives": sum(ks) / replications,
        })
    return {"plan": plan.to_dict(),
            "replications": replications,
            "base_seed": base_seed,
            "round_size": round_size,
            "cells": cells}


def compare_plans(plans: list[stats.PlanParams], true_ps: list[float],
                  replications: int, base_seed: int = 20261124,
                  round_size: int = 1, cap: int | None = None) -> list[dict]:
    """Run the same fictional lots through several fixed risk configs."""
    return [run_grid(p, true_ps, replications, base_seed, round_size, cap)
            for p in plans]
