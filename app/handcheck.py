"""Hand-calculation cross-checks for every per-round statistic.

The production engine (app.stats) computes LLRs from closed-form sufficient
statistics.  This module deliberately re-derives the same numbers a different
way -- one multiplicative step at a time, using only the stated likelihood
ratio of the next draw -- and asserts the two agree at every round.  A few
fully hand-worked numeric trajectories are included as constants so a human
can audit the arithmetic with a calculator.
"""

from __future__ import annotations

import math

from . import state, stats

# ---------------------------------------------------------------------------
# Hand-worked constants (standard binomial plan: p0=.03, p1=.10,
# alpha=.05, beta=.10).  Boundary logs:
#   B = ln(beta/(1-alpha)) = ln(0.10/0.95)
#   A = ln((1-beta)/alpha) = ln(0.90/0.05)
# ---------------------------------------------------------------------------

P0, P1, ALPHA, BETA = 0.03, 0.10, 0.05, 0.10
HAND_BOUNDARY_LOWER = math.log(BETA / (1 - ALPHA))
HAND_BOUNDARY_UPPER = math.log((1 - BETA) / ALPHA)

# Step LLR contributions used in the hand trajectory.
STEP_DEFECT = math.log(P1 / P0)
STEP_GOOD = math.log((1 - P1) / (1 - P0))

# Hand trajectory: outcomes G, D, G, G, D (G=conforming, D=defective).
# The expected running LLR is derived from the step constants (no hand-typed
# floats that could drift from the formula).
_HAND_LABELS = ["G", "D", "G", "G", "D"]
HAND_PATH = []
_running = 0.0
_n = _k = 0
for _label in _HAND_LABELS:
    _running += STEP_DEFECT if _label == "D" else STEP_GOOD
    _n += 1
    _k += 1 if _label == "D" else 0
    HAND_PATH.append((_label, _n, _k, _running))
del _label, _running, _n, _k


def standard_plan():
    return stats.make_plan(
        method="sprt_binomial", sampling="with_replacement",
        p0=P0, p1=P1, alpha=ALPHA, beta=BETA, name="hand-check")


def verify_hand_trajectory() -> list[dict]:
    """Drive the real state machine along the hand path and compare."""
    plan = standard_plan()
    batch = state.create_batch(plan, batch_id="HAND-1")
    rows = []
    running = 0.0
    n = k = 0
    for idx, (label, exp_n, exp_k, exp_llr) in enumerate(HAND_PATH):
        defective = label == "D"
        batch, _ = state.record_items(
            batch, [{"item_id": f"hand-{idx}", "defective": defective}])
        view = state.rebuild(batch)
        running += STEP_DEFECT if defective else STEP_GOOD
        n += 1
        k += 1 if defective else 0
        engine_llr = view["current"]["statistic"]
        closed_form = stats.binomial_llr(plan, n, k)
        # 1e-9 is far tighter than any hand/calculator precision while still
        # tolerating floating-point reordering (sum of steps vs closed form).
        match = (
            abs(running - engine_llr) < 1e-9
            and abs(closed_form - engine_llr) < 1e-9
            and abs(exp_llr - running) < 1e-9)
        rows.append({
            "round_item": idx + 1, "outcome": label, "n": n, "k": k,
            "hand_running_llr": running,
            "engine_llr": engine_llr,
            "closed_form_llr": closed_form,
            "expected_from_constants": exp_llr,
            "match": match,
        })
    return rows


def verify_boundary_constants(plan: stats.PlanParams) -> dict:
    lower, upper = stats.sprt_boundaries(plan)
    return {
        "hand_lower": HAND_BOUNDARY_LOWER,
        "engine_lower": lower,
        "hand_upper": HAND_BOUNDARY_UPPER,
        "engine_upper": upper,
        "match": (abs(lower - HAND_BOUNDARY_LOWER) < 1e-12
                  and abs(upper - HAND_BOUNDARY_UPPER) < 1e-12),
    }


def verify_hypergeometric_steps(plan: stats.PlanParams,
                                outcomes: list[bool]) -> list[dict]:
    """Step-by-step hypergeometric LLR vs the closed-form LLR each round."""
    rows = []
    n = k = 0
    running = 0.0
    for idx, defective in enumerate(outcomes, start=1):
        inc = stats.llr_increment(plan, n, k, defective)
        running += inc
        n += 1
        k += 1 if defective else 0
        closed = stats.hypergeometric_llr(plan, n, k)
        rows.append({
            "item": idx, "defective": defective, "n": n, "k": k,
            "step_increment": inc,
            "running_llr": running,
            "closed_form_llr": closed,
            "match": abs(running - closed) < 1e-9 or running == closed,
        })
    return rows


def verify_cs_steps(plan: stats.PlanParams,
                    outcomes: list[bool]) -> list[dict]:
    """Recompute the CS independently from the predictive product form.

    N_n(p) = m_n / L_n(p) with m_n = k!(n-k)!/(n+1)!; the log of m_n is
    accumulated from the Beta(1,1) predictive probabilities rather than the
    closed-form gamma expression, then compared to the engine.
    """
    rows = []
    n = k = 0
    log_mixture = 0.0
    for idx, defective in enumerate(outcomes, start=1):
        if defective:
            predictive = (k + 1) / (idx + 1)
            k += 1
        else:
            predictive = (idx - k) / (idx + 1)
        log_mixture += math.log(predictive)
        n += 1
        for p in (plan.p0, plan.p1):
            engine = stats.log_mixture_lr(p, n, k)
            log_point = k * math.log(p) + (n - k) * math.log(1 - p)
            independent = log_mixture - log_point
            if abs(engine - independent) > 1e-12:
                raise AssertionError(
                    f"CS mismatch at item {idx}, p={p}: "
                    f"engine={engine}, independent={independent}")
        lo, hi = stats.cs_bounds(n, k, plan.alpha_cs)
        rows.append({"item": idx, "n": n, "k": k,
                     "cs_lower": lo, "cs_upper": hi,
                     "decision": stats.snapshot(plan, n, k)["decision"]})
    return rows


def full_report() -> dict:
    """Run every hand check and return a structured, all-must-match report."""
    plan = standard_plan()
    hand_rows = verify_hand_trajectory()
    if not all(r["match"] for r in hand_rows):
        raise AssertionError("binomial hand trajectory mismatch")
    boundaries = verify_boundary_constants(plan)
    if not boundaries["match"]:
        raise AssertionError("boundary constant mismatch")

    hg_plan = stats.make_plan(
        method="sprt_hypergeometric", sampling="without_replacement",
        p0=0.03, p1=0.10, alpha=0.05, beta=0.10, N=500)
    hg_rows = verify_hypergeometric_steps(
        hg_plan, [False, True, False, False, True, False, False, False])
    if not all(r["match"] for r in hg_rows):
        raise AssertionError("hypergeometric step mismatch")

    cs_plan = stats.make_plan(
        method="cs_beta_mixture", sampling="with_replacement",
        p0=0.03, p1=0.10, alpha_cs=0.05)
    cs_rows = verify_cs_steps(
        cs_plan, [False, False, True, False, False, True, False, False])

    return {
        "binomial_hand_trajectory": hand_rows,
        "boundary_constants": boundaries,
        "hypergeometric_steps": hg_rows,
        "confidence_sequence_steps": cs_rows,
        "all_match": True,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(full_report(), indent=2, ensure_ascii=False))
