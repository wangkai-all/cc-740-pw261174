"""Sequential inspection statistics.

Three decision engines, all operating on simple sufficient summaries (n, k):

* ``sprt_binomial``      -- Wald SPRT for iid Bernoulli data (sampling WITH
                            replacement, or effectively infinite lot).
* ``sprt_hypergeometric`` -- SPRT on the hypergeometric likelihood, for
                            sampling WITHOUT replacement from a finite lot.
* ``cs_beta_mixture``    -- a guaranteed confidence sequence (Ville's
                            inequality, uniform Beta(1,1) mixture), iid
                            Bernoulli, with-replacement use only.

Every public function is pure: the same (plan, n, k) always yields the same
answer, which is what lets the state machine rebuild decisions by replaying
the event log.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

METHODS = ("sprt_binomial", "sprt_hypergeometric", "cs_beta_mixture")
SAMPLING_MODES = ("with_replacement", "without_replacement")

CONTINUE = "continue"
ACCEPT = "accepted"   # lot accepted (H0: good lot not rejected)
REJECT = "rejected"   # lot rejected (H1: bad lot supported)


class PlanError(ValueError):
    """Raised when a plan's parameters are inconsistent or unusable."""


@dataclass(frozen=True)
class PlanParams:
    method: str
    sampling: str
    p0: float                      # acceptable quality (AQL)
    p1: float                      # rejectable quality (RQL), p1 > p0
    alpha: float                   # producer / false-reject target (SPRT)
    beta: float                    # consumer / false-accept target (SPRT)
    alpha_cs: Optional[float] = None  # CS miscoverage level
    N: Optional[int] = None        # finite lot size (hypergeometric)
    name: str = ""

    def to_dict(self) -> dict:
        # NaN marks an inapplicable field (alpha/beta for the CS method);
        # emit JSON null rather than a non-standard NaN literal.
        def clean(v):
            return None if isinstance(v, float) and math.isnan(v) else v
        return {
            "method": self.method,
            "sampling": self.sampling,
            "p0": self.p0,
            "p1": self.p1,
            "alpha": clean(self.alpha),
            "beta": clean(self.beta),
            "alpha_cs": self.alpha_cs,
            "N": self.N,
            "name": self.name,
        }


def make_plan(**kw) -> PlanParams:
    """Validate raw keyword arguments and build a PlanParams."""
    method = kw.get("method")
    if method not in METHODS:
        raise PlanError(f"unknown method {method!r}; choose one of {METHODS}")
    sampling = kw.get("sampling")
    if sampling not in SAMPLING_MODES:
        raise PlanError(f"sampling must be one of {SAMPLING_MODES}")

    try:
        p0 = float(kw["p0"])
        p1 = float(kw["p1"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanError("p0 and p1 must be numbers") from exc
    if not (0.0 < p0 < p1 < 1.0):
        raise PlanError("require 0 < p0 < p1 < 1")

    alpha = beta = None
    alpha_cs = None
    if method.startswith("sprt"):
        try:
            alpha = float(kw["alpha"])
            beta = float(kw["beta"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanError("alpha and beta must be numbers") from exc
        if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
            raise PlanError("require 0 < alpha, beta < 1")
        if alpha + beta >= 1.0:
            raise PlanError("alpha + beta must be < 1")
    if method == "cs_beta_mixture":
        try:
            alpha_cs = float(kw.get("alpha_cs", kw.get("alpha")))
        except (TypeError, ValueError) as exc:
            raise PlanError("alpha_cs must be a number") from exc
        if not (0.0 < alpha_cs < 1.0):
            raise PlanError("require 0 < alpha_cs < 1")
        if sampling != "with_replacement":
            raise PlanError(
                "cs_beta_mixture assumes iid draws; use with_replacement "
                "(or an SPRT method for finite lots)"
            )

    N = kw.get("N")
    if method == "sprt_hypergeometric":
        if sampling != "without_replacement":
            raise PlanError("hypergeometric SPRT requires without_replacement")
        if N is None:
            raise PlanError("finite lot size N is required for hypergeometric SPRT")
        N = int(N)
        if N < 2:
            raise PlanError("lot size N must be >= 2")
        D0, D1 = defect_counts(PlanParams(
            method=method, sampling=sampling, p0=p0, p1=p1,
            alpha=alpha or 0.05, beta=beta or 0.10, N=N,
        ))
        if D1 <= D0:
            raise PlanError(
                f"N={N} too small to separate p0={p0}, p1={p1}: "
                f"D0={D0}, D1={D1}; increase N or move thresholds apart"
            )
    elif sampling == "without_replacement":
        raise PlanError("without_replacement requires sprt_hypergeometric")
    elif N is not None:
        N = int(N)
        if N < 1:
            raise PlanError("lot size N must be >= 1")

    return PlanParams(
        method=method, sampling=sampling, p0=p0, p1=p1,
        alpha=alpha if alpha is not None else float("nan"),
        beta=beta if beta is not None else float("nan"),
        alpha_cs=alpha_cs, N=N, name=str(kw.get("name", "")),
    )


# ---------------------------------------------------------------------------
# SPRT boundary machinery
# ---------------------------------------------------------------------------

def defect_counts(plan: PlanParams) -> tuple[int, int]:
    """Integer defective counts (D0, D1) implied by a finite lot."""
    if plan.N is None:
        raise PlanError("lot size N required")
    return round(plan.N * plan.p0), round(plan.N * plan.p1)


def sprt_boundaries(plan: PlanParams) -> tuple[float, float]:
    """Wald boundaries (B, A) on the log-likelihood ratio.

    H0 (good lot) accepted when LLR <= B; H1 (bad lot) when LLR >= A.
    With A = ln((1-beta)/alpha), B = ln(beta/(1-alpha)) the realised error
    probabilities are bounded by alpha/(1-beta) and beta/(1-alpha)
    respectively (Wald's inequality for likelihood-ratio martingales, valid
    for both the iid and the without-replacement likelihood ratios).
    """
    upper = math.log((1.0 - plan.beta) / plan.alpha)
    lower = math.log(plan.beta / (1.0 - plan.alpha))
    return lower, upper


def _log_falling_factorial(a: int, m: int) -> float:
    """log(a * (a-1) * ... * (a-m+1)); m=0 -> 0. Caller guarantees a>=m>=0."""
    if m == 0:
        return 0.0
    return math.lgamma(a + 1) - math.lgamma(a - m + 1)


def binomial_llr(plan: PlanParams, n: int, k: int) -> float:
    """Closed-form iid Bernoulli log likelihood ratio (order independent)."""
    if n == 0:
        return 0.0
    return (
        k * math.log(plan.p1 / plan.p0)
        + (n - k) * math.log((1.0 - plan.p1) / (1.0 - plan.p0))
    )


def hypergeometric_llr(plan: PlanParams, n: int, k: int) -> float:
    """Log LR of a specific observed sample under H1 vs H0.

    P_H(sequence) = D_(k) (N-D)_(n-k) / N_(n); the N_(n) denominator is
    identical under both hypotheses and cancels.  Impossible observations
    under one hypothesis return +/-inf exactly.
    """
    D0, D1 = defect_counts(plan)
    N = plan.N
    good0, good1 = N - D0, N - D1
    g0, g1 = n - k, n - k  # conforming draws

    # H0 cannot have produced this sample -> overwhelming evidence for H1.
    if k > D0 or g0 > good0:
        return math.inf
    # H1 cannot have produced it -> evidence for H0 (H0 checked first).
    if k > D1 or g1 > good1:
        return -math.inf

    log_num = _log_falling_factorial(D1, k) + _log_falling_factorial(good1, g1)
    log_den = _log_falling_factorial(D0, k) + _log_falling_factorial(good0, g0)
    return log_num - log_den


def llr(plan: PlanParams, n: int, k: int) -> float:
    if plan.method == "sprt_binomial":
        return binomial_llr(plan, n, k)
    if plan.method == "sprt_hypergeometric":
        return hypergeometric_llr(plan, n, k)
    raise PlanError(f"{plan.method} has no scalar LLR")


def llr_increment(plan: PlanParams, prev_n: int, prev_k: int,
                  defective: bool) -> float:
    """One-step LLR contribution, for the per-round audit trail.

    For the iid test this is constant per outcome.  For the without-
    replacement test it depends on the remaining composition of the lot,
    which is why withdrawals must trigger a full recompute.
    """
    x = 1 if defective else 0
    if plan.method == "sprt_binomial":
        if defective:
            return math.log(plan.p1 / plan.p0)
        return math.log((1 - plan.p1) / (1 - plan.p0))

    if plan.method == "sprt_hypergeometric":
        D0, D1 = defect_counts(plan)
        N = plan.N
        k_prev, g_prev = prev_k, prev_n - prev_k
        if defective:
            num, den = D1 - k_prev, D0 - k_prev
            if den <= 0:
                return math.inf       # H0 exhausted its defectives
            if num <= 0:
                return -math.inf
        else:
            num = (N - D1) - g_prev
            den = (N - D0) - g_prev
            if den <= 0:
                return math.inf
            if num <= 0:
                return -math.inf
        return math.log(num / den)

    raise PlanError(f"{plan.method} has no LLR increments")


# ---------------------------------------------------------------------------
# Beta(1,1) mixture confidence sequence
# ---------------------------------------------------------------------------

def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def log_mixture_lr(p: float, n: int, k: int) -> float:
    """log N_n(p), the Beta(1,1) predictive-mixture likelihood-ratio.

    For an ordered Bernoulli path with k defectives among n, the uniform
    Beta(1,1) prior's sequential predictive probability of that path is

        m_n = k! (n-k)! / (n+1)!
            = Beta(k+1, n-k+1) / (n+1).

    The likelihood ratio of the mixture against the point hypothesis is

        N_n(p) = m_n / [p^k (1-p)^(n-k)].

    N_n is a nonnegative martingale under iid Bernoulli(p) with E_p[N]=1
    (the test suite checks this in log space at several n and the one-step
    predictive telescoping identity).  N is U-shaped in p with its minimum
    at the MLE.  Ville's inequality on the martingale's upper tail gives
    the time-uniform guarantee

        P_p(exists n: N_n(p) >= 1/alpha) <= alpha,

    so the (1-alpha) confidence sequence is the interior set
    {p : N_n(p) < 1/alpha}.  The 1/(n+1) factor versus the plain Beta
    function is essential: it is what makes E[N]=1.
    """
    if n == 0:
        return 0.0
    log_point = k * math.log(p) + (n - k) * math.log(1.0 - p)
    log_mixture = (math.lgamma(k + 1) + math.lgamma(n - k + 1)
                   - math.lgamma(n + 2))
    return log_mixture - log_point


def _bisect_root(n: int, k: int, threshold: float,
                 lo: float, hi: float, increasing: bool) -> float:
    """Solve log N_n(p) = log(1/alpha) on a monotone arm of the U shape."""
    for _ in range(60):
        mid = (lo + hi) / 2.0
        above = log_mixture_lr(mid, n, k) > threshold
        if increasing:
            if above:
                hi = mid
            else:
                lo = mid
        else:
            if above:
                lo = mid
            else:
                hi = mid
    return (lo + hi) / 2.0


def cs_bounds(n: int, k: int, alpha_cs: float) -> tuple[float, float]:
    """Lower/upper 1-alpha_cs confidence-sequence bounds after n items.

    The confidence set {p : N_n(p) < 1/alpha_cs} is the interior of the
    U-shaped martingale: if even its minimum (at the MLE) is on or above
    the boundary, no p is excluded and the set is the whole [0,1].
    """
    if n == 0:
        return 0.0, 1.0
    threshold = -math.log(alpha_cs)
    p_hat = k / n
    eps = 1e-12

    if k == 0:
        # N increasing in p; one crossing gives the upper bound.
        if log_mixture_lr(1 - eps, n, k) <= threshold:
            return 0.0, 1.0
        upper = _bisect_root(n, k, threshold, eps, 1 - eps,
                             increasing=True)
        return 0.0, upper
    if k == n:
        # N decreasing in p; one crossing gives the lower bound.
        if log_mixture_lr(eps, n, k) <= threshold:
            return 0.0, 1.0
        lower = _bisect_root(n, k, threshold, eps, 1 - eps,
                             increasing=False)
        return lower, 1.0

    if log_mixture_lr(p_hat, n, k) >= threshold:
        return 0.0, 1.0

    # left arm of the U: N falls as p rises toward the MLE
    lower = _bisect_root(n, k, threshold, eps, p_hat, increasing=False)
    # right arm: N rises as p moves past the MLE
    upper = _bisect_root(n, k, threshold, p_hat, 1 - eps, increasing=True)
    return lower, upper


# ---------------------------------------------------------------------------
# Unified snapshot / decision
# ---------------------------------------------------------------------------

def snapshot(plan: PlanParams, n: int, k: int) -> dict:
    """Full statistical state after n inspected items, k defective.

    Returns n, k, the method-specific statistic, the active boundaries and
    one of continue / accepted / rejected.  Pure function of (plan, n, k).
    """
    if k < 0 or n < 0 or k > n:
        raise PlanError(f"infeasible counts n={n}, k={k}")
    if plan.N is not None and n > plan.N:
        raise PlanError(f"n={n} exceeds lot size N={plan.N}")

    base = {"n": n, "k": k, "method": plan.method}

    if plan.method in ("sprt_binomial", "sprt_hypergeometric"):
        lower, upper = sprt_boundaries(plan)
        value = llr(plan, n, k)
        if value >= upper:
            decision = REJECT
        elif value <= lower:
            decision = ACCEPT
        else:
            decision = CONTINUE
        base.update({
            "statistic": value,
            "statistic_name": "log_likelihood_ratio",
            "lower_boundary": lower,
            "upper_boundary": upper,
            "decision": decision,
        })
        if plan.method == "sprt_hypergeometric":
            D0, D1 = defect_counts(plan)
            base.update({"D0": D0, "D1": D1})
        return base

    # confidence sequence
    lower, upper = cs_bounds(n, k, plan.alpha_cs)
    if n > 0 and upper <= plan.p0:
        decision = ACCEPT
    elif n > 0 and lower >= plan.p1:
        decision = REJECT
    else:
        decision = CONTINUE
    base.update({
        "statistic": None,
        "statistic_name": "confidence_sequence",
        "cs_lower": lower,
        "cs_upper": upper,
        "p0": plan.p0,
        "p1": plan.p1,
        "decision": decision,
    })
    return base


def projection(plan: PlanParams, n: int, k: int) -> dict:
    """Deterministic look-ahead used to plan the next sampling round.

    Returns the smallest number of additional items that would force a
    decision if *all* were conforming (accept path) or all defective
    (reject path), plus remaining lot capacity.  None means unreachable
    before the lot is exhausted.

    Along either all-conforming or all-defective continuation the decision
    statistic moves monotonically toward one boundary, so the first hitting
    time is found by binary search rather than a linear walk (a linear walk
    would re-solve the confidence-sequence roots once per candidate step).
    """
    capacity = None if plan.N is None else plan.N - n
    out = {"remaining_capacity": capacity,
           "min_additional_all_conforming": None,
           "min_additional_all_defective": None}

    if snapshot(plan, n, k)["decision"] != CONTINUE:
        return out

    def decision_at(step: int, defective: bool) -> str:
        nn = n + step
        kk = k + (step if defective else 0)
        return snapshot(plan, nn, kk)["decision"]

    def steps_to(target: str, defective: bool) -> Optional[int]:
        limit = 10_000 if capacity is None else capacity
        # First establish that the target is actually reachable; the path is
        # monotone so a non-target terminal decision means it never hits it.
        final = decision_at(limit, defective)
        if final == CONTINUE:
            return None
        if final != target:
            return None
        lo, hi = 1, limit
        while lo < hi:
            mid = (lo + hi) // 2
            if decision_at(mid, defective) == target:
                hi = mid
            else:
                lo = mid + 1
        return lo

    out["min_additional_all_conforming"] = steps_to(ACCEPT, False)
    out["min_additional_all_defective"] = steps_to(REJECT, True)
    return out


def assumptions_text(plan: PlanParams) -> str:
    """Human-readable assumptions, stored verbatim in the plan record."""
    if plan.method == "sprt_binomial":
        return (
            "iid Bernoulli inspections with replacement (or lot large enough "
            "that depletion is negligible); simple hypotheses H0:p=p0 vs "
            "H1:p=p1; Wald boundaries guarantee real false-reject <= "
            "alpha/(1-beta) and real false-accept <= beta/(1-alpha)."
        )
    if plan.method == "sprt_hypergeometric":
        return (
            "random draws without replacement from a finite lot of N items; "
            "simple hypotheses on defective counts D0=round(N*p0) vs "
            "D1=round(N*p1); exact hypergeometric likelihood ratio; Wald "
            "error guarantees apply to the LR martingale."
        )
    return (
        "iid Bernoulli inspections with replacement; Beta(1,1) mixture "
        "confidence sequence via Ville's inequality: P(any n excludes true "
        "p) <= alpha_cs; accept only when the whole interval is <= p0, "
        "reject only when it is >= p1."
    )
