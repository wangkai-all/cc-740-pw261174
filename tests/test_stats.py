"""Unit tests for the statistics engines."""

import math
import unittest

from app import stats
from app.stats import PlanError


class TestPlanValidation(unittest.TestCase):
    def test_ordering_required(self):
        with self.assertRaises(PlanError):
            stats.make_plan(method="sprt_binomial", sampling="with_replacement",
                            p0=0.1, p1=0.05, alpha=0.05, beta=0.1)

    def test_hypergeometric_needs_N(self):
        with self.assertRaises(PlanError):
            stats.make_plan(method="sprt_hypergeometric",
                            sampling="without_replacement",
                            p0=0.03, p1=0.1, alpha=0.05, beta=0.1)

    def test_tiny_lot_rejected(self):
        with self.assertRaises(PlanError):
            stats.make_plan(method="sprt_hypergeometric",
                            sampling="without_replacement",
                            p0=0.03, p1=0.1, alpha=0.05, beta=0.1, N=5)

    def test_cs_requires_iid(self):
        with self.assertRaises(PlanError):
            stats.make_plan(method="cs_beta_mixture",
                            sampling="without_replacement",
                            p0=0.03, p1=0.1, alpha_cs=0.05, N=100)


class TestBinomialSPRT(unittest.TestCase):
    def setUp(self):
        self.plan = stats.make_plan(
            method="sprt_binomial", sampling="with_replacement",
            p0=0.03, p1=0.10, alpha=0.05, beta=0.10)

    def test_boundary_values(self):
        lower, upper = stats.sprt_boundaries(self.plan)
        self.assertAlmostEqual(lower, math.log(0.1 / 0.95))
        self.assertAlmostEqual(upper, math.log(0.9 / 0.05))

    def test_llr_formula(self):
        # hand: k ln(p1/p0) + (n-k) ln((1-p1)/(1-p0))
        n, k = 10, 2
        want = 2 * math.log(0.1 / 0.03) + 8 * math.log(0.9 / 0.97)
        self.assertAlmostEqual(stats.binomial_llr(self.plan, n, k), want)

    def test_increment_matches_total(self):
        # path independence for iid LLR
        total = 0.0
        n = k = 0
        for defective in [True, False, True, False, False]:
            total += stats.llr_increment(self.plan, n, k, defective)
            n += 1
            k += 1 if defective else 0
        self.assertAlmostEqual(total, stats.binomial_llr(self.plan, n, k))

    def test_decisions(self):
        # all defective -> H1 quickly; all conforming -> H0 eventually
        plan = self.plan
        n = 1
        while stats.snapshot(plan, n, n)["decision"] == stats.CONTINUE:
            n += 1
        self.assertEqual(stats.snapshot(plan, n, n)["decision"], stats.REJECT)

        n = 1
        while stats.snapshot(plan, n, 0)["decision"] == stats.CONTINUE:
            n += 1
        self.assertEqual(stats.snapshot(plan, n, 0)["decision"], stats.ACCEPT)

    def test_boundary_order(self):
        lower, upper = stats.sprt_boundaries(self.plan)
        self.assertLess(lower, 0 < upper)


class TestHypergeometricSPRT(unittest.TestCase):
    def setUp(self):
        self.plan = stats.make_plan(
            method="sprt_hypergeometric", sampling="without_replacement",
            p0=0.03, p1=0.10, alpha=0.05, beta=0.10, N=500)

    def test_llr_against_brute_force(self):
        # P(sequence) = D_(k)(N-D)_(n-k) / N_(n); ratio cancels the denom.
        N, D0, D1 = 500, 15, 50
        n, k = 10, 2
        want = (stats._log_falling_factorial(D1, k)
                + stats._log_falling_factorial(N - D1, n - k)
                - stats._log_falling_factorial(D0, k)
                - stats._log_falling_factorial(N - D0, n - k))
        self.assertAlmostEqual(stats.hypergeometric_llr(self.plan, n, k), want)

    def test_increment_path_dependence(self):
        # unlike the iid case, the same (n,k) reached by different orders
        # must still give the same closed-form LLR, but increments differ.
        plan = self.plan
        seq_a = [True, False, False, False]
        seq_b = [False, False, False, True]
        for seq in (seq_a, seq_b):
            n = k = 0
            total = 0.0
            increments = []
            for d in seq:
                inc = stats.llr_increment(plan, n, k, d)
                increments.append(inc)
                total += inc
                n += 1
                k += 1 if d else 0
            self.assertAlmostEqual(total, stats.hypergeometric_llr(plan, n, k))
            if seq is seq_a:
                inc_a = increments
            else:
                inc_b = increments
        self.assertNotEqual(inc_a, inc_b)  # depletion changes step weights

    def test_exhaustion_gives_infinity(self):
        # after D0 defectives seen, another defective is impossible under H0
        plan = stats.make_plan(
            method="sprt_hypergeometric", sampling="without_replacement",
            p0=0.03, p1=0.10, alpha=0.05, beta=0.10, N=200)
        D0, _ = stats.defect_counts(plan)  # D0 = 6
        self.assertEqual(stats.llr_increment(plan, D0, D0, True), math.inf)
        self.assertEqual(stats.hypergeometric_llr(plan, D0 + 1, D0 + 1),
                         math.inf)

    def test_full_lot_exact(self):
        # inspecting the whole lot identifies the true count deterministically
        plan = self.plan
        snap = stats.snapshot(plan, 500, 50)   # exactly D1
        self.assertEqual(snap["decision"], stats.REJECT)
        snap0 = stats.snapshot(plan, 500, 15)  # exactly D0
        self.assertEqual(snap0["decision"], stats.ACCEPT)


class TestConfidenceSequence(unittest.TestCase):
    def setUp(self):
        self.alpha = 0.05
        self.plan = stats.make_plan(
            method="cs_beta_mixture", sampling="with_replacement",
            p0=0.03, p1=0.10, alpha_cs=self.alpha)

    def test_bounds_are_ordered_and_valid(self):
        for n, k in [(1, 0), (5, 2), (50, 10), (100, 0), (100, 100)]:
            lo, hi = stats.cs_bounds(n, k, self.alpha)
            self.assertTrue(0.0 <= lo <= hi <= 1.0)

    def test_martingale_has_unit_expectation(self):
        # N_n = mixture/point is the Ville martingale; E_p[N_n] = 1.
        from math import comb
        p = 0.07
        for nn in (20, 100, 300):
            log_terms = [
                math.log(comb(nn, kk))
                + kk * math.log(p) + (nn - kk) * math.log(1 - p)
                + stats.log_mixture_lr(p, nn, kk)
                for kk in range(nn + 1)]
            m = max(log_terms)
            expectation = math.exp(m) * sum(
                math.exp(t - m) for t in log_terms)
            self.assertAlmostEqual(expectation, 1.0, places=6)

    def test_predictive_telescoping_identity(self):
        # building N one predictive draw at a time reproduces the closed form
        import random
        rng = random.Random(99)
        p = 0.07
        for _ in range(10):
            n = rng.randint(5, 60)
            path = [rng.random() < 0.2 for _ in range(n)]
            k = 0
            log_n = 0.0
            for t, defective in enumerate(path, start=1):
                if defective:
                    mix_prob = (k + 1) / (t + 1)
                    log_n += math.log(mix_prob) - math.log(p)
                    k += 1
                else:
                    mix_prob = (t - k) / (t + 1)
                    log_n += math.log(mix_prob) - math.log(1 - p)
            self.assertAlmostEqual(
                log_n, stats.log_mixture_lr(p, n, sum(path)), places=10)

    def test_roots_solve_equation(self):
        n, k = 500, 35
        lo, hi = stats.cs_bounds(n, k, self.alpha)
        self.assertGreater(lo, 0.0)
        self.assertLess(hi, 1.0)
        threshold = -math.log(self.alpha)
        self.assertAlmostEqual(stats.log_mixture_lr(lo, n, k), threshold,
                               places=8)
        self.assertAlmostEqual(stats.log_mixture_lr(hi, n, k), threshold,
                               places=8)

    def test_width_shrinks_with_n(self):
        # along a single realised path the interval never widens (a CS is a
        # nested sequence of confidence sets)
        import random
        rng = random.Random(777)
        stream = [rng.random() < 0.06 for _ in range(1500)]
        widths, k = [], 0
        for idx, defective in enumerate(stream, start=1):
            k += 1 if defective else 0
            if idx in (300, 600, 1200, 1500):
                lo, hi = stats.cs_bounds(idx, k, self.alpha)
                if hi - lo < 1.0:
                    widths.append(hi - lo)
        self.assertGreaterEqual(len(widths), 2)
        self.assertEqual(widths, sorted(widths, reverse=True))

    def test_exact_fixed_time_coverage(self):
        # sum the true Bernoulli pmf over exactly the k values whose set
        # contains p: must be >= 1-alpha (the mixture is conservative).
        from math import comb
        p = 0.07
        for nn in (100, 300, 500):
            coverage = sum(
                comb(nn, kk) * p ** kk * (1 - p) ** (nn - kk)
                for kk in range(nn + 1)
                if _contains(stats.cs_bounds(nn, kk, self.alpha), p))
            self.assertGreaterEqual(coverage, 1 - self.alpha - 1e-9)

    def test_time_uniform_coverage(self):
        # the actual Ville property: along whole paths, the fraction whose
        # interval ever excludes the true p stays at or below alpha (+ slack)
        import random
        p = 0.07
        rng = random.Random(2026)
        reps = 300
        violations = 0
        for _ in range(reps):
            k = 0
            for n in range(1, 301):
                k += 1 if rng.random() < p else 0
                lo, hi = stats.cs_bounds(n, k, self.alpha)
                if not _contains((lo, hi), p):
                    violations += 1
                    break
        self.assertLessEqual(violations / reps, self.alpha + 0.04)

    def test_decision_rules(self):
        # long run of conforming items: interval top falls under p0
        n = 1
        while True:
            lo, hi = stats.cs_bounds(n, 0, self.alpha)
            if hi <= 0.03:
                break
            n += 1
        self.assertEqual(stats.snapshot(self.plan, n, 0)["decision"],
                         stats.ACCEPT)
        # sustained 20% defectives: lower bound rises above p1
        n = 1
        while True:
            lo, hi = stats.cs_bounds(n, round(n * 0.2), self.alpha)
            if lo >= 0.10:
                break
            n += 1
        self.assertEqual(stats.snapshot(
            self.plan, n, round(n * 0.2))["decision"], stats.REJECT)


def _contains(bounds, p, tol=1e-12):
    return bounds[0] - tol <= p <= bounds[1] + tol


class TestProjection(unittest.TestCase):
    def test_steps_are_real_decisions(self):
        plan = stats.make_plan(
            method="sprt_binomial", sampling="with_replacement",
            p0=0.03, p1=0.10, alpha=0.05, beta=0.10)
        proj = stats.projection(plan, 10, 1)
        s_acc = stats.snapshot(plan, 10 + proj["min_additional_all_conforming"],
                               1)
        self.assertEqual(s_acc["decision"], stats.ACCEPT)
        s_rej = stats.snapshot(plan, 10 + proj["min_additional_all_defective"],
                               1 + proj["min_additional_all_defective"])
        self.assertEqual(s_rej["decision"], stats.REJECT)

    def test_hypergeometric_capacity(self):
        plan = stats.make_plan(
            method="sprt_hypergeometric", sampling="without_replacement",
            p0=0.03, p1=0.10, alpha=0.05, beta=0.10, N=100)
        proj = stats.projection(plan, 0, 0)
        self.assertEqual(proj["remaining_capacity"], 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
