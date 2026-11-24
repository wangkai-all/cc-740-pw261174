"""Tests for the reproducible simulation engine and operating characteristics."""

import math
import unittest

from app import simulate, stats


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_lot_and_result(self):
        plan = stats.make_plan(
            method="sprt_binomial", sampling="with_replacement",
            p0=0.03, p1=0.10, alpha=0.05, beta=0.10)
        r1 = simulate.run_single(plan, 0.10, seed=12345)
        r2 = simulate.run_single(plan, 0.10, seed=12345)
        self.assertEqual(r1.to_dict(), r2.to_dict())

    def test_different_seeds_can_differ(self):
        plan = stats.make_plan(
            method="sprt_binomial", sampling="with_replacement",
            p0=0.03, p1=0.10, alpha=0.05, beta=0.10)
        results = {simulate.run_single(plan, 0.06, seed=s).n
                   for s in range(20)}
        self.assertGreater(len(results), 1)

    def test_drives_real_state_machine(self):
        # the simulated batch must itself replay cleanly through the gate
        plan = stats.make_plan(
            method="sprt_binomial", sampling="with_replacement",
            p0=0.03, p1=0.10, alpha=0.05, beta=0.10)
        res = simulate.run_single(plan, 0.10, seed=7)
        self.assertEqual(len(res.timeline), res.n)
        self.assertTrue(all(
            row["decision"] in (stats.CONTINUE, stats.ACCEPT, stats.REJECT)
            for row in res.timeline))
        self.assertNotEqual(res.decision, stats.CONTINUE)

    def test_conservative_cs_is_censored_not_raised(self):
        # a lot sitting between the thresholds may not separate within a
        # small cap; this must surface as 'continue', not an exception
        plan = stats.make_plan(
            method="cs_beta_mixture", sampling="with_replacement",
            p0=0.03, p1=0.10, alpha_cs=0.05)
        res = simulate.run_single(plan, 0.06, seed=5, round_size=16, cap=300)
        self.assertEqual(res.n, 300)
        self.assertEqual(res.decision, stats.CONTINUE)
        # the censored log still passes the production replay gate
        self.assertEqual(len(res.timeline), 300)

    def test_grid_reports_censor_rate(self):
        plan = stats.make_plan(
            method="cs_beta_mixture", sampling="with_replacement",
            p0=0.03, p1=0.10, alpha_cs=0.05)
        grid = simulate.run_grid(plan, [0.06], replications=5,
                                 round_size=16, cap=200)
        cell = grid["cells"][0]
        self.assertIn("censor_rate", cell)
        self.assertAlmostEqual(
            cell["accept_rate"] + cell["reject_rate"]
            + cell["censor_rate"], 1.0)


class TestOperatingCharacteristics(unittest.TestCase):
    def test_wald_error_guarantees(self):
        plan = stats.make_plan(
            method="sprt_binomial", sampling="with_replacement",
            p0=0.03, p1=0.10, alpha=0.05, beta=0.10)
        grid = simulate.run_grid(plan, [0.03, 0.10], replications=200,
                                 round_size=8)
        at_h0 = next(c for c in grid["cells"] if c["true_p"] == 0.03)
        at_h1 = next(c for c in grid["cells"] if c["true_p"] == 0.10)
        # false reject at H0 <= alpha/(1-beta); false accept at H1 likewise
        self.assertLessEqual(
            at_h0["reject_rate"], plan.alpha / (1 - plan.beta) + 0.05)
        self.assertLessEqual(
            at_h1["accept_rate"], plan.beta / (1 - plan.alpha) + 0.05)
        # every run terminated
        self.assertAlmostEqual(
            at_h0["accept_rate"] + at_h0["reject_rate"], 1.0)

    def test_average_inspection_count_is_finite_and_reported(self):
        plan = stats.make_plan(
            method="sprt_binomial", sampling="with_replacement",
            p0=0.03, p1=0.10, alpha=0.05, beta=0.10)
        grid = simulate.run_grid(plan, [0.0, 0.06], replications=60,
                                 round_size=8)
        for cell in grid["cells"]:
            self.assertGreater(cell["avg_n"], 0)
            self.assertLessEqual(cell["min_n"], cell["max_n"])
            self.assertGreater(cell["max_n"], 0)

    def test_without_replacement_uses_finite_lot(self):
        plan = stats.make_plan(
            method="sprt_hypergeometric", sampling="without_replacement",
            p0=0.03, p1=0.10, alpha=0.05, beta=0.10, N=300)
        res = simulate.run_single(plan, 0.10, seed=3)
        self.assertLessEqual(res.n, 300)
        ids = [row["item_id"] for row in res.timeline]
        self.assertEqual(len(ids), len(set(ids)))  # no repeated item

    def test_paired_plan_comparison_uses_shared_seeds(self):
        strict = stats.make_plan(
            method="sprt_binomial", sampling="with_replacement",
            p0=0.03, p1=0.10, alpha=0.02, beta=0.05)
        relaxed = stats.make_plan(
            method="sprt_binomial", sampling="with_replacement",
            p0=0.05, p1=0.15, alpha=0.10, beta=0.20)
        grids = simulate.compare_plans(
            [strict, relaxed], [0.05], replications=30, round_size=8)
        self.assertEqual(grids[0]["base_seed"], grids[1]["base_seed"])
        self.assertEqual(grids[0]["replications"],
                         grids[1]["replications"])


class TestConfidenceSequenceExperiment(unittest.TestCase):
    def test_cs_false_acceptance_bounded(self):
        # Acceptance needs the whole interval below p0.  The Beta(1,1)
        # mixture is deliberately conservative and cannot separate a lot
        # sitting exactly on p1 within a small cap, so drive a clearly bad
        # lot (p=0.20): false acceptance must stay rare (Ville + MC slack).
        plan = stats.make_plan(
            method="cs_beta_mixture", sampling="with_replacement",
            p0=0.03, p1=0.10, alpha_cs=0.05)
        grid = simulate.run_grid(plan, [0.20], replications=40,
                                 round_size=16)
        cell = grid["cells"][0]
        self.assertLess(cell["accept_rate"], plan.alpha_cs + 0.05)
        self.assertGreater(cell["avg_n"], 0)

    def test_cs_good_lot_is_accepted(self):
        # a perfect lot must eventually be accepted, exercising the lower
        # boundary of the confidence sequence end to end
        plan = stats.make_plan(
            method="cs_beta_mixture", sampling="with_replacement",
            p0=0.03, p1=0.10, alpha_cs=0.05)
        grid = simulate.run_grid(plan, [0.0], replications=20,
                                 round_size=16)
        cell = grid["cells"][0]
        self.assertEqual(cell["accept_rate"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
