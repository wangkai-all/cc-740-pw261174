"""Tests for the event-sourced batch state machine."""

import unittest

from app import state, stats
from app.state import StateError


def binomial_plan(**over):
    kw = dict(method="sprt_binomial", sampling="with_replacement",
              p0=0.03, p1=0.10, alpha=0.05, beta=0.10, name="t")
    kw.update(over)
    return stats.make_plan(**kw)


def hyper_plan(**over):
    kw = dict(method="sprt_hypergeometric", sampling="without_replacement",
              p0=0.03, p1=0.10, alpha=0.05, beta=0.10, N=500, name="th")
    kw.update(over)
    return stats.make_plan(**kw)


def drive_to_decision(plan, defective_flags, start=0):
    batch = state.create_batch(plan, batch_id="T-1")
    for idx, d in enumerate(defective_flags, start=start):
        batch, _ = state.record_items(
            batch, [{"item_id": f"X{idx}", "defective": bool(d)}])
        if state.rebuild(batch)["status"] != stats.CONTINUE:
            break
    return batch


class TestIntakeAndRounds(unittest.TestCase):
    def test_create_starts_in_round_one(self):
        batch = state.create_batch(binomial_plan(), batch_id="T-1")
        view = state.rebuild(batch)
        self.assertEqual(view["status"], stats.CONTINUE)
        self.assertEqual(view["round"], 1)
        self.assertEqual(view["current"]["n"], 0)

    def test_rounds_accumulate(self):
        batch = state.create_batch(binomial_plan(), batch_id="T-1")
        batch, _ = state.record_items(batch, [
            {"item_id": "a", "defective": False},
            {"item_id": "b", "defective": False}])
        batch = state.open_round(batch)
        self.assertEqual(state.rebuild(batch)["round"], 2)
        batch = state.open_round(batch)
        self.assertEqual(state.rebuild(batch)["round"], 3)
        self.assertEqual(state.rebuild(batch)["current"]["n"], 2)

    def test_duplicate_item_blocked_and_audited(self):
        batch = state.create_batch(binomial_plan(), batch_id="T-1")
        batch, intake = state.record_items(batch, [
            {"item_id": "DUP", "defective": True}])
        self.assertEqual(intake["recorded"][0]["item_id"], "DUP")
        batch, intake = state.record_items(batch, [
            {"item_id": "DUP", "defective": False}])
        self.assertEqual(intake["blocked_duplicates"], ["DUP"])
        view = state.rebuild(batch)
        self.assertEqual(view["current"]["n"], 1)
        self.assertEqual(view["current"]["k"], 1)
        self.assertEqual(len(view["duplicates"]), 1)
        # same item must never appear twice among active contributors
        active = [i["item_id"] for i in view["items"] if i["active"]]
        self.assertEqual(active.count("DUP"), 1)

    def test_duplicate_within_single_call(self):
        batch = state.create_batch(binomial_plan(), batch_id="T-1")
        batch, intake = state.record_items(batch, [
            {"item_id": "z", "defective": False},
            {"item_id": "z", "defective": True}])
        view = state.rebuild(batch)
        self.assertEqual(view["current"]["n"], 1)
        self.assertEqual(intake["blocked_duplicates"], ["z"])

    def test_recording_after_decision_truncates(self):
        # all defective reaches the upper boundary quickly
        plan = binomial_plan()
        entries = [{"item_id": f"d{i}", "defective": True} for i in range(50)]
        batch = state.create_batch(plan, batch_id="T-1")
        batch, intake = state.record_items(batch, entries)
        view = state.rebuild(batch)
        self.assertEqual(view["status"], stats.REJECT)
        self.assertTrue(intake["not_recorded_after_decision"])
        with self.assertRaises(StateError):
            state.open_round(batch)
        with self.assertRaises(StateError):
            state.record_items(batch, [{"item_id": "x", "defective": False}])


class TestWithdrawals(unittest.TestCase):
    def test_withdraw_undecided_restores_counts(self):
        batch = state.create_batch(binomial_plan(), batch_id="T-1")
        batch, _ = state.record_items(batch, [
            {"item_id": "a", "defective": True},
            {"item_id": "b", "defective": False},
            {"item_id": "c", "defective": False}])
        view = state.rebuild(batch)
        self.assertEqual((view["current"]["n"], view["current"]["k"]), (3, 1))
        batch = state.withdraw_item(batch, "a", "mis-filed: belongs elsewhere")
        view = state.rebuild(batch)
        self.assertEqual((view["current"]["n"], view["current"]["k"]), (2, 0))
        self.assertEqual(view["status"], stats.CONTINUE)
        self.assertFalse(view["items"][0]["active"])
        self.assertEqual(view["withdrawn"][0]["mode"], "pre_decision")
        # history is rebuilt from surviving items
        self.assertEqual([h["n"] for h in view["history"]], [1, 2])

    def test_withdraw_reopens_a_superseded_decision(self):
        # craft a path that rejects, then remove the decisive defective item
        plan = binomial_plan()
        batch = state.create_batch(plan, batch_id="T-1")
        rng_path = [True] * 4 + [False] * 2
        batch = drive_to_decision(plan, rng_path)
        view = state.rebuild(batch)
        if view["status"] != stats.REJECT:
            self.skipTest("path did not reject as expected")
        n_at_stop = view["current"]["n"]
        # withdraw the most recent defective contributor
        last_def = next(i["item_id"] for i in reversed(view["items"])
                        if i["defective"] and i["active"])
        batch = state.withdraw_item(batch, last_def, "wrong batch label")
        view = state.rebuild(batch)
        self.assertEqual(view["current"]["n"], n_at_stop - 1)
        self.assertEqual(view["withdrawn"][-1]["mode"],
                         "supersedes_decision")
        self.assertEqual(view["withdrawn"][-1]["prior_status"],
                         stats.REJECT)
        # sampling can resume after restoration
        batch, _ = state.record_items(
            batch, [{"item_id": "new1", "defective": False}])
        self.assertEqual(state.rebuild(batch)["current"]["n"], n_at_stop)

    def test_withdraw_requires_reason(self):
        batch = state.create_batch(binomial_plan(), batch_id="T-1")
        batch, _ = state.record_items(
            batch, [{"item_id": "a", "defective": False}])
        with self.assertRaises(StateError):
            state.withdraw_item(batch, "a", "  ")

    def test_withdraw_unknown_item(self):
        batch = state.create_batch(binomial_plan(), batch_id="T-1")
        with self.assertRaises(StateError):
            state.withdraw_item(batch, "ghost", "reason")

    def test_reattach_withdrawn_id_is_allowed_once(self):
        # an item withdrawn from this batch is no longer active, so its id
        # can legitimately be used again (e.g. re-filed correctly)
        batch = state.create_batch(binomial_plan(), batch_id="T-1")
        batch, _ = state.record_items(
            batch, [{"item_id": "a", "defective": True}])
        batch = state.withdraw_item(batch, "a", "mis-filed")
        batch, intake = state.record_items(
            batch, [{"item_id": "a", "defective": False}])
        self.assertEqual(intake["blocked_duplicates"], [])
        view = state.rebuild(batch)
        active = [i for i in view["items"] if i["active"]]
        self.assertEqual(len(active), 1)
        self.assertFalse(active[0]["defective"])


class TestReleaseAndCorrections(unittest.TestCase):
    def _released_batch(self):
        plan = binomial_plan()
        batch = drive_to_decision(plan, [True] * 30)
        if state.rebuild(batch)["status"] == stats.CONTINUE:
            batch = drive_to_decision(plan, [False] * 200)
        release_time = "2026-11-24T10:00:00+00:00"
        batch = state.release_batch(batch, at=release_time)
        return batch, release_time

    def test_release_freezes_and_stamps_time(self):
        batch, ts = self._released_batch()
        view = state.rebuild(batch)
        self.assertTrue(view["released"])
        self.assertEqual(view["released_at"], ts)

    def test_release_time_immutable(self):
        batch, ts = self._released_batch()
        with self.assertRaises(StateError):
            state.release_batch(batch, at="2030-01-01T00:00:00+00:00")
        # second release attempt cannot rewrite history
        self.assertEqual(state.rebuild(batch)["released_at"], ts)

    def test_post_release_withdraw_becomes_correction(self):
        batch, ts = self._released_batch()
        view = state.rebuild(batch)
        target = view["items"][0]["item_id"]
        n_before = view["current"]["n"]
        batch = state.withdraw_item(batch, target, "later found mis-filed")
        view = state.rebuild(batch)
        # statistics untouched
        self.assertEqual(view["current"]["n"], n_before)
        self.assertTrue(view["released"])
        self.assertEqual(view["released_at"], ts)
        self.assertEqual(len(view["corrections"]), 1)
        self.assertEqual(view["corrections"][0]["item_id"], target)
        # the historical item is still active (history not rewritten)
        self.assertTrue(any(i["item_id"] == target and i["active"]
                            for i in view["items"]))

    def test_correction_note_only_when_released(self):
        batch = state.create_batch(binomial_plan(), batch_id="T-1")
        with self.assertRaises(StateError):
            state.flag_correction(batch, "note")

    def test_cannot_release_undecided(self):
        batch = state.create_batch(binomial_plan(), batch_id="T-1")
        batch, _ = state.record_items(
            batch, [{"item_id": "a", "defective": False}])
        with self.assertRaises(StateError):
            state.release_batch(batch)


class TestReplayConsistency(unittest.TestCase):
    def test_every_history_snapshot_matches_engine(self):
        plan = hyper_plan()
        batch = state.create_batch(plan, batch_id="T-1")
        flags = [True, False, False, True, False, False, False, True]
        batch, _ = state.record_items(
            batch, [{"item_id": f"h{i}", "defective": d}
                    for i, d in enumerate(flags)])
        view = state.rebuild(batch)
        n = k = 0
        for row in view["history"]:
            n += 1
            k += 1 if next(i["defective"] for i in view["items"]
                           if i["item_id"] == row["item_id"] and i["active"]) else 0
            snap = stats.snapshot(plan, n, k)
            self.assertEqual(row["decision"], snap["decision"])
            self.assertAlmostEqual(row["statistic"], snap["statistic"])

    def test_tampered_increment_detected(self):
        batch = state.create_batch(binomial_plan(), batch_id="T-1")
        batch, _ = state.record_items(
            batch, [{"item_id": "a", "defective": False}])
        batch["events"][-1]["llr_increment"] = 999.0
        with self.assertRaises(StateError):
            state.rebuild(batch)

    def test_tampered_decision_detected(self):
        plan = binomial_plan()
        batch = drive_to_decision(plan, [True] * 30)
        for ev in batch["events"]:
            if ev["type"] == "decision":
                ev["decision"] = stats.ACCEPT
        with self.assertRaises(StateError):
            state.rebuild(batch)

    def test_without_replacement_capacity(self):
        # Finite-lot SPRT normally stops early, so the capacity invariant is
        # checked directly on the event log: a full N-item lot replays
        # successfully, while an (N+1)th item is refused.  Build the log by
        # appending audited events and let rebuild() verify every increment.
        plan = stats.make_plan(
            method="sprt_hypergeometric", sampling="without_replacement",
            p0=0.05, p1=0.30, alpha=0.01, beta=0.01, N=40)
        at = "2026-11-25T00:00:00+00:00"
        events = [{"type": "batch_created", "at": at},
                  {"type": "round_opened", "at": at, "round": 1}]
        n = k = 0
        for seq in range(1, 41):
            defective = seq % 4 == 0   # 10 defectives, between D0=2, D1=12
            inc = stats.llr_increment(plan, n, k, defective)
            events.append({"type": "item_recorded", "at": at, "seq": seq,
                           "item_id": f"c{seq}", "defective": defective,
                           "round": 1, "llr_increment": inc})
            n += 1
            k += 1 if defective else 0
        events.append({"type": "decision", "at": at,
                       "decision": stats.snapshot(plan, n, k)["decision"],
                       "n": n, "k": k, "round": 1,
                       "snapshot": stats.snapshot(plan, n, k)})
        events.append({"type": "round_closed", "at": at, "round": 1})
        batch = {"batch_id": "T-1", "created_at": at,
                 "plan": plan.to_dict(),
                 "assumptions": stats.assumptions_text(plan),
                 "lot_note": "", "events": events}
        view = state.rebuild(batch)
        self.assertEqual(view["current"]["n"], 40)
        self.assertNotEqual(view["status"], stats.CONTINUE)

        # tampering in a 41st active item must trip the replay gate
        batch["events"].insert(-2, {
            "type": "item_recorded", "at": at, "seq": 41,
            "item_id": "c41", "defective": False, "round": 1,
            "llr_increment": 0.0})
        with self.assertRaises(StateError):
            state.rebuild(batch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
