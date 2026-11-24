"""Tests for JSON persistence and the append-only audit stream."""

import json
import tempfile
import unittest
from pathlib import Path

from app import state, stats, storage


def plan():
    return stats.make_plan(
        method="sprt_binomial", sampling="with_replacement",
        p0=0.03, p1=0.10, alpha=0.05, beta=0.10, name="standard")


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = storage.Store(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_preserves_event_log(self):
        batch = state.create_batch(plan(), batch_id="B-1")
        batch, _ = state.record_items(batch, [
            {"item_id": "a", "defective": False},
            {"item_id": "b", "defective": True}])
        self.store.save_batch(batch, "item_recorded")
        loaded = self.store.load_batch("B-1")
        self.assertEqual(loaded["events"], batch["events"])
        self.assertEqual(
            state.rebuild(loaded)["current"],
            state.rebuild(batch)["current"])

    def test_list_batches_summaries(self):
        batch = state.create_batch(plan(), batch_id="B-1")
        batch, _ = state.record_items(
            batch, [{"item_id": "a", "defective": True}])
        self.store.save_batch(batch, "item_recorded")
        summaries = self.store.list_batches()
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["batch_id"], "B-1")
        self.assertEqual(summaries[0]["n"], 1)

    def test_corrupt_batch_refused(self):
        batch = state.create_batch(plan(), batch_id="B-1")
        batch, _ = state.record_items(
            batch, [{"item_id": "a", "defective": False}])
        self.store.save_batch(batch, "item_recorded")
        path = self.store._batch_path("B-1")
        payload = json.loads(path.read_text())
        payload["events"][-1]["llr_increment"] = 1.0
        path.write_text(json.dumps(payload))
        with self.assertRaises(state.StateError):
            self.store.list_batches()

    def test_audit_stream_is_append_only(self):
        batch = state.create_batch(plan(), batch_id="B-1")
        self.store.save_batch(batch, "batch_created")
        batch, _ = state.record_items(
            batch, [{"item_id": "a", "defective": False}])
        self.store.save_batch(batch, "item_recorded")
        batch = state.withdraw_item(batch, "a", "mis-filed")
        self.store.save_batch(batch, "item_withdrawn")
        records = self.store.read_audit()
        actions = [r["action"] for r in records]
        self.assertEqual(
            actions, ["batch_created", "item_recorded", "item_withdrawn"])
        for r in records:
            self.assertIn("at", r)
            self.assertEqual(r["batch_id"], "B-1")
        # every audit line is independently valid JSON
        for line in Path(self.store.audit_path).read_text().strip().splitlines():
            self.assertIsInstance(json.loads(line), dict)

    def test_plan_presets_load(self):
        for preset in storage.RISK_PRESETS:
            self.store.save_plan(preset)
        names = {p["name"] for p in self.store.list_plans()}
        self.assertEqual(names, {p["name"] for p in storage.RISK_PRESETS})

    def test_atomic_write_leaves_no_tmp_files(self):
        batch = state.create_batch(plan(), batch_id="B-1")
        self.store.save_batch(batch, "batch_created")
        leftovers = list(Path(self.tmp.name).rglob("*.tmp"))
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
