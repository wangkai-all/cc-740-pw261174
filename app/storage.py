"""JSON file persistence with atomic writes and a global audit stream."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from . import state

# A few risk configurations fixed up front.  They are *presets*: every batch
# snapshots the full parameters it actually used, so later edits here can
# never change a historical decision.
RISK_PRESETS = [
    {
        "name": "strict",
        "label": "严格 α=2% / β=5%",
        "method": "sprt_binomial",
        "sampling": "with_replacement",
        "p0": 0.02, "p1": 0.08, "alpha": 0.02, "beta": 0.05,
    },
    {
        "name": "standard",
        "label": "标准 α=5% / β=10%",
        "method": "sprt_binomial",
        "sampling": "with_replacement",
        "p0": 0.03, "p1": 0.10, "alpha": 0.05, "beta": 0.10,
    },
    {
        "name": "relaxed",
        "label": "宽松 α=10% / β=20%",
        "method": "sprt_binomial",
        "sampling": "with_replacement",
        "p0": 0.05, "p1": 0.15, "alpha": 0.10, "beta": 0.20,
    },
    {
        "name": "finite-lot-standard",
        "label": "无放回有限批 α=5% / β=10% (N=500)",
        "method": "sprt_hypergeometric",
        "sampling": "without_replacement",
        "p0": 0.03, "p1": 0.10, "alpha": 0.05, "beta": 0.10, "N": 500,
    },
    {
        "name": "confidence-sequence",
        "label": "置信序列 α_cs=5%",
        "method": "cs_beta_mixture",
        "sampling": "with_replacement",
        "p0": 0.03, "p1": 0.10, "alpha_cs": 0.05,
    },
]


class Store:
    def __init__(self, data_dir: str | os.PathLike):
        self.root = Path(data_dir)
        self.batches_dir = self.root / "batches"
        self.plans_dir = self.root / "plans"
        self.audit_path = self.root / "audit.jsonl"
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path.touch(exist_ok=True)

    # -- generic JSON helpers ----------------------------------------------

    @staticmethod
    def _atomic_write(path: Path, payload: dict) -> None:
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2,
                          allow_nan=False)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise

    @staticmethod
    def _read_json(path: Path) -> dict:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)

    def audit(self, batch_id: str, action: str, detail: dict) -> None:
        record = {"at": state.now_iso(), "batch_id": batch_id,
                  "action": action, "detail": detail}
        with self.audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False,
                                allow_nan=False) + "\n")

    def read_audit(self) -> list[dict]:
        with self.audit_path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    # -- plans --------------------------------------------------------------

    def save_plan(self, plan_payload: dict) -> dict:
        name = plan_payload.get("name") or "plan"
        safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in name)
        path = self.plans_dir / f"{safe}.json"
        self._atomic_write(path, plan_payload)
        self.audit("PLAN", "plan_saved", {"plan": plan_payload})
        return plan_payload

    def list_plans(self) -> list[dict]:
        return [self._read_json(p) for p in
                sorted(self.plans_dir.glob("*.json"))]

    def get_plan(self, name: str) -> dict | None:
        path = self.plans_dir / f"{name}.json"
        return self._read_json(path) if path.exists() else None

    # -- batches ------------------------------------------------------------

    def _batch_path(self, batch_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "._-" else "-"
                       for c in batch_id)
        return self.batches_dir / f"{safe}.json"

    def save_batch(self, batch: dict, action: str, detail: dict | None = None):
        # Always rebuild before persisting: storage refuses an inconsistent log.
        state.rebuild(batch)
        self._atomic_write(self._batch_path(batch["batch_id"]), batch)
        self.audit(batch["batch_id"], action, detail or {})

    def load_batch(self, batch_id: str) -> dict:
        path = self._batch_path(batch_id)
        if not path.exists():
            raise KeyError(batch_id)
        return self._read_json(path)

    def list_batches(self) -> list[dict]:
        summaries = []
        for path in sorted(self.batches_dir.glob("*.json")):
            batch = self._read_json(path)
            derived = state.rebuild(batch)
            summaries.append({
                "batch_id": batch["batch_id"],
                "created_at": batch["created_at"],
                "lot_note": derived["lot_note"],
                "status": derived["status"],
                "released": derived["released"],
                "released_at": derived["released_at"],
                "n": derived["current"]["n"],
                "k": derived["current"]["k"],
                "round": derived["round"],
                "plan_name": batch["plan"].get("name", ""),
                "method": batch["plan"]["method"],
            })
        return summaries
