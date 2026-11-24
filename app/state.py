"""Batch state machine, driven by an append-only event log.

The event list is the source of truth.  Every state change appends an event
and rebuilds the whole batch by replaying the log, so the stored state can
never drift from what the statistics engine would compute from scratch.

States
------
* statistical decision: ``continue`` / ``accepted`` / ``rejected``
* a decided-but-not-released batch may have its decision superseded by a
  withdrawal of a mis-filed item;
* a released batch is immutable statistically: later problems can only be
  flagged as corrections, the historical release timestamp is preserved.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from . import stats

ACCEPT = stats.ACCEPT
REJECT = stats.REJECT
CONTINUE = stats.CONTINUE


class StateError(ValueError):
    """Illegal transition or inconsistent event log."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]")


def make_batch_id(name: str = "") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    slug = _SLUG_RE.sub("-", name).strip("-")[:24] if name else "lot"
    return f"B-{stamp}-{slug}"


# ---------------------------------------------------------------------------
# Event log construction
# ---------------------------------------------------------------------------

def _append_and_rebuild(batch: dict, event: dict) -> dict:
    return {**batch, "events": [*batch["events"], event]}


def create_batch(plan: stats.PlanParams, lot_note: str = "",
                 batch_id: Optional[str] = None, at: Optional[str] = None) -> dict:
    at = at or now_iso()
    batch = {
        "batch_id": batch_id or make_batch_id(plan.name),
        "created_at": at,
        "plan": plan.to_dict(),
        "assumptions": stats.assumptions_text(plan),
        "lot_note": lot_note,
        "events": [{"type": "batch_created", "at": at}],
    }
    fresh = _append_and_rebuild(batch,
                                {"type": "round_opened", "at": at, "round": 1})
    rebuild(fresh)  # consistency gate
    return fresh


def open_round(batch: dict, at: Optional[str] = None) -> dict:
    """Begin the next sampling round; legal only while still undecided."""
    state_view = rebuild(batch)
    if state_view["status"] != CONTINUE:
        raise StateError("batch already decided; no further sampling rounds")
    at = at or now_iso()
    next_round = state_view["round"] + 1
    events = list(batch["events"])
    events.append({"type": "round_closed", "at": at, "round": state_view["round"]})
    events.append({"type": "round_opened", "at": at, "round": next_round})
    fresh = {**batch, "events": events}
    rebuild(fresh)  # consistency check
    return fresh


def record_items(batch: dict, entries: list[dict],
                 at: Optional[str] = None) -> tuple[dict, dict]:
    """Record one round's items.

    Each entry: {"item_id": str, "defective": bool}.  Repeated item ids
    inside the same call or against any *active* item are blocked and logged
    as ``duplicate_blocked`` without contributing information.  If a decision
    boundary is crossed mid-batch, remaining entries are returned as
    ``not_recorded`` rather than silently consumed.
    """
    state = rebuild(batch)
    if state["status"] != CONTINUE:
        raise StateError(f"batch already {state['status']}; cannot record more items")
    at = at or now_iso()
    plan = stats.make_plan(**batch["plan"])
    events = list(batch["events"])
    active_ids = {i["item_id"] for i in state["items"] if i["active"]}
    round_no = state["round"]

    recorded, blocked, not_recorded = [], [], []
    n, k = state["current"]["n"], state["current"]["k"]
    seq = state["next_seq"]
    decided = False

    for entry in entries:
        item_id = str(entry.get("item_id", "")).strip()
        if not item_id:
            raise StateError("every entry needs a non-empty item_id")
        defective = bool(entry["defective"])

        if decided:
            not_recorded.append({"item_id": item_id, "defective": defective})
            continue
        if item_id in active_ids:
            events.append({"type": "duplicate_blocked", "at": at,
                           "item_id": item_id, "round": round_no})
            blocked.append(item_id)
            continue

        # CS methods carry no scalar per-step LLR (the interval is rebuilt
        # from (n, k)); store null and let replay validate via snapshot.
        if plan.method == "cs_beta_mixture":
            inc = None
        else:
            inc = stats.llr_increment(plan, n, k, defective)
        events.append({
            "type": "item_recorded", "at": at, "seq": seq,
            "item_id": item_id, "defective": defective,
            "round": round_no, "llr_increment": inc,
        })
        active_ids.add(item_id)
        recorded.append({"item_id": item_id, "defective": defective})
        n += 1
        k += 1 if defective else 0
        seq += 1

        snap = stats.snapshot(plan, n, k)
        if snap["decision"] != CONTINUE:
            events.append({
                "type": "decision", "at": at, "decision": snap["decision"],
                "n": n, "k": k, "round": round_no,
                "snapshot": _jsonable(snap),
            })
            events.append({"type": "round_closed", "at": at, "round": round_no})
            decided = True

    fresh = {**batch, "events": events}
    derived = rebuild(fresh)
    derived["last_intake"] = {
        "recorded": recorded, "blocked_duplicates": blocked,
        "not_recorded_after_decision": not_recorded,
    }
    return fresh, derived["last_intake"]


def withdraw_item(batch: dict, item_id: str, reason: str,
                  at: Optional[str] = None) -> dict:
    """Remove a mis-filed item's contribution, or flag a correction.

    * undecided batch            -> contribution removed, state restored;
    * decided but not released   -> old decision superseded, state restored
                                    to whatever the remaining evidence gives;
    * released batch             -> immutable: a ``correction_flagged`` event
                                    is appended instead, nothing is recomputed.
    """
    at = at or now_iso()
    state_view = rebuild(batch)
    active = next((i for i in state_view["items"]
                   if i["item_id"] == item_id and i["active"]), None)
    if active is None:
        raise StateError(f"no active item {item_id!r} in this batch")
    if not reason or not str(reason).strip():
        raise StateError("a withdrawal reason is required for the audit trail")

    prior_status = state_view["status"]
    if state_view["released"]:
        event = {
            "type": "correction_flagged", "at": at, "item_id": item_id,
            "reason": reason.strip(), "released_at": state_view["released_at"],
            "status_at_flag_time": prior_status,
        }
        fresh = {**batch, "events": [*batch["events"], event]}
        rebuild(fresh)
        return fresh

    mode = "pre_decision" if prior_status == CONTINUE else "supersedes_decision"
    events = [*batch["events"],
              {"type": "item_withdrawn", "at": at, "item_id": item_id,
               "reason": reason.strip(), "mode": mode,
               "prior_status": prior_status, "seq": active["seq"]}]
    fresh = {**batch, "events": events}
    rebuild(fresh)
    return fresh


def flag_correction(batch: dict, note: str, at: Optional[str] = None) -> dict:
    """Record a post-release correction note without touching statistics."""
    at = at or now_iso()
    state_view = rebuild(batch)
    if not state_view["released"]:
        raise StateError("corrections are only for already released batches; "
                         "use withdrawal before release instead")
    if not note or not note.strip():
        raise StateError("a correction note is required")
    event = {"type": "correction_flagged", "at": at,
             "item_id": None, "reason": note.strip(),
             "released_at": state_view["released_at"],
             "status_at_flag_time": state_view["status"]}
    fresh = {**batch, "events": [*batch["events"], event]}
    rebuild(fresh)
    return fresh


def release_batch(batch: dict, at: Optional[str] = None) -> dict:
    """Freeze an accepted/rejected batch and stamp the release time."""
    at = at or now_iso()
    state_view = rebuild(batch)
    if state_view["status"] == CONTINUE:
        raise StateError("cannot release a batch that is still undecided")
    if state_view["released"]:
        raise StateError("batch already released; release time is immutable")
    fresh = {**batch, "events": [
        *batch["events"], {"type": "batch_released", "at": at}]}
    rebuild(fresh)
    return fresh


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def _jsonable(obj):
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None if math.isnan(obj) else ("Infinity" if obj > 0 else "-Infinity")
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _close_enough(a, b, tol=1e-9) -> bool:
    # JSON has no infinities; _jsonable encodes them as sentinel strings.
    sentinels = {"Infinity": math.inf, "-Infinity": -math.inf, "NaN": math.nan}
    if isinstance(a, str) and a in sentinels:
        a = sentinels[a]
    if isinstance(b, str) and b in sentinels:
        b = sentinels[b]
    if a is None or b is None:
        return a == b
    if isinstance(a, float) and math.isinf(a):
        return a == b
    if isinstance(b, float) and math.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def rebuild(batch: dict) -> dict:
    """Replay the event log into the full derived state and audit it.

    Raises StateError if an event contradicts the statistics engine (wrong
    per-step increment, wrong decision snapshot) or the log is structurally
    invalid.  This is the single consistency gate used by storage, tests and
    the verification endpoint.
    """
    plan = stats.make_plan(**batch["plan"])
    events = batch["events"]
    if not events or events[0]["type"] != "batch_created":
        raise StateError("event log must start with batch_created")

    items: list[dict] = []
    withdrawn: list[dict] = []
    corrections: list[dict] = []
    duplicates: list[dict] = []
    rounds: list[dict] = []
    released = False
    released_at = None
    current_round = 0
    n = k = 0
    seq_counter = 0
    history: list[dict] = []
    status = CONTINUE
    decided_event: Optional[dict] = None
    active_ids: set[str] = set()

    for ev in events[1:]:
        t = ev["type"]

        if t == "round_opened":
            if status != CONTINUE:
                raise StateError("round opened after decision")
            if rounds and rounds[-1].get("closed_at") is None:
                raise StateError("previous round was never closed")
            current_round = ev["round"]
            rounds.append({"round": ev["round"], "opened_at": ev["at"],
                           "closed_at": None})

        elif t == "round_closed":
            if not rounds or rounds[-1]["round"] != ev["round"]:
                raise StateError("round_closed without matching open round")
            rounds[-1]["closed_at"] = ev["at"]

        elif t == "duplicate_blocked":
            if ev["item_id"] not in active_ids:
                raise StateError(
                    f"duplicate_blocked for non-active item {ev['item_id']!r}")
            duplicates.append({"item_id": ev["item_id"], "at": ev["at"],
                               "round": ev["round"]})

        elif t == "item_recorded":
            if status != CONTINUE:
                raise StateError("item recorded after a decision")
            if ev["item_id"] in active_ids:
                raise StateError(f"item {ev['item_id']!r} contributes twice; "
                                 "duplicates must be blocked")
            if plan.N is not None and n >= plan.N:
                raise StateError("lot capacity exhausted")
            if plan.method == "cs_beta_mixture":
                if ev["llr_increment"] is not None:
                    raise StateError(
                        "CS events must carry a null llr_increment")
                want_inc = None
            else:
                want_inc = stats.llr_increment(
                    plan, n, k, ev["defective"])
                if not _close_enough(want_inc, ev["llr_increment"]):
                    raise StateError(
                        f"audit mismatch at seq={ev['seq']}: stored increment "
                        f"{ev['llr_increment']} != recomputed {want_inc}")
            seq_counter += 1
            if ev["seq"] != seq_counter:
                raise StateError("non-contiguous item sequence number")
            n += 1
            k += 1 if ev["defective"] else 0
            items.append({"seq": seq_counter, "item_id": ev["item_id"],
                          "defective": ev["defective"], "round": ev["round"],
                          "recorded_at": ev["at"], "active": True,
                          "llr_increment": want_inc})
            active_ids.add(ev["item_id"])
            history.append({**stats.snapshot(plan, n, k),
                            "seq": seq_counter,
                            "item_id": ev["item_id"]})

        elif t == "decision":
            snap = stats.snapshot(plan, n, k)
            if snap["decision"] == CONTINUE:
                raise StateError("decision event emitted while still continue")
            if ev["decision"] != snap["decision"] or ev["n"] != n or ev["k"] != k:
                raise StateError("decision event disagrees with recomputation")
            for key in ("lower_boundary", "upper_boundary", "cs_lower",
                        "cs_upper", "statistic"):
                if key in ev["snapshot"] and not _close_enough(
                        ev["snapshot"][key], snap.get(key)):
                    raise StateError(
                        f"decision snapshot field {key} disagrees on replay")
            status = snap["decision"]
            decided_event = ev

        elif t == "item_withdrawn":
            if released:
                raise StateError("withdrawal forbidden after release; "
                                 "use correction_flagged")
            target = next((i for i in items
                           if i["item_id"] == ev["item_id"] and i["active"]), None)
            if target is None:
                raise StateError(f"withdrawal of unknown/inactive item "
                                 f"{ev['item_id']!r}")
            target["active"] = False
            active_ids.discard(target["item_id"])
            withdrawn.append({"item_id": target["item_id"], "at": ev["at"],
                              "reason": ev["reason"], "mode": ev["mode"],
                              "prior_status": ev["prior_status"],
                              "seq": target["seq"]})
            # Recompute everything from the surviving items.
            n = sum(1 for i in items if i["active"])
            k = sum(1 for i in items if i["active"] and i["defective"])
            history = _replay_history(plan, items)
            status = stats.snapshot(plan, n, k)["decision"]
            decided_event = None

        elif t == "batch_released":
            if status == CONTINUE:
                raise StateError("release while still continue")
            if released:
                raise StateError("duplicate release event")
            released = True
            released_at = ev["at"]

        elif t == "correction_flagged":
            corrections.append({"at": ev["at"], "item_id": ev.get("item_id"),
                                "reason": ev["reason"],
                                "released_at": ev["released_at"]})
        else:
            raise StateError(f"unknown event type {t!r}")

    current = stats.snapshot(plan, n, k)
    if current["decision"] != status:
        raise StateError("replayed status disagrees with current snapshot")

    return {
        "batch_id": batch["batch_id"],
        "created_at": batch["created_at"],
        "plan": plan.to_dict(),
        "assumptions": batch["assumptions"],
        "lot_note": batch.get("lot_note", ""),
        "status": status,
        "released": released,
        "released_at": released_at,
        "round": current_round,
        "rounds": rounds,
        "items": items,
        "withdrawn": withdrawn,
        "corrections": corrections,
        "duplicates": duplicates,
        "current": current,
        "history": history,
        "next_seq": seq_counter + 1,
        "projection": stats.projection(plan, n, k),
        "decision_event": decided_event,
        "event_count": len(events),
    }


def _replay_history(plan: stats.PlanParams, items: list[dict]) -> list[dict]:
    """Recompute the per-item snapshot history from active items, in order."""
    n = k = 0
    out = []
    for seq, item in enumerate((i for i in items if i["active"]), start=1):
        n += 1
        k += 1 if item["defective"] else 0
        out.append({**stats.snapshot(plan, n, k), "seq": seq,
                    "item_id": item["item_id"]})
    return out
