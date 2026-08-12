"""Timeline planning service — build an edit timeline from ordered material.

MAPO's Selection stage has exactly TWO user-facing editing modes. They differ ONLY in
the UNIT of editing; both emit the same structured plan, which Delivery compiles
verbatim:

    CLIP ASSEMBLY   — the unit is the WHOLE CLIP. Ordered complete clips, each keeping
                      its full measured length. There is NO trimming and NO duration
                      control in this mode (a target duration does not apply).
    MOMENT ASSEMBLY — the unit is a temporal EVENT inside a clip. Ordered moments, each
                      cut to its own measured in/out timecodes, with an OPTIONAL target
                      total duration.

The old TRIM / TIMED / FULL CLIP / EVENT modes are gone: they conflated editing
GRANULARITY (clip vs moment) with DURATION CONTROL. So is clip-level proportional
allocation — screen time is never "split between clips" any more.

DURATION HANDLING (Moment Assembly only) is an OPTIMISATION, not an allocation. The
agent first selects enough meaningful moments to satisfy the editing intent; only if
their combined length overruns the target does :func:`compress_to_target` shave it back,
under these rules:

    * Fine-grained trimming is preferred over removing content.
    * Compression is PRIORITY-ORDERED: repetitive / transitional / low-impact moments
      (low importance) are compressed first and hardest; high-value moments retain
      proportionally more of their screen time and are the last to be touched.
    * Every moment has a retention FLOOR — it can never be shaved to an unreadable
      flash, and a protected moment is never trimmed at all.
    * A trim NEVER leaves the moment's original event boundaries; the kept window is
      placed around the moment's editorial focus (its centre by default), so nothing is
      mechanically trimmed "from the beginning".
    * This module NEVER removes a moment. When trimming alone cannot reach the target it
      reports the shortfall plus the weakest-contribution candidates, and the AGENT makes
      the editorial call — anything it drops is reported back as alternative/backup
      material with a reason.

Everything here is PURE and deterministic (no LLM, no I/O).
"""

from __future__ import annotations

import json
from pathlib import Path

# ── Compression tuning ────────────────────────────────────────────────────────
# A moment shorter than this reads as a flash frame rather than a beat, so no trim ever
# takes a moment below it (unless the source event is itself shorter).
MIN_MOMENT_SECONDS = 1.0
# Retention floor as a FRACTION of a moment's own length, interpolated by importance:
# the least valuable moment in the set may lose over half its length, the most valuable
# keeps nearly all of it. This is what makes high-value moments retain more screen time.
RETAIN_LOW = 0.45
RETAIN_HIGH = 0.90
# Fraction of the kept window that sits BEFORE the moment's focus point (0.5 = centred).
FOCUS_BIAS = 0.5

_TOL = 0.05          # seconds — arithmetic tolerance
_ON_TARGET_FRAC = 0.02   # within 2% (or 0.5s) of the target counts as "on target"

MODE_CLIP = "clip_assembly"
MODE_MOMENT = "moment_assembly"


# ── Small helpers ─────────────────────────────────────────────────────────────


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _weights(importance, n: int) -> list[float]:
    """Normalise a per-item importance list to exactly ``n`` positive floats."""
    base = [max(0.0, _f(w, 1.0)) for w in (importance or [])]
    out = [(base[i] if i < len(base) else 1.0) for i in range(n)]
    return [w if w > 0 else 1.0 for w in out]


def _aligned(values, n: int, default=None) -> list:
    seq = list(values or [])
    return [(seq[i] if i < len(seq) else default) for i in range(n)]


# Backup material is a CURATED shortlist, not an inventory of everything that did not make
# the cut: the editor wants the few genuine alternatives, so a long list is truncated here
# (and the drop reported via ``excluded_omitted`` — never silently). The Selection Agent's
# `excluded_json` docstrings quote this number — keep them in sync if it changes.
MAX_BACKUP_ITEMS = 5

# Fields an excluded entry can carry beyond the agent's own words. They are filled in by
# the caller from the CATALOGUE (real filename + measured timecodes), never by the model —
# an event id like "9" means nothing to an editor, "warmup.mov 18.0s-26.0s" does.
# ``also``/``also_details`` let ONE entry stand for several near-identical rejected
# moments, so grouping does not cost shortlist slots.
_ALTERNATE_RESOLVED_FIELDS = ("event_id", "file_path", "start_seconds", "end_seconds",
                              "label", "also_details")


def _as_ref_list(value) -> list[str]:
    """Coerce a grouped-reference field into a list of plain ref strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def normalise_alternates(raw) -> list[dict]:
    """Coerce the agent's excluded/backup material into a uniform list of dicts.

    Accepts a JSON string, a list of dicts, or a list of plain strings. Every entry ends
    up as ``{"ref", "name", "reason", "suggested_use"}`` — the reason a moment or clip did
    NOT make the edit, and how it could still be used in another one — plus, when the
    caller has already resolved them against the catalogue, ``event_id`` / ``file_path`` /
    ``start_seconds`` / ``end_seconds`` / ``label`` so the editor sees a real clip and real
    timecodes instead of a database id. Those resolved fields pass through untouched, so
    running an already-enriched list back through this function is lossless.

    An entry may also carry ``also`` — further refs the SAME rejection covers — so several
    near-identical moments group into one shortlist item instead of flooding it.

    Nothing is invented: a missing reason stays an empty string, and an entry that could
    not be matched to a catalogued clip simply has no file path or timecodes.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return [{"ref": raw.strip(), "name": Path(raw.strip()).name,
                     "reason": "", "suggested_use": ""}] if raw.strip() else []
    if isinstance(raw, dict):
        raw = [raw]
    out: list[dict] = []
    for item in (raw or []):
        if isinstance(item, str):
            entry = {"ref": item.strip(), "reason": "", "suggested_use": "", "also": []}
        elif isinstance(item, dict):
            entry = {
                "ref": str(item.get("ref") or item.get("event_id") or item.get("file_path")
                           or item.get("name") or "").strip(),
                "reason": str(item.get("reason") or "").strip(),
                "suggested_use": str(item.get("suggested_use")
                                     or item.get("use") or "").strip(),
                "also": _as_ref_list(item.get("also") or item.get("also_refs")
                                     or item.get("similar")),
            }
            for key in _ALTERNATE_RESOLVED_FIELDS:
                if item.get(key) is not None:
                    entry[key] = item[key]
            if item.get("name"):
                entry["name"] = str(item["name"])
        else:
            continue
        if not (entry["ref"] or entry["reason"]):
            continue
        entry.setdefault("name", Path(entry["ref"]).name if entry["ref"] else "")
        out.append(entry)
    return out


# Below this many segments an "order is unchanged" check is meaningless — with two items
# a coincidental match is 50/50, so flagging it would be noise rather than signal.
MIN_ORDER_CHECK_ITEMS = 3


def compare_order(emitted: list, reference: list, *, reference_label: str = "") -> dict:
    """Flag a timeline whose order is IDENTICAL to the order the material arrived in.

    Ordering is meant to be an editorial decision, but the agent sees its candidates in a
    fixed order (moments grouped by file and chronological within each file; clips in the
    editor's Bin order), and an LLM can simply echo that listing back. Nothing downstream
    can tell the difference — a deliberate chronological edit and a lazy one produce the
    same plan — so this records the comparison for the agent and the editor to judge.

    It is a PROMPT, never a rejection: chronological is genuinely right for a tactical
    breakdown or a build-up-to-payoff story. The check just makes the coincidence visible
    instead of silent.

    Args:
        emitted: The order the agent chose, as comparable keys.
        reference: The same keys in the order the material was presented in.
        reference_label: Human-readable name for that reference order.

    Returns:
        ``{"checked", "unchanged", "reference", "items"}``. ``checked`` is False when
        there are too few segments, or the reference could not be reconstructed.
    """
    n = len(emitted)
    if n < MIN_ORDER_CHECK_ITEMS or len(reference) != n:
        return {"checked": False, "unchanged": False,
                "reference": reference_label, "items": n}
    return {"checked": True, "unchanged": list(emitted) == list(reference),
            "reference": reference_label, "items": n}


def _duration_status(total: float, target: float | None) -> tuple[str, float]:
    """Classify the plan's length against its target. Returns ``(status, delta)``."""
    if not target or target <= 0:
        return "unconstrained", 0.0
    slack = max(0.5, target * _ON_TARGET_FRAC)
    if abs(total - target) <= slack:
        return "on_target", round(total - target, 3)
    if total > target:
        return "over_target", round(total - target, 3)
    return "under_target", round(target - total, 3)


# ── Duration optimisation (Moment Assembly only) ──────────────────────────────


def compress_to_target(spans: list[float], target_seconds: float | None, *,
                       importance: list[float] | None = None,
                       protect: list[bool] | None = None,
                       min_moment_seconds: float = MIN_MOMENT_SECONDS) -> dict:
    """Shave a set of moments back towards ``target_seconds`` — least valuable first.

    This is deliberately NOT an allocator: nothing is "split" across the moments and no
    screen time is handed out proportionally to importance. Every moment starts at its
    FULL measured length and is only reduced when there is a real overrun, in ascending
    order of importance:

        1. Each moment gets a retention FLOOR — ``max(min_moment_seconds, length ×
           retain)`` where ``retain`` scales from ``RETAIN_LOW`` for the least important
           moment in the set to ``RETAIN_HIGH`` for the most important. Protected moments
           get a floor equal to their full length (never trimmed).
        2. Moments are banded into importance TIERS. The lowest tier absorbs as much of
           the overrun as its slack allows (shared inside the tier in proportion to each
           moment's own slack, so nothing in a tier is singled out), then the next tier
           up, and so on. Compression stops the instant the target is met — high-value
           tiers are usually never reached.
        3. If every floor is hit and the plan is still long, the residual is reported as
           ``shortfall`` and NOTHING is removed. Removal is an editorial decision the
           agent must make and justify.

    Args:
        spans: Each moment's full measured length in seconds, in timeline order.
        target_seconds: Target TOTAL length, or ``None``/0 for no constraint.
        importance: Per-moment editorial weight (any positive scale, higher = more
            valuable). Defaults to equal weighting.
        protect: Per-moment flags — ``True`` means "never trim this moment".
        min_moment_seconds: Absolute floor for any single trimmed moment.

    Returns:
        ``{"keeps", "floors", "excess", "absorbed", "shortfall", "notes"}`` where
        ``keeps[i]`` is the seconds moment ``i`` should retain (≤ its full length).
    """
    n = len(spans)
    full = [max(0.0, _f(s)) for s in spans]
    keeps = list(full)
    total = sum(full)
    idle = {"keeps": [round(k, 3) for k in keeps], "floors": [round(f, 3) for f in full],
            "excess": 0.0, "absorbed": 0.0, "shortfall": 0.0, "notes": []}
    if n == 0 or not target_seconds or target_seconds <= 0:
        return idle
    excess = total - float(target_seconds)
    if excess <= _TOL:
        return idle

    w = _weights(importance, n)
    prot = [bool(p) for p in _aligned(protect, n, False)]
    lo, hi = min(w), max(w)

    floors: list[float] = []
    for i in range(n):
        if prot[i] or full[i] <= 0:
            floors.append(full[i])
            continue
        norm = 0.5 if hi <= lo else (w[i] - lo) / (hi - lo)
        retain = RETAIN_LOW + (RETAIN_HIGH - RETAIN_LOW) * norm
        floors.append(min(full[i], max(min_moment_seconds, full[i] * retain)))

    tiers: dict[float, list[int]] = {}
    for i in range(n):
        tiers.setdefault(round(w[i], 3), []).append(i)

    remaining = excess
    notes: list[str] = []
    for tier_weight in sorted(tiers):              # ascending importance = compress first
        if remaining <= _TOL:
            break
        idxs = tiers[tier_weight]
        slack = {i: max(0.0, full[i] - floors[i]) for i in idxs}
        pool = sum(slack.values())
        if pool <= _TOL:
            continue
        take = min(remaining, pool)
        touched = 0
        for i in idxs:
            if slack[i] <= 0:
                continue
            keeps[i] = max(floors[i], keeps[i] - take * slack[i] / pool)
            touched += 1
        remaining -= take
        notes.append(f"importance {tier_weight:g}: trimmed {take:.1f}s across "
                     f"{touched} moment(s)")

    return {
        "keeps": [round(k, 3) for k in keeps],
        "floors": [round(f, 3) for f in floors],
        "excess": round(excess, 3),
        "absorbed": round(excess - max(0.0, remaining), 3),
        "shortfall": round(max(0.0, remaining), 3),
        "notes": notes,
    }


def place_window(start: float, end: float, keep: float,
                 focus: float | None = None, bias: float = FOCUS_BIAS) -> tuple[float, float]:
    """Place a ``keep``-second window INSIDE ``[start, end]``, around the editorial focus.

    The kept window is centred on ``focus`` (the moment's peak — its midpoint when the
    agent gives none), with ``bias`` of it before that point, then clamped so it can never
    leave the original event boundaries. This is why nothing is trimmed by a fixed rule
    such as "always drop the first N seconds": what survives is the core of the action.
    """
    span = max(0.0, end - start)
    if keep >= span - 1e-9 or span <= 0:
        return start, end
    centre = start + span * 0.5 if focus is None else min(max(float(focus), start), end)
    in_p = centre - keep * bias
    in_p = min(max(in_p, start), end - keep)
    return in_p, in_p + keep


# ── Timeline builders (one per editing mode) ──────────────────────────────────


def _finish(mode: str, segments: list[dict], *, target_seconds: float | None = None,
            raw_seconds: float | None = None, excluded=None, aspect_ratio: str = "",
            ordering_strategy: str = "", order_check: dict | None = None,
            extra: dict | None = None) -> dict:
    """Assemble the common plan envelope shared by both modes."""
    total = round(sum(s["duration"] for s in segments), 3)
    status, delta = _duration_status(total, target_seconds)
    validation_errors = [
        {"order": s["order"], "name": s["name"], "error": s["validation_error"]}
        for s in segments if not s["valid"]
    ]
    # Backup material is a SHORTLIST of real alternatives the agent weighed and rejected —
    # not every unused clip/event. The agent is told to send few and strongest-first; if it
    # over-sends anyway, the surplus is cut here and COUNTED (never dropped silently).
    alternates = normalise_alternates(excluded)
    kept_alternates = alternates[:MAX_BACKUP_ITEMS]
    plan = {
        "mode": mode,
        # The SHAPE the agent chose and why (declared before it ordered anything), plus the
        # check on whether the emitted order merely echoes how the material was listed.
        # Order is an editorial decision, so it is recorded as one rather than left implicit.
        "ordering_strategy": (ordering_strategy or "").strip(),
        "order_check": order_check or {"checked": False, "unchanged": False,
                                       "reference": "", "items": len(segments)},
        # The editor's OUTPUT aspect ratio rides along the plan untouched. Nothing in this
        # module acts on it — it is a delivery SPEC, not an editing decision — but carrying
        # it here is what gets it from the Selection UI to the Delivery compiler.
        "aspect_ratio": aspect_ratio or None,
        "target_seconds": round(float(target_seconds), 3) if target_seconds else None,
        "raw_seconds": round(float(raw_seconds), 3) if raw_seconds is not None else total,
        "total_seconds": total,
        "duration_status": status,
        "duration_delta": delta,
        "valid": not validation_errors,
        "validation_errors": validation_errors,
        "segments": segments,
        "excluded": kept_alternates,
        "excluded_omitted": len(alternates) - len(kept_alternates),
    }
    if extra:
        plan.update(extra)
    return plan


def build_clip_timeline(clips: list[dict], labels: list[str] | None = None,
                        excluded=None, aspect_ratio: str = "",
                        ordering_strategy: str = "",
                        order_check: dict | None = None) -> dict:
    """CLIP ASSEMBLY — ordered COMPLETE clips, each at its full measured length.

    The unit of editing is the whole clip. There is no trimming, no screen-time
    allocation and no target duration: the plan's length is simply the sum of the clips
    the agent kept. All the editorial work in this mode is the agent's — dropping
    candidates that do not serve the intent (reported via ``excluded``) and deciding the
    order, which is preserved EXACTLY (clip 0 = timeline step 1).

    Args:
        clips: Ordered, resolved clip dicts with ``file_path`` and (ideally)
            ``duration_seconds``; ``shot_id`` optional.
        labels: Optional per-clip step labels ("cold open", "hero moment", ...).
        excluded: Candidate clips deliberately left out — see :func:`normalise_alternates`.
        aspect_ratio: The editor's requested OUTPUT aspect ratio, carried through to
            Delivery untouched. It is a delivery spec, so it changes nothing here — no
            clip is cropped, resized or reframed by the planner.
        ordering_strategy: The agent's one-line statement of the SHAPE it ordered the
            clips into, recorded on the plan so the choice is explicit and reviewable.
        order_check: Result of :func:`compare_order` against the candidate list order.

    Returns:
        The standard plan dict with ``mode='clip_assembly'``.
    """
    segments = []
    for i, c in enumerate(clips):
        src = round(max(0.0, _f(c.get("duration_seconds"))), 3)
        # A measured length gives explicit in/out points. An UNKNOWN length (0) emits
        # in/out = None so the compiler uses the whole clip — we never fabricate a range.
        in_p, out_p = (0.0, src) if src > 0 else (None, None)
        segments.append({
            "order": i + 1,
            "shot_id": c.get("shot_id"),
            "file_path": c.get("file_path"),
            "name": Path(c.get("file_path") or "").name,
            "label": (labels[i] if labels and i < len(labels) else ""),
            "importance": 1.0,
            "source_duration": src,
            "in_point": in_p,
            "out_point": out_p,
            "duration": src,
            "trimmed": False,
            "trim_note": "",
            "valid": True,
            "validation_error": None,
        })
    return _finish(MODE_CLIP, segments, target_seconds=None, excluded=excluded,
                   aspect_ratio=aspect_ratio, ordering_strategy=ordering_strategy,
                   order_check=order_check)


def build_moment_timeline(events: list[dict], *, target_seconds: float | None = None,
                          importance: list[float] | None = None,
                          labels: list[str] | None = None,
                          focus: list[float | None] | None = None,
                          protect: list[bool] | None = None,
                          excluded=None, aspect_ratio: str = "",
                          ordering_strategy: str = "", order_check: dict | None = None,
                          min_moment_seconds: float = MIN_MOMENT_SECONDS) -> dict:
    """MOMENT ASSEMBLY — ordered temporal moments, optionally optimised to a target.

    Each segment starts as the event's OWN measured ``start_seconds``/``end_seconds``, so
    the edit cuts to the exact moment the action happens. Order is preserved EXACTLY.

    When ``target_seconds`` is given AND the chosen moments overrun it, the overrun is
    absorbed by :func:`compress_to_target` — low-value moments compressed first, every
    trim kept strictly inside its own event boundaries and placed around the moment's
    focus. Nothing is ever removed here; an overrun that trimming cannot absorb is
    reported (``duration_status='over_target'`` + ``removal_candidates``) for the agent to
    resolve editorially. When the moments fall SHORT of the target, footage is never
    stretched — the plan simply reports ``under_target``.

    Args:
        events: Ordered event dicts with ``file_path``, ``start_seconds``, ``end_seconds``
            and ideally ``shot_id``, ``event_id`` and the parent ``source_duration``.
        target_seconds: Optional target TOTAL length for the edit.
        importance: Per-moment editorial weight (higher = more valuable, harder to cut).
        labels: Optional per-moment step labels (defaults to the event's action text).
        focus: Per-moment peak timecode (absolute, inside the event) to trim around.
        protect: Per-moment "never trim" flags.
        excluded: Moments deliberately left out — see :func:`normalise_alternates`.
        aspect_ratio: The editor's requested OUTPUT aspect ratio, carried through to
            Delivery untouched (a delivery spec — no clip is cropped, resized or reframed
            here, and it never changes a moment's in/out points).
        ordering_strategy: The agent's one-line statement of the SHAPE it arranged the
            moments into, recorded on the plan so the choice is explicit and reviewable.
        order_check: Result of :func:`compare_order` against the source chronology.
        min_moment_seconds: Absolute floor for any single trimmed moment.

    Returns:
        The standard plan dict with ``mode='moment_assembly'``, plus ``compression`` and
        ``removal_candidates``.
    """
    n = len(events)
    weights = _weights(importance, n)
    focuses = _aligned(focus, n, None)
    protects = [bool(p) for p in _aligned(protect, n, False)]

    starts, ends, spans, srcs = [], [], [], []
    for e in events:
        start = max(0.0, _f(e.get("start_seconds")))
        end = _f(e.get("end_seconds"))
        starts.append(start)
        ends.append(end)
        spans.append(max(0.0, end - start))
        srcs.append(_f(e.get("source_duration") or e.get("clip_duration")))

    raw_seconds = sum(spans)
    report = compress_to_target(spans, target_seconds, importance=weights,
                                protect=protects, min_moment_seconds=min_moment_seconds)
    keeps = report["keeps"]

    segments = []
    for i, e in enumerate(events):
        start, end, src = starts[i], ends[i], srcs[i]
        keep = min(keeps[i], spans[i])
        in_p, out_p = place_window(start, end, keep, focuses[i])
        dur = round(max(0.0, out_p - in_p), 3)
        trimmed = spans[i] - dur > _TOL

        valid, validation_error = True, None
        if spans[i] <= 0:
            valid, validation_error = False, f"event range is empty ({start:g}s–{end:g}s)"
        elif src > 0 and end > src + 1e-6:
            valid, validation_error = False, (
                f"event out-point {end:g}s exceeds source length {src:g}s")
        elif dur <= 0:
            valid, validation_error = False, (
                f"nothing remains of the moment after optimisation "
                f"(event {start:g}s–{end:g}s)")

        segments.append({
            "order": i + 1,
            "shot_id": e.get("shot_id"),
            "event_id": e.get("event_id"),
            "file_path": e.get("file_path"),
            "name": Path(e.get("file_path") or "").name,
            "label": (labels[i] if labels and i < len(labels)
                      else (e.get("action") or "")[:60]),
            "importance": round(weights[i], 3),
            "source_duration": round(src, 3),
            "event_start": round(start, 3),
            "event_end": round(end, 3),
            "event_duration": round(spans[i], 3),
            "in_point": round(in_p, 3),
            "out_point": round(out_p, 3),
            "duration": dur,
            "trimmed": trimmed,
            "trim_note": (f"compressed {spans[i] - dur:.1f}s (kept {in_p:.1f}s–{out_p:.1f}s "
                          f"of the {start:.1f}s–{end:.1f}s event)") if trimmed else "",
            "protected": protects[i],
            "valid": valid,
            "validation_error": validation_error,
        })

    # Weakest-contribution moments, offered to the AGENT when trimming alone fell short.
    # This module never acts on them — removal is an editorial decision, not a rule.
    removal_candidates = []
    if report["shortfall"] > _TOL:
        ranked = sorted(range(n), key=lambda i: (weights[i], -spans[i]))
        for i in ranked[:3]:
            if protects[i]:
                continue
            removal_candidates.append({
                "order": i + 1,
                "event_id": events[i].get("event_id"),
                "name": segments[i]["name"],
                "label": segments[i]["label"],
                "importance": round(weights[i], 3),
                "would_save_seconds": segments[i]["duration"],
            })

    return _finish(
        MODE_MOMENT, segments,
        target_seconds=target_seconds, raw_seconds=raw_seconds, excluded=excluded,
        aspect_ratio=aspect_ratio, ordering_strategy=ordering_strategy,
        order_check=order_check,
        extra={
            "compression": {
                "applied": report["absorbed"] > _TOL,
                "overrun_seconds": report["excess"],
                "absorbed_seconds": report["absorbed"],
                "shortfall_seconds": report["shortfall"],
                "trimmed_count": sum(1 for s in segments if s["trimmed"]),
                "notes": report["notes"],
            },
            "removal_candidates": removal_candidates,
        },
    )
