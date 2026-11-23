"""Deterministic timeline planning for Selection."""

from __future__ import annotations
import json
from pathlib import Path

MIN_MOMENT_SECONDS = 1.0
RETAIN_LOW = 0.45
RETAIN_HIGH = 0.90
FOCUS_BIAS = 0.5

_TOL = 0.05          
_ON_TARGET_FRAC = 0.02   

MODE_CLIP = "clip_assembly"
MODE_MOMENT = "moment_assembly"


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _weights(importance, n: int) -> list[float]:
    """Return exactly n positive importance weights."""
    base = [max(0.0, _f(w, 1.0)) for w in (importance or [])]
    out = [(base[i] if i < len(base) else 1.0) for i in range(n)]
    return [w if w > 0 else 1.0 for w in out]


def _aligned(values, n: int, default=None) -> list:
    seq = list(values or [])
    return [(seq[i] if i < len(seq) else default) for i in range(n)]

MAX_BACKUP_ITEMS = 5

_ALTERNATE_RESOLVED_FIELDS = ("event_id", "file_path", "start_seconds", "end_seconds",
                              "label", "also_details")


def _as_ref_list(value) -> list[str]:
    """Normalise a grouped reference field to a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def normalise_alternates(raw) -> list[dict]:
    """Normalise excluded or backup material into a consistent structure."""
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


MIN_ORDER_CHECK_ITEMS = 3


def compare_order(emitted: list, reference: list, *, reference_label: str = "") -> dict:
    """Report whether the emitted order matches the input order."""
    n = len(emitted)
    if n < MIN_ORDER_CHECK_ITEMS or len(reference) != n:
        return {"checked": False, "unchanged": False,
                "reference": reference_label, "items": n}
    return {"checked": True, "unchanged": list(emitted) == list(reference),
            "reference": reference_label, "items": n}


def _duration_status(total: float, target: float | None) -> tuple[str, float]:
    """Classify total duration relative to the target."""
    if not target or target <= 0:
        return "unconstrained", 0.0
    slack = max(0.5, target * _ON_TARGET_FRAC)
    if abs(total - target) <= slack:
        return "on_target", round(total - target, 3)
    if total > target:
        return "over_target", round(total - target, 3)
    return "under_target", round(target - total, 3)


def compress_to_target(spans: list[float], target_seconds: float | None, *,
                       importance: list[float] | None = None,
                       protect: list[bool] | None = None,
                       min_moment_seconds: float = MIN_MOMENT_SECONDS) -> dict:
    """Compress moments towards a target, prioritising lower-importance moments."""
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
    for tier_weight in sorted(tiers):             
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
    """Place a shortened window inside the original event boundaries."""
    span = max(0.0, end - start)
    if keep >= span - 1e-9 or span <= 0:
        return start, end
    centre = start + span * 0.5 if focus is None else min(max(float(focus), start), end)
    in_p = centre - keep * bias
    in_p = min(max(in_p, start), end - keep)
    return in_p, in_p + keep


def _finish(mode: str, segments: list[dict], *, target_seconds: float | None = None,
            raw_seconds: float | None = None, excluded=None, aspect_ratio: str = "",
            ordering_strategy: str = "", order_check: dict | None = None,
            extra: dict | None = None) -> dict:
    """Build the common timeline plan structure."""
    total = round(sum(s["duration"] for s in segments), 3)
    status, delta = _duration_status(total, target_seconds)
    validation_errors = [
        {"order": s["order"], "name": s["name"], "error": s["validation_error"]}
        for s in segments if not s["valid"]
    ]

    alternates = normalise_alternates(excluded)
    kept_alternates = alternates[:MAX_BACKUP_ITEMS]
    plan = {
        "mode": mode,
        "ordering_strategy": (ordering_strategy or "").strip(),
        "order_check": order_check or {"checked": False, "unchanged": False,
                                       "reference": "", "items": len(segments)},
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
    """Build a timeline from complete clips without trimming."""
    segments = []
    for i, c in enumerate(clips):
        src = round(max(0.0, _f(c.get("duration_seconds"))), 3)
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
    """Build a timeline from temporal events with optional duration compression."""
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
