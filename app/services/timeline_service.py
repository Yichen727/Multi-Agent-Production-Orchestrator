"""Timeline planning service — turn an ordered set of clips into structured segments.

This backs the Selection stage's operating modes:

    TRIM MODE    — the intent asks to drop the first/last N seconds of EACH clip (a
                   per-clip head/tail trim). Every clip keeps its full remaining middle
                   and clips are assembled sequentially in order.
    TIMED MODE   — the editing intent names a target length ("15s", "1 min",
                   "2 minutes", "1:30"). The target is split across the clips
                   PROPORTIONALLY TO THEIR IMPORTANCE and each clip is trimmed to its
                   allotted length, producing a time-coded timeline whose total ≈ target
                   (capped by how much footage actually exists).
    FULL CLIP MODE — no duration is named. Clips keep their full length and are simply
                   concatenated in order, untrimmed.

Everything here is PURE and deterministic (no LLM, no I/O): the Selection Agent's tool
feeds it resolved clip metadata and it returns structured segments. Downstream, the
Delivery Agent compiles those segments verbatim — order and timings preserved.
"""

from __future__ import annotations

import re
from pathlib import Path

# Duration tokens. Minutes must be tried before the bare "m"/"s" so "1m30s" and
# "2 minutes" both parse. A negative lookahead stops "m"/"s" swallowing a following
# letter (so "promo" / "seconds" aren't mis-read) while still allowing a digit after
# (so "1m30s" works).
_MINUTES_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:minutes|minute|mins|min|m)(?![a-z])")
_SECONDS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:seconds|second|secs|sec|s)(?![a-z])")
_CLOCK_RE = re.compile(r"\b(\d{1,3}):([0-5]?\d)\b")  # mm:ss

# A PER-CLIP trim phrase ("trim the first 2 seconds", "drop the last 5s", "cut 3
# seconds off the start") describes head/tail trimming of EACH clip — it is NOT a
# target TOTAL length. Such durations are stripped before target detection so they
# never get summed into a bogus target (the bug that shrank every clip to ~0.2s).
_TRIM_PHRASE_RE = re.compile(
    r"\b(?:first|last|initial|final|leading|trailing|head|tail)\s+"
    r"\d+(?:\.\d+)?\s*(?:seconds|second|secs|sec|s|minutes|minute|mins|min|m)(?![a-z])",
    re.IGNORECASE,
)


def parse_target_duration(text: str | None) -> float | None:
    """Detect a target TOTAL duration in free text and return it in seconds, else ``None``.

    Recognises, in one string, any mix of:
        - clock form ``mm:ss``  (e.g. "1:30" → 90)
        - minutes ("1 min", "2 minutes", "1m")
        - seconds ("15s", "30 sec", "45 seconds")
        - combined ("1m30s", "1 min 30 s")

    Per-clip TRIM phrases ("first/last N seconds") are ignored — they mean head/tail
    trimming of each clip, not a target length (see TRIM MODE in ``plan_segments``).

    Returns ``None`` when no duration is present (→ FULL CLIP MODE). Never guesses a
    number that carries no time unit.
    """
    if not text:
        return None
    # Remove per-clip trim phrases first so "trim the first/last 2 seconds" cannot be
    # mistaken for (and summed into) a target total length.
    t = _TRIM_PHRASE_RE.sub(" ", text.lower())

    clock = _CLOCK_RE.search(t)
    if clock:
        return float(int(clock.group(1)) * 60 + int(clock.group(2)))

    total = 0.0
    found = False
    for m in _MINUTES_RE.finditer(t):
        total += float(m.group(1)) * 60.0
        found = True
    for s in _SECONDS_RE.finditer(t):
        total += float(s.group(1))
        found = True
    return round(total, 3) if found else None


def allocate_durations(sources: list[float], weights: list[float],
                       target: float) -> list[float]:
    """Split ``target`` seconds across clips proportionally to ``weights``.

    A clip can never be given more than its own source length (water-filling: when a
    clip's proportional share exceeds the footage available, it is capped at its full
    length and the surplus is redistributed among the clips that still have room). A
    source length of 0/unknown is treated as uncapped so the clip still receives its
    share. Clips with weight ≤ 0 receive nothing.

    Returns a list of allocated seconds, aligned to ``sources``. Its sum is
    ``min(target, total available footage)``.
    """
    n = len(sources)
    if n == 0:
        return []

    # Normalise weights: fall back to equal weighting if none are positive.
    weights = [max(0.0, float(w)) for w in weights]
    if sum(weights) <= 0:
        weights = [1.0] * n

    caps = [s if (s and s > 0) else float("inf") for s in sources]
    total_footage = sum(c for c in caps if c != float("inf"))
    uncapped = any(c == float("inf") for c in caps)
    achievable = target if uncapped else min(target, total_footage)

    alloc = [0.0] * n
    active = {i for i in range(n) if weights[i] > 0}

    # At most n rounds: each round either caps ≥1 clip or finishes.
    for _ in range(n + 1):
        remaining = achievable - sum(alloc)
        if remaining <= 1e-9 or not active:
            break
        wsum = sum(weights[i] for i in active)
        if wsum <= 0:
            break
        capped_now = []
        for i in active:
            share = remaining * weights[i] / wsum
            room = caps[i] - alloc[i]
            if room != float("inf") and share >= room - 1e-12:
                alloc[i] = caps[i]
                capped_now.append(i)
        if capped_now:
            active -= set(capped_now)
            continue
        for i in active:
            alloc[i] += remaining * weights[i] / wsum
        break

    return [round(a, 3) for a in alloc]


def plan_segments(clips: list[dict], target_seconds: float | None,
                  weights: list[float] | None = None,
                  labels: list[str] | None = None,
                  head_trim: float = 0.0, tail_trim: float = 0.0) -> dict:
    """Build the structured timeline plan from ordered, resolved clips.

    Order is preserved EXACTLY (clip 0 is timeline step 1). Three modes:

        TRIM MODE (``head_trim`` and/or ``tail_trim`` > 0) — drop ``head_trim`` seconds
            off the START and ``tail_trim`` seconds off the END of EVERY clip, keeping
            the FULL remaining middle. Clips are assembled sequentially, untrimmed
            otherwise. Takes precedence over any target. A clip shorter than
            ``head_trim + tail_trim`` has no middle → its duration is 0 (reported by the
            caller, never fabricated).
        TIMED MODE (a positive ``target_seconds``) — the target is split across clips
            proportionally to importance and each clip trimmed from its head to its share.
        FULL CLIP MODE (neither) — each clip keeps its full source duration.

    Args:
        clips: Ordered clip dicts, each with ``file_path`` and (ideally)
            ``duration_seconds``; ``shot_id`` optional.
        target_seconds: Target total length, or ``None`` for full-clip mode.
        weights: Per-clip importance weights (aligned to ``clips``); defaults to equal.
        labels: Optional per-clip free-form step labels.
        head_trim: Seconds to drop from the START of every clip (TRIM MODE).
        tail_trim: Seconds to drop from the END of every clip (TRIM MODE).

    Returns:
        ``{"mode", "target_seconds", "head_trim", "tail_trim", "total_seconds",
        "segments": [...]}`` where each segment has order/shot_id/file_path/name/label/
        importance/source_duration/in_point/out_point/duration.
    """
    n = len(clips)
    sources = [float(c.get("duration_seconds") or 0.0) for c in clips]
    if weights is None or len(weights) != n:
        base = list(weights or [])
        weights = [(base[i] if i < len(base) else 1.0) for i in range(n)]
    weights = [max(0.0, float(w)) for w in weights]

    head_trim = max(0.0, float(head_trim or 0.0))
    tail_trim = max(0.0, float(tail_trim or 0.0))
    trimming = head_trim > 0 or tail_trim > 0
    timed = (not trimming) and target_seconds is not None and target_seconds > 0

    alloc = allocate_durations(sources, weights, float(target_seconds)) if timed else None

    segments = []
    for i, c in enumerate(clips):
        src = sources[i]
        if trimming:
            # Keep the middle: [head_trim, src - tail_trim]. When the clip is too short
            # to have a middle, in==out → duration 0 (a real fact, not fabricated).
            if src > 0:
                in_p = min(head_trim, src)
                out_p = max(in_p, src - tail_trim)
            else:
                in_p = out_p = 0.0  # unknown source length — cannot trim, leave to Delivery
            dur = round(max(0.0, out_p - in_p), 3)
        elif timed:
            in_p = 0.0
            dur = round(alloc[i], 3)
            out_p = dur
        else:  # full clip
            in_p = 0.0
            dur = round(src, 3)
            out_p = dur
        segments.append({
            "order": i + 1,
            "shot_id": c.get("shot_id"),
            "file_path": c.get("file_path"),
            "name": Path(c.get("file_path") or "").name,
            "label": (labels[i] if labels and i < len(labels) else ""),
            "importance": round(weights[i], 3),
            "source_duration": round(src, 3),
            "in_point": round(in_p, 3),
            "out_point": round(out_p, 3),
            "duration": dur,
        })

    mode = "trim" if trimming else ("timed" if timed else "full")
    return {
        "mode": mode,
        "target_seconds": round(float(target_seconds), 3) if timed else None,
        "head_trim": head_trim,
        "tail_trim": tail_trim,
        "total_seconds": round(sum(s["duration"] for s in segments), 3),
        "segments": segments,
    }
