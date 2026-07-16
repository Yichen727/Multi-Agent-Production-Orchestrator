"""Retrieval service — unified HYBRID search over the production catalogue.

This is the single semantic-recall entry point for the MAPO Search stage. It replaces
the old set of fragmented per-attribute SQL tools with one function that combines:

    1. SQL HARD FILTERS  — orientation, duration, people, shot type. Cheaply
       narrows the catalogue to rows that satisfy the structured constraints.
    2. VECTOR SEMANTIC RECALL — when a free-text ``keywords`` query is given and the
       filtered rows carry embeddings (written at ingest), the query is embedded and
       ranked by cosine similarity against them.
    3. LEXICAL FALLBACK — when embeddings are unavailable (no API key, or the demo
       seed which has no vectors), it degrades to keyword/description substring
       scoring rather than returning nothing.

Both the Search Agent's ``search_catalogue`` tool and the Streamlit curation UI call
``hybrid_search`` directly, so retrieval behaves identically whether driven by the LLM
or by the user. Nothing here fabricates data: it only ranks rows that already exist in
the catalogue.
"""

import json

import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy import text as _sql

from app.services.database_service import _engine
from app.services.openai_service import embed_texts, llm_fast
from app.utils.logger import get_logger

logger = get_logger("retrieval_service")


# Columns pulled for every candidate — the metadata the UI preview and the Selection
# Agent need. ``embedding`` is fetched too but stripped from the returned dict.
_SELECT_COLUMNS = (
    "shot_id, file_path, shot_type, duration_seconds, "
    "orientation, width, height, fps, keywords, description, "
    "people_count, camera_motion, lighting, mood, subject_position, embedding"
)

# Relevance combines a CALIBRATED vector-cosine signal with lexical term overlap.
# Raw text-embedding-3-small cosine for related text sits in a compressed ~0.18–0.55
# band, so it is remapped onto [0,1] (see ``_calibrate_cosine``) BEFORE being shown or
# blended — otherwise a genuinely strong semantic match under-reads as ~40%. The two
# signals are then combined with a REINFORCING (noisy-OR) rule, not a weighted average,
# so a strong hit in EITHER layer drives relevance up instead of one diluting the other.
_SUGGEST_THRESHOLD = 0.55
_NEUTRAL_THRESHOLD = 0.30

# Calibration band for raw cosine → relevance. At/below the floor a row is treated as
# unrelated (0.0); at/above the ceiling as a full semantic match (1.0); linear between.
_COSINE_FLOOR = 0.18
_COSINE_CEIL = 0.55


def _calibrate_cosine(cos: float) -> float:
    """Map a raw text-embedding-3-small cosine onto a calibrated [0,1] relevance.

    The model packs 'related' content into a narrow high-floor band, so the raw number
    is not a usable percentage. This stretches that band across the full 0–1 range.
    """
    if cos <= _COSINE_FLOOR:
        return 0.0
    if cos >= _COSINE_CEIL:
        return 1.0
    return (cos - _COSINE_FLOOR) / (_COSINE_CEIL - _COSINE_FLOOR)


# Query understanding: translate the user's free-text query to English and expand it
# into retrieval keywords. This is what the LLM-driven Search agent does in its prompt;
# it is factored out here so the UI's DIRECT (non-LLM) search path gets the same
# translation + synonym expansion instead of embedding a raw foreign-language sentence.
_EXPAND_INSTRUCTION = (
    "You expand a video-search query into English retrieval keywords. Given the user's "
    "query in ANY language, output ONLY a comma-separated list of lowercase English "
    "terms: translate the core visual concepts to English, then add close synonyms, "
    "broader/related terms, and common singular/plural variants. Drop filler words "
    "(please, find, show, video, clip, footage). No sentences, no explanation — just "
    "the comma-separated terms."
)


def expand_query(raw_query: str) -> str:
    """Translate + expand a natural-language query into English retrieval keywords.

    Returns a comma-separated English keyword string. Falls back to the raw query
    unchanged when the query is empty or the LLM is unavailable (no API key / error),
    so search still runs — it just does not benefit from expansion.
    """
    q = (raw_query or "").strip()
    if not q:
        return q
    try:
        resp = llm_fast.invoke([SystemMessage(_EXPAND_INSTRUCTION), HumanMessage(q)])
        expanded = (resp.content or "").strip()
        return expanded or q
    except Exception as e:  # no key / network / model — degrade to the raw query
        logger.warning(f"Query expansion unavailable, using raw query: {e}")
        return q


def group_size(people_count) -> str:
    """Map a raw people_count to a coarse group label for editing decisions.

    Returns 'unknown' when the count was never measured (None) — never a guess.
    """
    if people_count is None:
        return "unknown"
    if people_count <= 0:
        return "none"
    if people_count == 1:
        return "solo"
    if people_count <= 5:
        return "group"
    return "crowd"


def _cosine(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between a query vector and a matrix of vectors.

    Zero vectors and shape mismatches yield 0.0 similarity rather than errors.
    """
    if matrix.size == 0:
        return np.array([])
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return np.zeros(matrix.shape[0])
    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms[row_norms == 0] = 1e-9
    return (matrix @ query_vec) / (row_norms * q_norm)


def _normalise_orientation(orientation: str | None) -> str | None:
    if not orientation:
        return None
    norm = orientation.strip().lower()
    norm = {"vertical": "portrait", "horizontal": "landscape"}.get(norm, norm)
    return norm if norm in ("portrait", "landscape", "square") else None


# Words that name a clip's FORMAT (orientation), not its content. Splitting them by
# ambiguity is what lets us hoist safely (see ``hoist_orientation``):
#   - UNAMBIGUOUS: 'horizontal'/'vertical' essentially never describe scene content.
#   - AMBIGUOUS: 'landscape'/'portrait'/'square' double as subject matter ("a beautiful
#     landscape", "a portrait of a woman"), so they only count as orientation when the
#     query is otherwise pure format.
_UNAMBIGUOUS_ORIENT = {"horizontal": "landscape", "vertical": "portrait",
                       "widescreen": "landscape"}
_AMBIGUOUS_ORIENT = {"landscape": "landscape", "portrait": "portrait", "square": "square"}

# Generic words that carry no CONTENT meaning. Stripped when parsing a free-text query
# so "find all horizontal shots" reduces to a pure orientation filter (nothing left to
# score), instead of scoring every clip against the noise word "shots".
_FILLER_WORDS = {
    "shot", "shots", "clip", "clips", "video", "videos", "footage", "scene", "scenes",
    "all", "any", "every", "find", "show", "get", "give", "me", "the", "a", "an", "of",
    "with", "please", "some",
}


def hoist_orientation(keywords: str | None,
                      orientation: str | None) -> tuple[str | None, str | None]:
    """Pull an orientation named inside a free-text query into the structured filter.

    "find all horizontal shots" must behave as an orientation FILTER — return only the
    matching clips, with no relevance % (every result is a full match) — not a keyword
    search that scores every clip against the word "horizontal". Both the LLM and the
    UI's ``expand_query`` routinely leave the orientation word in ``keywords``; this
    hoists it out DETERMINISTICALLY so neither path can misclassify a format request.

    Unambiguous format words are always hoisted; ambiguous ones only when nothing else
    content-bearing remains, so "beautiful landscape" still searches scenery. An
    ``orientation`` already supplied by the caller is trusted and left untouched.

    Returns (cleaned_keywords, orientation).
    """
    if not keywords or orientation:
        return keywords, orientation

    terms = [t.strip() for t in keywords.replace(",", " ").split() if t.strip()]
    detected = None          # a definite (unambiguous) orientation signal
    ambiguous = None         # first ambiguous orientation word seen
    content: list[str] = []  # everything that is neither format nor filler
    for t in terms:
        low = t.lower()
        if low in _UNAMBIGUOUS_ORIENT:
            detected = _UNAMBIGUOUS_ORIENT[low]          # drop from keywords
        elif low in _AMBIGUOUS_ORIENT:
            ambiguous = ambiguous or _AMBIGUOUS_ORIENT[low]
            content.append(t)                            # provisional content
        elif low not in _FILLER_WORDS:
            content.append(t)

    if detected:
        # A definite format word settles it: this IS a format request. Drop the
        # ambiguous orientation words too (they were just echoing the same format).
        content = [t for t in content if t.lower() not in _AMBIGUOUS_ORIENT]
        return (" ".join(content) or None), detected

    # No unambiguous word. If an ambiguous word stood ALONE (rest was filler), treat it
    # as a pure format request; otherwise leave it in the query as genuine content.
    non_ambiguous = [t for t in content if t.lower() not in _AMBIGUOUS_ORIENT]
    if ambiguous and not non_ambiguous:
        return None, ambiguous
    return (" ".join(content) or None), orientation


def _fetch_filtered(project_id, shot_type, orientation, people,
                    min_duration, max_duration) -> list[dict]:
    """Run the SQL hard-filter layer and return matching rows as dicts."""
    clauses = ["project_id = :pid"]
    params: dict = {"pid": project_id}

    orient = _normalise_orientation(orientation)
    if orient:
        clauses.append("orientation = :orient")
        params["orient"] = orient
    if shot_type:
        clauses.append("shot_type LIKE :stype")
        params["stype"] = f"%{shot_type.strip().lower()}%"
    if people is not None:
        clauses.append("people_count IS NOT NULL AND people_count >= :people")
        params["people"] = int(people)
    if min_duration is not None:
        clauses.append("duration_seconds >= :mindur")
        params["mindur"] = float(min_duration)
    if max_duration is not None:
        clauses.append("duration_seconds <= :maxdur")
        params["maxdur"] = float(max_duration)

    where = " AND ".join(clauses)
    query = f"SELECT {_SELECT_COLUMNS} FROM shots WHERE {where} ORDER BY file_path"

    with _engine.begin() as conn:
        rows = conn.execute(_sql(query), params)
        return [dict(r._mapping) for r in rows]


# A clip that matches this many distinct query terms is considered a full lexical hit.
# Using a small saturating target (rather than a fraction of ALL terms) means a
# GENEROUS synonym expansion no longer dilutes the score — matching ~3 strong terms
# scores 1.0 whether the query had 3 synonyms or 30.
_LEXICAL_TARGET_HITS = 3


def _lexical_score(row: dict, terms: list[str]) -> float:
    """Saturating lexical overlap between the query terms and a row's tags (0–1)."""
    if not terms:
        return 0.0
    haystack = f"{row.get('keywords') or ''} {row.get('description') or ''}".lower()
    hits = sum(1 for t in set(terms) if t and t in haystack)
    denom = min(_LEXICAL_TARGET_HITS, len(set(terms)))
    return min(1.0, hits / denom) if denom else 0.0


def _suggestion(relevance: float | None) -> str:
    """Derive the UI suggestion marker from a blended relevance score (0–1)."""
    if relevance is None:
        return "neutral"
    if relevance >= _SUGGEST_THRESHOLD:
        return "suggested"
    if relevance >= _NEUTRAL_THRESHOLD:
        return "neutral"
    return "low"


def _fetch_project_events(project_id, wanted_paths: set) -> list[dict]:
    """Fetch a project's temporal events, restricted to a set of parent file paths.

    Reads every ``clip_events`` row for the project once and keeps only those whose parent
    clip survived the SQL hard filters (``wanted_paths``), so event-aware recall never
    surfaces a clip the structured filters already excluded. Empty list when the project
    has no events (→ retrieval degrades to pure clip-level, unchanged).
    """
    if not wanted_paths:
        return []
    with _engine.begin() as conn:
        rows = conn.execute(
            _sql("SELECT file_path, start_seconds, end_seconds, action, state_change, "
                 "keywords, subjects, embedding FROM clip_events WHERE project_id = :pid"),
            {"pid": project_id},
        )
        return [dict(r._mapping) for r in rows if r._mapping["file_path"] in wanted_paths]


def _score_events(rows: list[dict], terms: list[str],
                  query_vec) -> list[tuple[dict, float]]:
    """Score event rows against a query (vector cosine noisy-OR lexical), shared logic.

    Used both by ``search_events`` (Selection's moment primitive) and by
    ``hybrid_search``'s event-aware clip recall, so event scoring has ONE source of truth.
    Returns ``(row, relevance)`` pairs aligned to ``rows``.
    """
    sim: dict[int, float] = {}
    rows_with_vec = [r for r in rows if r.get("embedding")]
    if query_vec is not None and rows_with_vec:
        try:
            matrix = np.asarray(
                [json.loads(r["embedding"]) for r in rows_with_vec], dtype=float)
            sims = _cosine(query_vec, matrix)
            sim = {id(r): float(max(0.0, s)) for r, s in zip(rows_with_vec, sims)}
        except (ValueError, TypeError) as e:
            logger.error(f"Event cosine ranking failed, using lexical only: {e}")
    scored = []
    for r in rows:
        lex = _event_lexical_score(r, terms)
        vec = sim.get(id(r))
        rel = (1.0 - (1.0 - _calibrate_cosine(vec)) * (1.0 - lex)) if vec is not None else lex
        scored.append((r, rel))
    return scored


def _best_event_by_path(project_id, wanted_paths: set, terms: list[str],
                        query_vec) -> dict[str, dict]:
    """Best-matching temporal event per parent clip, for event-aware clip recall.

    Aggregates each clip's events to the single strongest match, so a clip whose OVERALL
    tags miss the query still surfaces when a MOMENT inside it hits — while the retrieval
    unit stays the clip. Returns ``{file_path: {start, end, action, relevance}}``.
    """
    rows = _fetch_project_events(project_id, wanted_paths)
    if not rows:
        return {}
    best: dict[str, dict] = {}
    for r, rel in _score_events(rows, terms, query_vec):
        fp = r["file_path"]
        cur = best.get(fp)
        if cur is None or rel > cur["relevance"]:
            best[fp] = {"file_path": fp, "start_seconds": r.get("start_seconds"),
                        "end_seconds": r.get("end_seconds"),
                        "action": r.get("action"), "relevance": rel}
    return best


def hybrid_search(project_id, *, keywords: str = None, shot_type: str = None,
                  orientation: str = None, people: int = None,
                  min_duration: float = None, max_duration: float = None,
                  top_k: int = 20) -> list[dict]:
    """Unified hybrid retrieval: SQL hard filters + EVENT-AWARE semantic/lexical recall.

    The retrieval unit is always the CLIP (Search browses footage; Selection cuts moments
    — a hard responsibility boundary). But recall is event-aware: a clip's relevance is
    the reinforcing (noisy-OR) combination of its OWN clip-level match and the best match
    among its temporal events, so a clip whose overall tags miss the query still surfaces
    when a MOMENT inside it hits (e.g. "celebration" finds a clip whose only celebratory
    beat is one event). The matched moment is attached as ``matched_event`` for preview
    context only — it never turns the result into a moment.

    Args:
        project_id: Project whose catalogue to search.
        keywords: Free-text semantic query (comma- or space-separated terms). Drives
            the vector / lexical recall layer. Omit for a pure structured filter.
        shot_type: Filter by shot type (substring match).
        orientation: 'portrait' | 'landscape' | 'square' (synonyms vertical/horizontal).
        people: Minimum number of people visible.
        min_duration / max_duration: Duration window in seconds.
        top_k: Maximum number of candidates to return.

    Returns:
        A list of CLIP candidate dicts, each carrying real catalogue metadata plus a
        ``relevance`` (0–1 or None when no query was given), a ``group_size`` label, a
        ``suggestion`` marker ('suggested' | 'neutral' | 'low'), and (when a moment drove
        the match) a ``matched_event``. Ranked by relevance when a query is present,
        otherwise by file path. Never fabricates rows.
    """
    # A format word ("horizontal") left inside the free-text query is a FILTER, not a
    # thing to score — hoist it into `orientation` so it narrows rows instead of giving
    # every clip a misleading relevance %. No-op once `orientation` is explicitly set.
    keywords, orientation = hoist_orientation(keywords, orientation)

    rows = _fetch_filtered(project_id, shot_type, orientation, people,
                           min_duration, max_duration)
    if not rows:
        return []

    query = (keywords or "").strip()
    scored: list[tuple[dict, float | None]] = []

    if query:
        terms = [t.strip().lower() for t in query.replace(",", " ").split() if t.strip()]

        # Embed the query ONCE and reuse it for both the clip and event vector layers
        # (returns None without an API key → both layers fall back to lexical).
        query_vecs = embed_texts([query])
        q = np.asarray(query_vecs[0], dtype=float) if query_vecs else None

        # Clip vector layer: cosine against rows that carry a clip-level embedding.
        sim_by_id: dict[int, float] = {}
        rows_with_vec = [r for r in rows if r.get("embedding")]
        if q is not None and rows_with_vec:
            try:
                matrix = np.asarray(
                    [json.loads(r["embedding"]) for r in rows_with_vec], dtype=float
                )
                sims = _cosine(q, matrix)
                sim_by_id = {id(r): float(max(0.0, s))
                             for r, s in zip(rows_with_vec, sims)}
            except (ValueError, TypeError) as e:
                logger.error(f"Cosine ranking failed, using lexical only: {e}")

        # Event-aware layer: best temporal-event match per clip (empty when no events).
        best_event = _best_event_by_path(
            project_id, {r["file_path"] for r in rows}, terms, q)

        # Combine clip + event signals with a reinforcing (noisy-OR) rule: a strong hit in
        # the clip's own tags OR in one of its moments each push relevance toward 1.0, and
        # neither dilutes the other.
        for r in rows:
            lex = _lexical_score(r, terms)
            vec = sim_by_id.get(id(r))
            rel_clip = (1.0 - (1.0 - _calibrate_cosine(vec)) * (1.0 - lex)
                        ) if vec is not None else lex
            ev = best_event.get(r["file_path"])
            if ev is not None:
                rel = 1.0 - (1.0 - rel_clip) * (1.0 - ev["relevance"])
                r["matched_event"] = {
                    "start_seconds": ev["start_seconds"], "end_seconds": ev["end_seconds"],
                    "action": ev["action"], "relevance": round(ev["relevance"], 4),
                }
            else:
                rel = rel_clip
            scored.append((r, rel))

        scored.sort(key=lambda pair: (pair[1] if pair[1] is not None else 0.0),
                    reverse=True)
    else:
        # No free-text query — pure structured filter / list-all. Keep SQL order.
        scored = [(r, None) for r in rows]

    candidates = []
    for r, rel in scored[:top_k]:
        r.pop("embedding", None)
        r["relevance"] = round(rel, 4) if rel is not None else None
        r["group_size"] = group_size(r.get("people_count"))
        r["suggestion"] = _suggestion(rel)
        candidates.append(r)
    return candidates


# ── Event-level (temporal) retrieval — "find the MOMENT", not just the clip ─────

# Columns pulled for every event candidate: the event's own timing/action fields plus a
# little parent-clip context (path, shot type, orientation) for the UI/Selection. The
# event embedding is fetched for cosine ranking then stripped from the returned dict.
_EVENT_SELECT = (
    "e.event_id, e.shot_id, e.file_path, e.event_order, "
    "e.start_seconds, e.end_seconds, e.duration_seconds, "
    "e.action, e.state_change, e.subjects, e.keywords, e.embedding, "
    "s.shot_type AS shot_type, s.orientation AS orientation, "
    "s.duration_seconds AS clip_duration"
)


def _event_lexical_score(row: dict, terms: list[str]) -> float:
    """Saturating lexical overlap between query terms and an event's action/keywords."""
    if not terms:
        return 0.0
    haystack = (f"{row.get('action') or ''} {row.get('keywords') or ''} "
                f"{row.get('state_change') or ''} {row.get('subjects') or ''}").lower()
    hits = sum(1 for t in set(terms) if t and t in haystack)
    denom = min(_LEXICAL_TARGET_HITS, len(set(terms)))
    return min(1.0, hits / denom) if denom else 0.0


def search_events(project_id, *, keywords: str = None, shot_type: str = None,
                  orientation: str = None, top_k: int = 20) -> list[dict]:
    """Retrieve temporal EVENTS (moments) matching a query, scoped to a project.

    The moment-level primitive backing the SELECTION stage (Selection queries a clip's
    events to cut to a precise moment). It is NOT a Search-stage user mode — Search always
    returns clips (see :func:`hybrid_search`, which uses events only to improve clip
    recall). Instead of whole clips, this ranks individual ``clip_events`` and returns the
    parent clip PLUS the exact in/out timecodes of each event, which Selection/Delivery
    trim to verbatim.

    Ranking mirrors ``hybrid_search`` (shared ``_score_events``): a query embedding
    cosine-matched against per-event embeddings, combined (noisy-OR) with lexical overlap
    on the event's action/keywords. Falls back to lexical alone when no embeddings/API key.

    Args:
        project_id: Project whose events to search.
        keywords: Free-text moment query. Omit for an unranked chronological listing.
        shot_type / orientation: Optional parent-clip filters.
        top_k: Maximum number of events to return.

    Returns:
        A list of event candidate dicts (event timing + action + parent-clip context +
        ``relevance`` + ``suggestion``), ranked by relevance when a query is given, else
        by (file_path, event_order). Never fabricates — no events yields an empty list.
    """
    keywords, orientation = hoist_orientation(keywords, orientation)

    clauses = ["e.project_id = :pid"]
    params: dict = {"pid": project_id}
    orient = _normalise_orientation(orientation)
    if orient:
        clauses.append("s.orientation = :orient")
        params["orient"] = orient
    if shot_type:
        clauses.append("s.shot_type LIKE :stype")
        params["stype"] = f"%{shot_type.strip().lower()}%"
    where = " AND ".join(clauses)
    query_sql = (f"SELECT {_EVENT_SELECT} FROM clip_events e "
                 f"JOIN shots s ON s.shot_id = e.shot_id "
                 f"WHERE {where} ORDER BY e.file_path, e.event_order")

    with _engine.begin() as conn:
        rows = [dict(r._mapping) for r in conn.execute(_sql(query_sql), params)]
    if not rows:
        return []

    query = (keywords or "").strip()
    if query:
        terms = [t.strip().lower() for t in query.replace(",", " ").split() if t.strip()]
        query_vecs = embed_texts([query])
        q = np.asarray(query_vecs[0], dtype=float) if query_vecs else None
        scored = _score_events(rows, terms, q)
        scored.sort(key=lambda pair: pair[1], reverse=True)
    else:
        scored = [(r, None) for r in rows]

    candidates = []
    for r, rel in scored[:top_k]:
        r.pop("embedding", None)
        subj = r.get("subjects")
        if isinstance(subj, str) and subj:
            try:
                r["subjects"] = json.loads(subj)
            except (ValueError, TypeError):
                pass
        r["relevance"] = round(rel, 4) if rel is not None else None
        r["suggestion"] = _suggestion(rel)
        candidates.append(r)
    return candidates
