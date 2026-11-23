"""Hybrid retrieval service for the MAPO production catalogue."""

import json
import re
from typing import NamedTuple

import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage
from sqlalchemy import text as _sql

from app.services.database_service import _engine
from app.services.openai_service import embed_texts, llm_fast
from app.utils.logger import get_logger

logger = get_logger("retrieval_service")


# Metadata returned for each candidate. Embeddings are used internally only.
_SELECT_COLUMNS = (
    "shot_id, file_path, shot_type, duration_seconds, "
    "orientation, width, height, fps, keywords, description, "
    "people_count, mood, embedding"
)

# Relevance thresholds.
_SUGGEST_THRESHOLD = 0.55
_NEUTRAL_THRESHOLD = 0.30

# Raw cosine calibration range.
_COSINE_FLOOR = 0.18
_COSINE_CEIL = 0.55

# Core terms are stronger evidence than expanded terms.
_CORE_LEXICAL_SCORE = 0.9        
_RELATED_LEXICAL_CEILING = 0.6   
_RELATED_VEC_DAMP = 0.9          

# Maximum relevance from inferred catalogue metadata.
_MAX_INFERRED_RELEVANCE = 1.0


def _finalise_inferred(relevance: float) -> float:
    """Clamp inferred relevance to the reportable range [0, 0.95]."""
    return max(0.0, min(_MAX_INFERRED_RELEVANCE, relevance))


def _calibrate_cosine(cos: float) -> float:
    """Map raw cosine similarity to calibrated [0, 1] relevance."""
    if cos <= _COSINE_FLOOR:
        return 0.0
    if cos >= _COSINE_CEIL:
        return 1.0
    return (cos - _COSINE_FLOOR) / (_COSINE_CEIL - _COSINE_FLOOR)


# Query translation and constrained synonym expansion.
_EXPAND_INSTRUCTION = (
    "You turn a video-search query into English retrieval terms.\n"
    "Answer in EXACTLY two lines, nothing else:\n"
    "CORE: <the things the user literally asked for, translated to English — the "
    "entities, subjects and actions themselves. Lowercase, comma-separated, NO synonyms.>\n"
    "RELATED: <up to 8 lowercase comma-separated alternative terms the catalogue might "
    "use INSTEAD of a core term. May be empty.>\n"
    "\n"
    "Rules for RELATED:\n"
    "- Every related term must be able to REPLACE a core term: a synonym, an alternative "
    "name, a spelling / singular-plural variant, or a specific type of it. "
    "phone -> mobile phone, smartphone, cellphone, telephone.\n"
    "- NEVER broaden a concrete object, person or place into its category, its context, "
    "or things that merely appear near it. phone -> device, gadget, electronics, "
    "technology, communication, call, screen is WRONG.\n"
    "- NEVER emit a term that merely CONTAINS a core word with a different meaning. "
    "phone -> microphone, headphone, earphone is WRONG.\n"
    "- ABSTRACT concepts (mood, atmosphere, weather, scenery, activity) MAY be broadened "
    "to closely related terms: celebration -> party, cheering, applause, festival.\n"
    "- Drop filler words (please, find, show, video, clip, footage)."
)


def _parse_expansion(text: str) -> tuple[str, str]:
    """Parse the expander's CORE and RELATED terms."""
    core = related = ""
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("core:"):
            core = line.split(":", 1)[1].strip()
        elif low.startswith("related:"):
            related = line.split(":", 1)[1].strip()
    if not core and not related:
        related = " ".join(text.split())
    return core, related


def expand_query_terms(raw_query: str) -> tuple[str, str]:
    """Translate and expand a query into (core, related) terms."""
    q = (raw_query or "").strip()
    if not q:
        return "", ""
    try:
        resp = llm_fast.invoke([SystemMessage(_EXPAND_INSTRUCTION), HumanMessage(q)])
        core, related = _parse_expansion(resp.content or "")
        return (core or q), related
    except Exception as e:  # no key / network / model — degrade to the raw query
        logger.warning(f"Query expansion unavailable, using raw query: {e}")
        return q, ""


def expand_query(raw_query: str) -> str:
    """Return the combined core and related retrieval terms."""
    core, related = expand_query_terms(raw_query)
    return ", ".join(t for t in (core, related) if t) or (raw_query or "").strip()


def group_size(people_count) -> str:
    """Map people count to a coarse editing label."""
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
    """Calculate row-wise cosine similarity."""
    if matrix.size == 0:
        return np.array([])
    q_norm = np.linalg.norm(query_vec)
    if q_norm == 0:
        return np.zeros(matrix.shape[0])
    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms[row_norms == 0] = 1e-9
    return (matrix @ query_vec) / (row_norms * q_norm)


def _normalise_orientation(orientation: str | None) -> str | None:
    """Normalise orientation aliases."""
    if not orientation:
        return None
    norm = orientation.strip().lower()
    norm = {"vertical": "portrait", "horizontal": "landscape"}.get(norm, norm)
    return norm if norm in ("portrait", "landscape", "square") else None


# Orientation terms that can be extracted from free-text queries.
_UNAMBIGUOUS_ORIENT = {
    "horizontal": "landscape",
    "vertical": "portrait",
    "widescreen": "landscape",
}

_AMBIGUOUS_ORIENT = {
    "landscape": "landscape",
    "portrait": "portrait",
    "square": "square",
}

_FILLER_WORDS = {
    "shot", "shots", "clip", "clips", "video", "videos", "footage", "scene", "scenes",
    "all", "any", "every", "find", "show", "get", "give", "me", "the", "a", "an", "of",
    "with", "please", "some",
}


def hoist_orientation(keywords: str | None,
                      orientation: str | None) -> tuple[str | None, str | None]:
    """Extract orientation terms from free-text into the structured filter."""
    if not keywords or orientation:
        return keywords, orientation

    terms = [t.strip() for t in keywords.replace(",", " ").split() if t.strip()]
    detected = None          
    ambiguous = None         
    content: list[str] = []  
    for t in terms:
        low = t.lower()
        if low in _UNAMBIGUOUS_ORIENT:
            detected = _UNAMBIGUOUS_ORIENT[low]          
        elif low in _AMBIGUOUS_ORIENT:
            ambiguous = ambiguous or _AMBIGUOUS_ORIENT[low]
            content.append(t)                            
        elif low not in _FILLER_WORDS:
            content.append(t)

    if detected:
        content = [t for t in content if t.lower() not in _AMBIGUOUS_ORIENT]
        return (" ".join(content) or None), detected

    non_ambiguous = [t for t in content if t.lower() not in _AMBIGUOUS_ORIENT]
    if ambiguous and not non_ambiguous:
        return None, ambiguous
    return (" ".join(content) or None), orientation


def _fetch_filtered(project_id, shot_type, orientation, people,
                    min_duration, max_duration) -> list[dict]:
    """Apply SQL hard filters and return matching catalogue rows."""
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


# Number of related-term hits needed to reach the tier ceiling.
_LEXICAL_TARGET_HITS = 3

_WORD_RE = re.compile(r"[a-z0-9]+")


class QueryTerms(NamedTuple):
    """Core and expanded query terms."""

    core: tuple[str, ...]     
    related: tuple[str, ...]  

    def __bool__(self) -> bool:
        return bool(self.core or self.related)


def _singular(word: str) -> str:
    """Apply simple singularisation for lexical matching."""
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("shes", "ches", "xes", "zes", "ses")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def _tokenise(text: str) -> list[str]:
    """Convert text into normalised word tokens."""
    return [_singular(w) for w in _WORD_RE.findall((text or "").lower())]


def _term_matches(term: str, tokens: list[str]) -> bool:
    """Match a term as a whole word or consecutive phrase."""
    parts = _tokenise(term)
    if not parts:
        return False
    if len(parts) == 1:
        return parts[0] in tokens
    n = len(parts)
    return any(tokens[i:i + n] == parts for i in range(len(tokens) - n + 1))


def _split_terms(text: str | None) -> list[str]:
    """Parse and deduplicate query terms."""
    if not text or not text.strip():
        return []
    parts = text.split(",") if "," in text else text.split()
    out: list[str] = []
    for p in parts:
        term = " ".join(w for w in _WORD_RE.findall(p.lower()) if w not in _FILLER_WORDS)
        if term and term not in out:
            out.append(term)
    return out


def parse_terms(keywords: str | None, core_keywords: str | None = None) -> QueryTerms:
    """Split full query terms into core and related terms."""
    core = _split_terms(core_keywords)
    related = [t for t in _split_terms(keywords) if t not in core]
    return QueryTerms(tuple(core), tuple(related))


def _text_lexical_score(haystack: str, terms: QueryTerms) -> float:
    """Calculate two-tier whole-word lexical relevance."""
    if not terms:
        return 0.0
    tokens = _tokenise(haystack)
    if not tokens:
        return 0.0

    core_score = 0.0
    if terms.core:
        hits = sum(1 for t in terms.core if _term_matches(t, tokens))
        core_score = (hits / len(terms.core)) * _CORE_LEXICAL_SCORE

    related_score = 0.0
    if terms.related:
        hits = sum(1 for t in terms.related if _term_matches(t, tokens))
        denom = min(_LEXICAL_TARGET_HITS, len(terms.related))
        ceiling = _RELATED_LEXICAL_CEILING if terms.core else 1.0
        related_score = min(1.0, hits / denom) * ceiling

    return max(core_score, related_score)


def _lexical_score(row: dict, terms: QueryTerms) -> float:
    """Calculate lexical relevance from clip metadata."""
    return _text_lexical_score(
        f"{row.get('keywords') or ''} {row.get('description') or ''}", terms)


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
    """Fetch events belonging to the filtered clips."""
    if not wanted_paths:
        return []
    with _engine.begin() as conn:
        rows = conn.execute(
            _sql("SELECT file_path, start_seconds, end_seconds, action, state_change, "
                 "keywords, subjects, embedding FROM clip_events WHERE project_id = :pid"),
            {"pid": project_id},
        )
        return [dict(r._mapping) for r in rows if r._mapping["file_path"] in wanted_paths]


def _embed_query(core_text: str, full_text: str):
    """Batch-embed core and expanded query text."""
    texts, keys = [], []
    if core_text:
        texts.append(core_text)
        keys.append("core")
    if full_text and full_text != core_text:
        texts.append(full_text)
        keys.append("full")
    if not texts:
        return None, None
    vecs = embed_texts(texts)
    if not vecs or len(vecs) != len(texts):
        return None, None
    by_key = {k: np.asarray(v, dtype=float) for k, v in zip(keys, vecs)}
    return by_key.get("core"), by_key.get("full")


def _vector_scores(rows: list[dict], q_core, q_full) -> dict[int, float]:
    """Calculate calibrated vector relevance for catalogue rows."""
    rows_with_vec = [r for r in rows if r.get("embedding")]
    if not rows_with_vec or (q_core is None and q_full is None):
        return {}
    try:
        matrix = np.asarray(
            [json.loads(r["embedding"]) for r in rows_with_vec], dtype=float)
    except (ValueError, TypeError) as e:
        logger.error(f"Cosine ranking failed, using lexical only: {e}")
        return {}

    best: dict[int, float] = {}
    for vec, damp in ((q_core, 1.0), (q_full, _RELATED_VEC_DAMP)):
        if vec is None:
            continue
        for r, s in zip(rows_with_vec, _cosine(vec, matrix)):
            cal = _calibrate_cosine(max(0.0, float(s))) * damp
            if cal > best.get(id(r), -1.0):
                best[id(r)] = cal
    return best


def _score_events(rows: list[dict], terms: QueryTerms,
                  q_core, q_full) -> list[tuple[dict, float]]:
    """Score events using vector and lexical relevance."""
    sim = _vector_scores(rows, q_core, q_full)
    scored = []
    for r in rows:
        lex = _event_lexical_score(r, terms)
        vec = sim.get(id(r))
        rel = (1.0 - (1.0 - vec) * (1.0 - lex)) if vec is not None else lex
        scored.append((r, _finalise_inferred(rel)))
    return scored


def _best_event_by_path(project_id, wanted_paths: set, terms: QueryTerms,
                        q_core, q_full) -> dict[str, dict]:
    """Return the strongest matching event for each parent clip."""
    rows = _fetch_project_events(project_id, wanted_paths)
    if not rows:
        return {}
    best: dict[str, dict] = {}
    for r, rel in _score_events(rows, terms, q_core, q_full):
        fp = r["file_path"]
        cur = best.get(fp)
        if cur is None or rel > cur["relevance"]:
            best[fp] = {"file_path": fp, "start_seconds": r.get("start_seconds"),
                        "end_seconds": r.get("end_seconds"),
                        "action": r.get("action"), "relevance": rel}
    return best


def hybrid_search(project_id, *, keywords: str = None, core_keywords: str = None,
                  shot_type: str = None, orientation: str = None, people: int = None,
                  min_duration: float = None, max_duration: float = None,
                  top_k: int = 100) -> list[dict]:
    """Search clips using SQL filters plus hybrid semantic retrieval.

    Retrieval combines clip-level and temporal-event relevance. Events improve
    clip recall but remain separate from the clip retrieval unit.

    Returns catalogue metadata with relevance, group size, suggestion label,
    and an optional matched event.
    """
    keywords, orientation = hoist_orientation(keywords, orientation)
    orient = _normalise_orientation(orientation)

    rows = _fetch_filtered(project_id, shot_type, orientation, people,
                           min_duration, max_duration)
    if not rows:
        return []

    query = (keywords or "").strip()
    core = (core_keywords or "").strip()
    scored: list[tuple[dict, float | None, float]] = []

    if query or core:
        terms = parse_terms(query, core)

        # clip and event layers (None without an API key → both fall back to lexical).
        q_core, q_full = _embed_query(core, query or core)

        # Clip vector layer: cosine against rows that carry a clip-level embedding.
        sim_by_id = _vector_scores(rows, q_core, q_full)

        # Event-aware layer: best temporal-event match per clip (empty when no events).
        best_event = _best_event_by_path(
            project_id, {r["file_path"] for r in rows}, terms, q_core, q_full)

        for r in rows:
            lex = _lexical_score(r, terms)
            vec = sim_by_id.get(id(r))
            rel_clip = (1.0 - (1.0 - vec) * (1.0 - lex)) if vec is not None else lex
            ev = best_event.get(r["file_path"])
            if ev is not None:
                rel = 1.0 - (1.0 - rel_clip) * (1.0 - ev["relevance"])
                r["matched_event"] = {
                    "start_seconds": ev["start_seconds"], "end_seconds": ev["end_seconds"],
                    "action": ev["action"], "relevance": round(ev["relevance"], 4),
                }
            else:
                rel = rel_clip
            scored.append((r, _finalise_inferred(rel), vec or 0.0))

        scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
        
    else:
        # Structured-only searches report exactness based on filter source.
        inferred_filter = bool(shot_type) or people is not None
        measured_filter = bool(orient) or min_duration is not None or max_duration is not None
        if inferred_filter:
            exact = _MAX_INFERRED_RELEVANCE
        elif measured_filter:
            exact = 1.0
        else:
            exact = None
        scored = [(r, exact, 0.0) for r in rows]

    candidates = []
    for r, rel, _tiebreak in scored[:top_k]:
        r.pop("embedding", None)
        r["relevance"] = round(rel, 4) if rel is not None else None
        r["group_size"] = group_size(r.get("people_count"))
        r["suggestion"] = _suggestion(rel)
        candidates.append(r)
    return candidates


# Event retrieval returns precise temporal moments for Selection.
_EVENT_SELECT = (
    "e.event_id, e.shot_id, e.file_path, e.event_order, "
    "e.start_seconds, e.end_seconds, e.duration_seconds, "
    "e.action, e.state_change, e.subjects, e.keywords, e.embedding, "
    "s.shot_type AS shot_type, s.orientation AS orientation, "
    "s.duration_seconds AS clip_duration"
)


def _event_lexical_score(row: dict, terms: QueryTerms) -> float:
    """Calculate lexical relevance from event metadata."""
    return _text_lexical_score(
        f"{row.get('action') or ''} {row.get('keywords') or ''} "
        f"{row.get('state_change') or ''} {row.get('subjects') or ''}", terms)


def search_events(project_id, *, keywords: str = None, core_keywords: str = None,
                  shot_type: str = None, orientation: str = None,
                  top_k: int = 20) -> list[dict]:
    """Retrieve temporal events matching a query.

    Returns event timing, action, parent-clip metadata, relevance, and
    suggestion labels. Used by Selection to identify precise edit moments.
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
    core = (core_keywords or "").strip()
    if query or core:
        terms = parse_terms(query, core)
        q_core, q_full = _embed_query(core, query or core)
        scored = _score_events(rows, terms, q_core, q_full)
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
