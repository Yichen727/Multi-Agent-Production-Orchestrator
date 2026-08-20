"""Retrieval service — unified HYBRID search over the production catalogue.

This is the single semantic-recall entry point for the MAPO Search stage. It replaces
the old set of fragmented per-attribute SQL tools with one function that combines:

    1. SQL HARD FILTERS  — orientation, duration, people, shot type. Cheaply
       narrows the catalogue to rows that satisfy the structured constraints.
    2. VECTOR SEMANTIC RECALL — when a free-text ``keywords`` query is given and the
       filtered rows carry embeddings (written at ingest), the query is embedded and
       ranked by cosine similarity against them.
    3. LEXICAL OVERLAP — WHOLE-WORD (never substring) matching of the query terms
       against a row's keywords/description, tiered by whether a term is the user's own
       or came from synonym expansion. It also carries retrieval alone when embeddings
       are unavailable (no API key, or rows ingested before the embedding step).

Relevance is tiered by how trustworthy the evidence is: ffprobe-MEASURED constraints
(orientation, duration) can report a full 1.0, while anything derived from the vision
model's tags caps below that — see the evidence-tier constants below.

Both the Search Agent's ``search_catalogue`` tool and the Streamlit curation UI call
``hybrid_search`` directly, so retrieval behaves identically whether driven by the LLM
or by the user. Nothing here fabricates data: it only ranks rows that already exist in
the catalogue.
"""

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


# Columns pulled for every candidate — the metadata the UI preview and the Selection
# Agent need. ``embedding`` is fetched too but stripped from the returned dict.
_SELECT_COLUMNS = (
    "shot_id, file_path, shot_type, duration_seconds, "
    "orientation, width, height, fps, keywords, description, "
    "people_count, mood, embedding"
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

# ── Evidence tiers ─────────────────────────────────────────────────────────────
# A free-text query carries TWO tiers of evidence (see ``QueryTerms``):
#   CORE    — what the user literally asked for, translated to English.
#   RELATED — synonyms/variants added by query expansion.
# Expansion is a RECALL device: it must be able to SURFACE a clip, but it must never be
# able to certify one, because a synonym set is looser than the user's own word. So a
# core-term hit scores far above a synonym hit, and synonym-only evidence is capped.
_CORE_LEXICAL_SCORE = 0.9        # every core term matched = STRONG evidence, not proof
_RELATED_LEXICAL_CEILING = 0.6   # related-term-only text evidence caps here
_RELATED_VEC_DAMP = 0.9          # the expanded query's cosine is worth slightly less

# Ceiling on any relevance derived from INFERRED metadata (keywords, description,
# shot_type, people_count — and the embedding, which is computed FROM that same text).
# Those signals are not independent witnesses: one vision mis-tag (a microphone tagged
# `phone`) inflates the lexical AND the vector layer at once, so no amount of internal
# agreement can certify a content match. Content relevance therefore measures the strength
# of the CATALOGUE's evidence, never ground truth — the editor's eyes are the last check.
#
# MEASURED constraints are exempt: orientation / duration come from ffprobe, not from a
# model, so a row that passes them matches EXACTLY and is reported as a full 1.0.
_MAX_INFERRED_RELEVANCE = 1.0


def _finalise_inferred(relevance: float) -> float:
    """Clamp a content-derived relevance into the reportable band [0, 0.95]."""
    return max(0.0, min(_MAX_INFERRED_RELEVANCE, relevance))


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
# The reply is deliberately SPLIT into the two evidence tiers (CORE / RELATED) so the
# scorer can tell the user's own words apart from the expansion, and the expansion itself
# is CONSTRAINED: a concrete entity must never be broadened into its category or its
# surroundings, because "phone → device, technology, communication" is exactly what drags
# an unrelated microphone clip into a phone search.
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
    """Read the expander's two-line CORE:/RELATED: reply, tolerating a stray format."""
    core = related = ""
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("core:"):
            core = line.split(":", 1)[1].strip()
        elif low.startswith("related:"):
            related = line.split(":", 1)[1].strip()
    if not core and not related:
        # The model ignored the format. Treat everything it said as RELATED — unlabelled
        # output is never promoted to core (strong) evidence.
        related = " ".join(text.split())
    return core, related


def expand_query_terms(raw_query: str) -> tuple[str, str]:
    """Translate + expand a natural-language query into ``(core, related)`` term strings.

    ``core`` holds the user's OWN terms in English — the strong evidence a strict match is
    measured against; ``related`` holds the constrained synonym expansion, which serves
    recall only. Both are comma-separated. Falls back to ``(raw_query, "")`` when the query
    is empty or the LLM is unavailable (no API key / error), so search still runs — it just
    does not benefit from expansion.
    """
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
    """The full retrieval term set (core + related) for a query, comma-separated."""
    core, related = expand_query_terms(raw_query)
    return ", ".join(t for t in (core, related) if t) or (raw_query or "").strip()


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


# A row that matches this many distinct RELATED terms saturates that tier. Using a small
# saturating target (rather than a fraction of ALL terms) means a GENEROUS synonym
# expansion no longer dilutes the score — matching ~3 terms saturates whether the query
# carried 3 synonyms or 30.
_LEXICAL_TARGET_HITS = 3

_WORD_RE = re.compile(r"[a-z0-9]+")


class QueryTerms(NamedTuple):
    """A parsed query split into its two evidence tiers (see the constants at the top)."""

    core: tuple[str, ...]      # the user's own terms, in English
    related: tuple[str, ...]   # the constrained synonym expansion

    def __bool__(self) -> bool:
        return bool(self.core or self.related)


def _singular(word: str) -> str:
    """Crude singulariser so a 'phones' query still matches a 'phone' tag, and vice versa.

    Applied to BOTH sides of every comparison, so even an imperfect stem matches
    consistently; it is deliberately conservative to avoid merging unrelated words.
    """
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
    """Split text into normalised WORD tokens — the unit every lexical match is made on."""
    return [_singular(w) for w in _WORD_RE.findall((text or "").lower())]


def _term_matches(term: str, tokens: list[str]) -> bool:
    """True when ``term`` occurs in ``tokens`` as a whole word / consecutive phrase.

    Whole-word matching is what stops a query for "phone" scoring a clip tagged
    "microphone" as a lexical hit: 'phone' IS inside the string "microphone", but it is
    not one of its words. Substring containment is never a match here.
    """
    parts = _tokenise(term)
    if not parts:
        return False
    if len(parts) == 1:
        return parts[0] in tokens
    n = len(parts)
    return any(tokens[i:i + n] == parts for i in range(len(tokens) - n + 1))


def _split_terms(text: str | None) -> list[str]:
    """Parse a term string into deduplicated terms, dropping filler words.

    Comma-separated input keeps each part as a PHRASE ("mobile phone"); a plain
    whitespace query is split into individual words.
    """
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
    """Split a query into CORE (the user's own words) and RELATED (the expansion).

    ``keywords`` is the full term set; ``core_keywords`` marks which of those terms came
    from the user. A term in both counts as core only, so nothing is scored twice. When
    the caller cannot identify the core, every term is treated as related — and the
    related-only CAP is then not applied (see ``_text_lexical_score``), so such callers
    keep exactly the previous scoring behaviour.
    """
    core = _split_terms(core_keywords)
    related = [t for t in _split_terms(keywords) if t not in core]
    return QueryTerms(tuple(core), tuple(related))


def _text_lexical_score(haystack: str, terms: QueryTerms) -> float:
    """Two-tier lexical evidence for one text blob (0–1).

    CORE coverage — the user's own words present as WHOLE words — is strong evidence and
    scores up to ``_CORE_LEXICAL_SCORE``; RELATED terms saturate at the lower
    ``_RELATED_LEXICAL_CEILING``. The tiers combine with ``max``, never a sum, so piling
    on synonyms can never manufacture strong evidence.
    """
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
    """Lexical evidence from a CLIP's own tags."""
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


def _embed_query(core_text: str, full_text: str):
    """Embed the CORE query and the FULL expanded query in one batched call.

    Keeping them apart matters twice over: the expanded string is an AVERAGE of many
    synonyms, which both dilutes a true match on the user's own word and drags the vector
    toward whatever the expansion added. Scoring then takes the best of the two (the
    expansion slightly damped), so expansion can still rescue a synonym-only match without
    getting to decide a strict one. Returns ``(core_vec | None, full_vec | None)``.
    """
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
    """CALIBRATED vector relevance per row (keyed by ``id(row)``), best of both queries.

    Rows carrying no embedding are absent from the result, so the caller falls back to
    lexical evidence for them rather than scoring them as a 0.0 mismatch.
    """
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
    """Score event rows against a query (vector cosine noisy-OR lexical), shared logic.

    Used both by ``search_events`` (Selection's moment primitive) and by
    ``hybrid_search``'s event-aware clip recall, so event scoring has ONE source of truth.
    Returns ``(row, relevance)`` pairs aligned to ``rows``.
    """
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
    """Best-matching temporal event per parent clip, for event-aware clip recall.

    Aggregates each clip's events to the single strongest match, so a clip whose OVERALL
    tags miss the query still surfaces when a MOMENT inside it hits — while the retrieval
    unit stays the clip. Returns ``{file_path: {start, end, action, relevance}}``.
    """
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
    """Unified hybrid retrieval: SQL hard filters + EVENT-AWARE semantic/lexical recall.

    The retrieval unit is always the CLIP (Search browses footage; Selection cuts moments
    — a hard responsibility boundary). But recall is event-aware: a clip's relevance is
    the reinforcing (noisy-OR) combination of its OWN clip-level match and the best match
    among its temporal events, so a clip whose overall tags miss the query still surfaces
    when a MOMENT inside it hits (e.g. "celebration" finds a clip whose only celebratory
    beat is one event). The matched moment is attached as ``matched_event`` for preview
    context only — it never turns the result into a moment.

    Relevance is TIERED by how trustworthy the evidence is. ffprobe-MEASURED constraints
    (orientation, duration) are exact, so a pure structured filter reports 1.0 — every
    surviving row genuinely matches. Everything content-related is INFERRED by the vision
    model and caps at 0.95, with the user's own words (``core_keywords``) scoring far above
    the synonym expansion and whole-word matching only, so "phone" never scores a clip
    tagged "microphone".

    Args:
        project_id: Project whose catalogue to search.
        keywords: Free-text semantic query (comma- or space-separated terms) — the FULL
            term set including synonym expansion. Drives the vector / lexical recall
            layer. Omit for a pure structured filter.
        core_keywords: The subset of ``keywords`` the USER actually asked for (their own
            entities/actions, translated to English, no synonyms). Matching these is
            strong evidence; matching only expanded synonyms is capped. Omit when the
            caller cannot tell the two apart — scoring then behaves as it did before.
        shot_type: Filter by shot type (substring match).
        orientation: 'portrait' | 'landscape' | 'square' (synonyms vertical/horizontal).
        people: Minimum number of people visible.
        min_duration / max_duration: Duration window in seconds.
        top_k: Maximum number of candidates to return (default 100). This is only the
            FINAL slice — filtering, clip-vector scoring and event scoring already run
            over every matching row in the project, so raising it costs no extra DB or
            embedding work; it only widens what the UI/agent has to render. Kept
            generous so a large catalogue is not silently truncated mid-relevance —
            weak matches simply land in the collapsed 'low' tier.

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
    orient = _normalise_orientation(orientation)

    rows = _fetch_filtered(project_id, shot_type, orientation, people,
                           min_duration, max_duration)
    if not rows:
        return []

    query = (keywords or "").strip()
    core = (core_keywords or "").strip()
    # (row, relevance, tiebreak) — the tiebreak is the raw semantic signal.
    scored: list[tuple[dict, float | None, float]] = []

    if query or core:
        terms = parse_terms(query, core)

        # Embed the query ONCE (core + expanded, batched) and reuse both vectors for the
        # clip and event layers (None without an API key → both fall back to lexical).
        q_core, q_full = _embed_query(core, query or core)

        # Clip vector layer: cosine against rows that carry a clip-level embedding.
        sim_by_id = _vector_scores(rows, q_core, q_full)

        # Event-aware layer: best temporal-event match per clip (empty when no events).
        best_event = _best_event_by_path(
            project_id, {r["file_path"] for r in rows}, terms, q_core, q_full)

        # Combine clip + event signals with a reinforcing (noisy-OR) rule: a strong hit in
        # the clip's own tags OR in one of its moments each push relevance up, and neither
        # dilutes the other. The result is INFERRED evidence, so it caps below a full match.
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

        # Sort on relevance, breaking ties on the raw semantic signal: several clips can
        # legitimately share a capped score (e.g. all strict core matches), and falling
        # back to alphabetical file order there would be arbitrary.
        scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
    else:
        # No free-text query — the result reports the CONSTRAINTS, not content. Measured
        # constraints (orientation/duration, from ffprobe) are exact, so every surviving
        # row is a genuine 100% match. A vision-INFERRED constraint (shot_type, people) is
        # only as good as the tag, so it caps. No constraints at all is a plain listing,
        # which has no relevance to report (None) — keep SQL order.
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


def _event_lexical_score(row: dict, terms: QueryTerms) -> float:
    """Two-tier lexical evidence from an EVENT's action/keywords (same rules as clips)."""
    return _text_lexical_score(
        f"{row.get('action') or ''} {row.get('keywords') or ''} "
        f"{row.get('state_change') or ''} {row.get('subjects') or ''}", terms)


def search_events(project_id, *, keywords: str = None, core_keywords: str = None,
                  shot_type: str = None, orientation: str = None,
                  top_k: int = 20) -> list[dict]:
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
        keywords: Free-text moment query (full term set incl. expansion). Omit for an
            unranked chronological listing.
        core_keywords: The subset of ``keywords`` the user actually asked for — matching
            these is strong evidence, matching only synonyms is capped (see
            :func:`hybrid_search`).
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
