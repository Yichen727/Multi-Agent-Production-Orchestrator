"""Project-scoped catalogue identifier resolution (shared by Selection & Delivery).

Resolves a ``shot_id`` or a file-path / file-name identifier to EXACTLY ONE real
catalogued clip WITHIN A GIVEN PROJECT. Two guarantees the pipeline depends on:

  * PROJECT ISOLATION (audit C-03 / C-04) — every query is filtered by ``project_id``,
    so an identifier can never resolve to another project's media. Numeric shot IDs and
    file names are both scoped; project membership is never inferred from the identifier.
  * NO SILENT AMBIGUITY (audit H-01) — a file name that matches more than one clip raises
    :class:`AmbiguousIdentifier` (listing the candidate shot IDs) instead of silently
    taking the first row (the old ``LIKE '%tok%' ... ORDER BY shot_id LIMIT 1`` behaviour
    that quietly confused two ``take01.mov`` files). Callers surface the conflict so the
    editor disambiguates with a shot_id.

Matching order for a non-numeric token is most-specific-first: exact full path → exact
file name (basename, either path separator) → loose substring. Never fabricates: an
identifier that matches nothing resolves to ``None``. All SQL uses bound parameters, and
LIKE metacharacters in the token are escaped, so a token can never alter the query shape.
"""

from sqlalchemy import text as _sql

from app.services.database_service import _engine

# Metadata pulled for a resolved clip — the superset both Selection and Delivery need.
_DEFAULT_COLUMNS = (
    "shot_id, file_path, shot_type, duration_seconds, orientation, fps, "
    "width, height, codec, has_audio, keywords, description, "
    "people_count, mood, "
    "audio_channels, audio_sample_rate, audio_bit_depth"
)


class AmbiguousIdentifier(Exception):
    """A file-name identifier matched more than one clip in the project.

    Carries the offending ``token`` and the list of candidate ``(shot_id, file_path)``
    tuples so the caller can tell the editor exactly which clips collided.
    """

    def __init__(self, token: str, candidates: list[tuple]):
        self.token = token
        self.candidates = candidates
        listing = ", ".join(f"#{sid} {fp}" for sid, fp in candidates)
        super().__init__(
            f"'{token}' matches {len(candidates)} clips in this project ({listing}). "
            "Use the shot_id to pick one."
        )


def _escape_like(term: str) -> str:
    """Escape LIKE metacharacters so a token can't act as a wildcard (ESCAPE '\\')."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _basename(path) -> str:
    """Final path component, treating both '/' and '\\' as separators."""
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def _match_by_path(conn, project_id, token: str, columns: str) -> list:
    base = f"SELECT {columns} FROM shots WHERE project_id = :pid AND "

    # 1) exact full path (no wildcards — cannot be ambiguous by definition, but a
    #    catalogue without the UNIQUE constraint could still hold dupes, so return all).
    rows = conn.execute(_sql(base + "file_path = :t"),
                        {"pid": project_id, "t": token}).fetchall()
    if rows:
        return rows

    # 2) pull a loose (escaped) substring superset, then keep only exact basename matches.
    esc = _escape_like(token)
    superset = conn.execute(
        _sql(base + "file_path LIKE :c ESCAPE '\\'"),
        {"pid": project_id, "c": f"%{esc}%"},
    ).fetchall()
    exact_base = [r for r in superset if _basename(r._mapping["file_path"]) == token]
    if exact_base:
        return exact_base

    # 3) fall back to the loose substring superset.
    return superset


def resolve_one(project_id, token, *, columns: str = _DEFAULT_COLUMNS) -> dict | None:
    """Resolve a single identifier to one catalogue row within ``project_id``.

    ``token`` is a numeric shot_id or a file path / name. Returns the row dict, or
    ``None`` when nothing matches. Raises :class:`AmbiguousIdentifier` when a file-name
    token matches more than one clip (never silently picks the first).
    """
    token = (str(token) if token is not None else "").strip()
    if not token:
        return None

    with _engine.begin() as conn:
        if token.isdigit():
            rows = conn.execute(
                _sql(f"SELECT {columns} FROM shots "
                     "WHERE project_id = :pid AND shot_id = :sid"),
                {"pid": project_id, "sid": int(token)},
            ).fetchall()
        else:
            rows = _match_by_path(conn, project_id, token, columns)

    if not rows:
        return None
    if len(rows) > 1:
        raise AmbiguousIdentifier(
            token, [(r._mapping["shot_id"], r._mapping["file_path"]) for r in rows]
        )
    return dict(rows[0]._mapping)


def resolve_ordered(project_id, tokens, *, columns: str = _DEFAULT_COLUMNS):
    """Resolve tokens IN ORDER within a project, preserving order and repeats.

    Returns ``(rows, problems)``:
      - ``rows``   — resolved row dicts, each tagged with ``_identifier`` (the token that
        matched it); order preserved, repeats kept.
      - ``problems`` — ``(token, reason)`` tuples for tokens that did not resolve to
        exactly one clip (unresolved or ambiguous).

    A caller that must not fabricate should REFUSE when ``problems`` is non-empty rather
    than proceed with a partial / mis-ordered list.
    """
    rows: list[dict] = []
    problems: list[tuple[str, str]] = []
    for tok in tokens:
        try:
            row = resolve_one(project_id, tok, columns=columns)
        except AmbiguousIdentifier as e:
            problems.append((tok, str(e)))
            continue
        if row is None:
            problems.append((tok, "no catalogued clip matched in this project"))
        else:
            row["_identifier"] = tok
            rows.append(row)
    return rows, problems
