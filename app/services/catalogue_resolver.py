"""Project-scoped catalogue identifier resolution for Selection and Delivery."""

from sqlalchemy import text as _sql

from app.services.database_service import _engine

# Metadata required by Selection and Delivery.
_DEFAULT_COLUMNS = (
    "shot_id, file_path, shot_type, duration_seconds, orientation, fps, "
    "width, height, codec, has_audio, keywords, description, "
    "people_count, mood, "
    "audio_channels, audio_sample_rate, audio_bit_depth"
)


class AmbiguousIdentifier(Exception):
    """Raised when an identifier matches multiple clips in a project."""
    def __init__(self, token: str, candidates: list[tuple]):
        self.token = token
        self.candidates = candidates
        listing = ", ".join(f"#{sid} {fp}" for sid, fp in candidates)
        super().__init__(
            f"'{token}' matches {len(candidates)} clips in this project ({listing}). "
            "Use the shot_id to pick one."
        )


def _escape_like(term: str) -> str:
    """Escape LIKE metacharacters."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _basename(path) -> str:
    """Return the final path component using either path separator."""
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def _match_by_path(conn, project_id, token: str, columns: str) -> list:
    """Find path-based matches within a project."""
    base = f"SELECT {columns} FROM shots WHERE project_id = :pid AND "

    rows = conn.execute(_sql(base + "file_path = :t"),
                        {"pid": project_id, "t": token}).fetchall()
    if rows:
        return rows

    esc = _escape_like(token)
    superset = conn.execute(
        _sql(base + "file_path LIKE :c ESCAPE '\\'"),
        {"pid": project_id, "c": f"%{esc}%"},
    ).fetchall()
    exact_base = [r for r in superset if _basename(r._mapping["file_path"]) == token]
    if exact_base:
        return exact_base

    return superset


def resolve_one(project_id, token, *, columns: str = _DEFAULT_COLUMNS) -> dict | None:
    """Resolve one identifier to a catalogue row within a project."""
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
    """Resolve identifiers in order while preserving repeats."""
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
