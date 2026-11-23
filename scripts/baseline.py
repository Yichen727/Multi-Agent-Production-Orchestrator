"""Capture a WAL-consistent local baseline without importing the application."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import uuid
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
DIRECT_DEPENDENCIES = (
    "langchain", "langchain-openai", "langgraph", "langchain-community", "langsmith",
    "SQLAlchemy", "python-dotenv", "pydantic", "streamlit", "ffmpeg-python", "numpy",
    "pytest",
)
NAME_PATTERN = re.compile(
    r"\b(?:shots|shot_id|ShotMetadata|shot_metadata|selected_shots|total_shots|"
    r"get_catalogued_shots|replace_project_shots|list_all_shots|shot_type)\b"
    r"|Media Pool|Run Ingest Analysis|Editing mode|Editing intent|Sequence name"
    r"|key\s*=\s*[\"'][^\"']+[\"']"
)


def read_only_database(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def dependency_versions() -> dict[str, str]:
    """Pin the installed runtime/test dependency closure, not unrelated Conda packages."""
    pending = list(DIRECT_DEPENDENCIES)
    found = {}
    while pending:
        name = pending.pop()
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in found:
            continue
        dist = metadata.distribution(name)
        found[key] = dist.version
        for item in dist.requires or []:
            requirement = Requirement(item)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return dict(sorted(found.items()))


def command_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return result.stdout.strip().splitlines()[0] if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def source_files(root: Path):
    """Explicit source allowlist: never walk .env, runtime data, credentials or media."""
    for name in ("main.py", "README.md", "PLAN.md", "CLAUDE.md", ".gitignore", ".env.example",
                 "pytest.ini", "requirements.txt", "requirements-dev.txt", "requirements-p0.lock.txt"):
        path = root / name
        if path.is_file():
            yield path
    for folder in ("app", "tests", "scripts"):
        for path in sorted((root / folder).rglob("*.py")):
            if "data" not in path.relative_to(root).parts and not path.is_symlink():
                yield path
    for folder, suffix in (("docs", ".md"), ("benchmarks", ".json"), ("tests/fixtures", ".sql")):
        for path in sorted((root / folder).rglob("*" + suffix)):
            if path.is_file() and not path.is_symlink():
                yield path


def catalogue_schema(conn: sqlite3.Connection) -> dict:
    schema = {}
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )]
    for name in names:
        # Names originate in sqlite_master; quote SQL identifiers, never execute row contents.
        quoted = '"' + name.replace('"', '""') + '"'
        schema[name] = {
            "count": conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0],
            "columns": [dict(r) for r in conn.execute(f"PRAGMA table_info({quoted})")],
            "foreign_keys": [dict(r) for r in conn.execute(f"PRAGMA foreign_key_list({quoted})")],
            "indexes": [dict(r) for r in conn.execute(f"PRAGMA index_list({quoted})")],
        }
    return schema


def select_samples(conn: sqlite3.Connection, limit: int = 26) -> list[dict]:
    modern = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='clips'").fetchone()
    table, identifier = ("clips", "clip_id") if modern else ("shots", "shot_id")
    active = " AND e.event_set_id=s.active_event_set_id" if modern else ""
    rows = [dict(r) for r in conn.execute(
        f"SELECT s.{identifier} AS clip_id, s.project_id, s.file_path, s.duration_seconds, s.codec, "
        "s.orientation, s.fps, s.scene_count, s.source_size, s.source_mtime, "
        f"COUNT(e.event_id) AS event_count FROM {table} s "
        f"LEFT JOIN clip_events e ON e.{identifier}=s.{identifier}{active} "
        f"GROUP BY s.{identifier} ORDER BY s.project_id, s.file_path"
    )]
    chosen = {}
    for pid in sorted({r["project_id"] for r in rows}):
        group = [r for r in rows if r["project_id"] == pid]
        for key in ("duration_seconds", "scene_count", "event_count", "fps"):
            ordered = sorted(group, key=lambda r: (r[key] or 0, r["file_path"]))
            for row in (ordered[0], ordered[-1]):
                chosen[row["clip_id"]] = row
        for orientation in ("portrait", "landscape", "square"):
            row = next((r for r in group if r["orientation"] == orientation), None)
            if row:
                chosen[row["clip_id"]] = row
    remainder = [r for r in rows if r["clip_id"] not in chosen]
    random.Random(20261123).shuffle(remainder)
    for row in remainder:
        if len(chosen) >= limit:
            break
        chosen[row["clip_id"]] = row
    return list(chosen.values())[:limit]


def capture(database: Path, output: Path) -> Path:
    if not database.is_file():
        raise FileNotFoundError(database)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    destination = output.resolve() / stamp
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = destination / "catalogue.sqlite3"
    source = read_only_database(database)
    try:
        with closing(sqlite3.connect(snapshot)) as target:
            # SQLite's backup API includes committed WAL contents. Never copy a live .db alone.
            source.backup(target)
            # The backup inherits the source journal mode. Make the snapshot a
            # self-contained file so later readers cannot invalidate its manifest
            # by creating/removing transient WAL/SHM sidecars.
            target.execute("PRAGMA journal_mode=DELETE")
    finally:
        source.close()
    with closing(read_only_database(snapshot)) as conn:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Snapshot integrity check failed")
        schema = catalogue_schema(conn)
        samples = select_samples(conn)
        foreign_key_errors = [tuple(r) for r in conn.execute("PRAGMA foreign_key_check")]
    dump_json(destination / "schema.json", {"tables": schema, "foreign_key_errors": foreign_key_errors})
    dump_json(destination / "samples.json", {
        "seed": 20261123, "annotation_status": "pending_human_review", "samples": samples,
    })
    exports = ROOT / "app/data/output/exports"
    for path in sorted(exports.iterdir() if exports.exists() else []):
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in {".json", ".xml"}:
            target = destination / "exports" / path.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(path, target)
    references = []
    with zipfile.ZipFile(destination / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files(ROOT):
            archive.write(path, path.relative_to(ROOT).as_posix())
            if path.suffix in {".py", ".md"}:
                for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if NAME_PATTERN.search(line):
                        references.append({"file": path.relative_to(ROOT).as_posix(), "line": lineno,
                                           "text": line.strip()[:240]})
    dump_json(destination / "naming_references.json", references)
    versions = dependency_versions()
    (destination / "requirements.lock.txt").write_text(
        "# Installed P0 runtime/test dependency closure; Python " + platform.python_version() + "\n"
        + "\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n", encoding="utf-8")
    dump_json(destination / "environment.json", {
        "python": platform.python_version(), "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "dependencies": versions,
        "ffmpeg": command_output(["ffmpeg", "-version"]),
        "ffprobe": command_output(["ffprobe", "-version"]),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
    })
    files = {p.relative_to(destination).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
             for p in destination.rglob("*") if p.is_file() and not p.name.endswith(("-wal", "-shm"))}
    dump_json(destination / "manifest.json", {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "database_source": str(database.resolve()), "files": files,
        "includes": "Committed catalogue/WAL snapshot, export examples, source, environment; no secrets or media",
    })
    return destination


def verify(destination: Path) -> dict:
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    checked = 0
    ignored = []
    for name, record in manifest["files"].items():
        # P0's older manifests accidentally included transient read sidecars.
        # Only ignore SHM or an explicitly empty WAL for the known snapshot.
        # A nonempty WAL is never silently discarded.
        if name == "catalogue.sqlite3-wal" and record["bytes"] == 0:
            wal = destination / name
            if wal.exists() and wal.stat().st_size:
                raise ValueError("Snapshot has an unexpected nonempty WAL")
        if name == "catalogue.sqlite3-shm" or (name == "catalogue.sqlite3-wal" and record["bytes"] == 0):
            ignored.append(name)
            continue
        path = (destination / name).resolve()
        if not path.is_relative_to(destination.resolve()) or sha256(path) != record["sha256"]:
            raise ValueError(f"Baseline checksum failed: {name}")
        checked += 1
    with closing(read_only_database(destination / "catalogue.sqlite3")) as conn:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Baseline database integrity failed")
    return {"verified_files": checked, "ignored_transient_files": ignored, "status": "ok"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    take = sub.add_parser("capture")
    take.add_argument("--database", type=Path, default=ROOT / "app/data/mapo_catalogue.db")
    take.add_argument("--output", type=Path, default=ROOT / ".local/p0")
    check = sub.add_parser("verify")
    check.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.action == "capture":
        destination = capture(args.database, args.output)
        print(json.dumps({"directory": str(destination), **verify(destination)}))
    else:
        print(json.dumps(verify(args.directory)))


if __name__ == "__main__":
    main()
