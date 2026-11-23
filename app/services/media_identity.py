"""Stable path and source identity; normal imports perform no media analysis."""

from __future__ import annotations

import hashlib
from pathlib import Path


def canonical_path(path) -> str:
    if not str(path or "").strip():
        raise ValueError("A media path is required")
    # resolve() handles relative paths, existing symlinks and native Windows names.
    # Do not blindly lowercase: case-sensitive volumes must remain distinguishable.
    return str(Path(path).expanduser().resolve(strict=False))


def fingerprint(path) -> dict:
    stat = Path(path).stat()
    return {"source_size": stat.st_size, "source_mtime": stat.st_mtime,
            "source_mtime_ns": stat.st_mtime_ns}


def same_stat(cached: dict, current: dict) -> bool:
    size, ns = cached.get("source_size"), cached.get("source_mtime_ns")
    return (size is not None and ns is not None and
            size == current.get("source_size") and ns == current.get("source_mtime_ns"))


def content_hash(path) -> str:
    before = fingerprint(path)
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if fingerprint(path) != before:
        raise ValueError("Source changed while its fingerprint was being calculated")
    return digest.hexdigest()
