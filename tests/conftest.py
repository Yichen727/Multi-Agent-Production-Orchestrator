"""Pytest configuration for MAPO.

Forces the metadata catalogue to an in-memory SQLite database BEFORE any app module
imports, so tests run against the deterministic demo seed and never read or write the
persistent on-disk catalogue the app uses at runtime.
"""

import os

# Must be set before `app.config` is imported (which reads env at import time).
os.environ.setdefault("METADATA_DB_PATH", ":memory:")
