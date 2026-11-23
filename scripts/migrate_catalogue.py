"""Explicit v0 -> v1 catalogue deployment. No dotenv, agents or model calls."""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from app.services.catalogue_migrations import initialise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error("Database must already exist; check its path")
    with closing(sqlite3.connect(args.database, timeout=10)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")
        result = initialise(conn, allow_migration=True, database_path=args.database)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
