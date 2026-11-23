"""Offline baseline: lexical/event retrieval and local compilation on a DB copy."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sqlite3
import tempfile
import time
from pathlib import Path
from contextlib import ExitStack
from xml.etree import ElementTree as ET

from scripts.baseline import ROOT, dump_json, read_only_database, verify


def percentiles(values):
    ordered = sorted(values)
    def percentile(p):
        position = (len(ordered) - 1) * p
        lo = int(position)
        return round(ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (position - lo), 3)
    return {"samples": len(values), "p50_ms": percentile(.50), "p95_ms": percentile(.95)}


def run(baseline: Path, output: Path, repeats: int = 5) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    verify(baseline)
    if "app.config" in __import__("sys").modules:
        raise RuntimeError("Run benchmark in a fresh process so it cannot reuse a live app database")
    os.environ.update({"PYTHON_DOTENV_DISABLED": "1", "OPENAI_API_KEY": "offline-benchmark",
                       "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
                       "LANGSMITH_API_KEY": "", "METRICS_ENABLED": "false"})
    def blocked(*args, **kwargs):
        raise RuntimeError("Network is disabled in offline benchmark")
    old_connect, old_connection = socket.socket.connect, socket.create_connection
    socket.socket.connect = blocked
    socket.create_connection = blocked
    local = ROOT / ".local/benchmark-work"
    local.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=local) as directory, ExitStack() as cleanup:
            database = Path(directory) / "benchmark.sqlite3"
            shutil.copy2(baseline / "catalogue.sqlite3", database)
            os.environ["METADATA_DB_PATH"] = str(database)
            from app.services.catalogue_migrations import initialise
            with sqlite3.connect(database) as migration:
                migration.execute("PRAGMA foreign_keys=ON")
                initialise(migration, allow_migration=True)
            migration.close()
            from app.services import database_service as db, retrieval_service as retrieval
            cleanup.callback(db._engine.dispose)
            from app.services.timeline_service import build_clip_timeline, build_moment_timeline
            from app.services.premiere_export_service import compile_project
            retrieval.embed_texts = lambda texts: None
            cases = json.loads((ROOT / "benchmarks/quality_cases.json").read_text(encoding="utf-8"))
            searches = []
            for case in cases["search_cases"]:
                durations = []
                results = []
                kwargs = {key: case[key] for key in ("core_keywords", "orientation", "shot_type", "people",
                                                   "min_duration", "max_duration") if key in case}
                kwargs["keywords"] = kwargs.get("core_keywords")
                # First call recorded separately; these are process-local cold data, not a cold OS cache.
                first = None
                for i in range(repeats + 1):
                    start = time.perf_counter()
                    results = retrieval.hybrid_search(case["project_id"], **kwargs)
                    elapsed = (time.perf_counter() - start) * 1000
                    if i == 0:
                        first = elapsed
                    else:
                        durations.append(elapsed)
                expected = case["relevant_paths"]
                recall = (len(set(expected) & {r["file_path"] for r in results[:20]}) / len(set(expected))
                          if expected else None)
                searches.append({"case_id": case["id"], "first_call_ms": round(first, 3),
                                 **percentiles(durations), "result_count": len(results), "recall_at_20": recall})
            edits = []
            for pid in (1, 2):
                all_clips = db.get_catalogued_shots(pid)
                clips = list(all_clips.values())[:12]
                events = [e for group in db.get_catalogued_events(pid).values() for e in group][:40]
                durations = {r["file_path"]: r["duration_seconds"] for r in clips}
                for event in events:
                    if event["file_path"] not in durations:
                        durations[event["file_path"]] = all_clips[event["file_path"]]["duration_seconds"]
                    event["source_duration"] = durations[event["file_path"]]
                for mode in ("clip", "moment"):
                    times = []
                    for _ in range(repeats):
                        start = time.perf_counter()
                        plan = (build_clip_timeline(clips) if mode == "clip" else
                                build_moment_timeline(events, target_seconds=60))
                        rows = db.get_catalogued_shots(pid)
                        media = [{**rows[s["file_path"]], "in_point": s["in_point"], "out_point": s["out_point"]}
                                 for s in plan["segments"]]
                        result = compile_project(media, write=False, verify_media=False)
                        ET.fromstring(result["xml"])
                        times.append((time.perf_counter() - start) * 1000)
                    edits.append({"project_id": pid, "mode": mode, "segments": len(media), **percentiles(times)})
            historical = []
            for path in sorted((baseline / "exports").glob("*.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                xml = ET.parse(path.with_suffix(".xml"))
                historical.append({"name": path.name, "clips": len(data["clips"]),
                                   "xml_clips": len(xml.findall("./sequence/media/video/track/clipitem")),
                                   "total_frames": data["sequence"]["total_frames"]})
            report = {"mode": "offline_lexical_and_event_retrieval_local_compile",
                      "network_calls": 0, "model_cost_usd": 0, "repeats": repeats,
                      "limitations": ["No LLM or embedding latency/semantic evaluation", "No FFmpeg/media verification",
                                      "No Premiere import validation", "Human labels pending; null recall is not zero"],
                      "searches": searches, "local_timeline_and_xml": edits, "historical_exports": historical}
            dump_json(output, report)
            return report
    finally:
        socket.socket.connect = old_connect
        socket.create_connection = old_connection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/p0-offline-results.json")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    result = run(args.baseline, args.output, args.repeats)
    print(json.dumps({"output": str(args.output), "search_cases": len(result["searches"]),
                      "local_edit_cases": len(result["local_timeline_and_xml"]), "model_cost_usd": 0}))


if __name__ == "__main__":
    main()
