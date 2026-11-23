# 🎬 MAPO — Multi-Agent Production Orchestrator

An AI-assisted film/video **post-production pipeline** built on [LangGraph](https://langchain-ai.github.io/langgraph/). MAPO ingests raw footage, builds a searchable catalogue, helps you find and curate the right clips, arranges them into an edit timeline, and exports a project file **Adobe Premiere Pro imports natively**.

Built for a small company editing team using one workstation. Author: **Yichen Zheng**.

---

## The pipeline

Four stages run in a fixed order, driven from a [Streamlit](https://streamlit.io/) UI. Selection is the reasoning agent; ingestion, search, and export use direct services. You stay in control at two points: which clips participate, and when to export.

```
①  Ingest   →  ②  Search   →  [ Curation ]  →  ③  Selection   →  ④  Deliver
  catalogue     find clips   tick Media Pool    edit timeline     Premiere XML
```

| Stage | What it does |
|---|---|
| **① Ingest** | Analyses footage with ffprobe, FFmpeg, GPT-5.5 Vision, and embeddings, then builds a local searchable catalogue of clips and temporal events. |
| **② Search** | Retrieves and ranks whole clips against a natural-language query, with relevance scores and retrieval reasons. |
| **Curation** | Lets the editor select candidate clips in the Media Pool. Only selected clips reach Selection. |
| **③ Selection** | Acts as an assistant editor, turning the curated candidate set into an explained edit timeline according to the editing intent. |
| **④ Deliver** | Compiles the timeline into FCP7 XML (`xmeml` v5) + JSON for Premiere Pro import. |

---

## Getting started

### Prerequisites

- **Python 3.10+**
- **FFmpeg / ffprobe** on `PATH` — [official download page](https://ffmpeg.org/download.html).
- An **OpenAI API key** 


### Install

```bash
pip install -r requirements.txt
```

### Configure

Create a `.env` in the project root:

```dotenv
# Required — add your own OpenAI API key here
OPENAI_API_KEY=sk-...

# defaults shown
LLM_MODEL=gpt-5.5
LLM_MODEL_FALLBACK=gpt-5.5-mini
VISION_MODEL=gpt-5.5
EMBEDDING_MODEL=text-embedding-3-small

METADATA_DB_PATH=./app/data/mapo_catalogue.db
RAW_FOOTAGE_DIR=./app/data/raw_footage
PROCESSED_OUTPUT_DIR=./app/data/output

# Optional — LangSmith observability
# LANGSMITH_API_KEY=...
# LANGSMITH_TRACING=true
```

### Run

```bash
python main.py                 
```

---

## User Manual

### UI Overview

The interface is split into two areas:

- **Sidebar (left)**: project configuration, footage folder, and the Media Pool.
- **Main workspace (full width)**: the four production stages arranged top-to-bottom.

**Stage-locking mechanism**: downstream operations stay disabled (🔒 Locked) until the required preceding stage completes:

- Search / Selection / Deliver unlock when the current project has online catalogued clips, including after a restart;
- Selection requires at least 1 clip ticked in the Media Pool;
- Deliver additionally requires Selection to produce a **structured timeline**.

### Step-by-step Walkthrough

#### Step 1 · Project Setup and Footage Ingestion 

1. In the sidebar `⚙️ Project Settings`, enter a **Project ID** (default `1`) and an **Editor ID** (default `editor_01`).
2. In `📁 Footage`, set **Full path to the folder containing your video files** (default `./app/data/raw_footage`). If the folder exists, the UI shows `✓ Found N video files`; otherwise it warns `⚠ Folder not found`. Supported formats: `mp4, mov, avi, mxf, r3d, braw`.
3. Click `🎬 Run Ingest Analysis` in the main area. Ingestion performs: ffprobe probing → FFmpeg scene detection → GPT-5.5 Vision tagging → temporal event splitting → embedding → SQLite catalogue write.
4. On success the Media Pool is populated and downstream stages unlock.

> **Ingestion notes:**
>
> - The default import accepts **up to 200 video files** and refuses oversized scans before writing. The service API also supports explicit pages via `batch_offset=0` and `result.next_offset`; paged imports append/update and cannot synchronize missing files.
> - **Incremental reuse**: unchanged files reuse their cached analysis, skipping FFmpeg/LLM calls.


#### Step 2 · Search
1. Type what you are looking for in **Search query** and click `🔍 Search`.
   Describe the **content itself** — objects, scenes, actions, or atmosphere, rather than editing instructions.
   Example: `Find all clips showing mobile phones.`
2. Results are grouped into three relevance tiers: `🟡 Suggested` / `⚪ Neutral` / `🔴 Low`.
3. Each result card shows: **clip name · relevance % · duration**, plus a `💡` retrieval reason.
4. Add/remove clips from the Media Pool: use the per-card `➕` / `➖`, or the per-tier `➕ Add all (N)` / `➖ Remove all (N)` for one-click bulk actions.
5. `✖ Clear results` only clears the result cards.


#### Step 3 · Candidate Curation

- Tick the clips you want to use in the sidebar `🎞️ Media Pool`. `Select all` ticks/unticks the whole pool in one click; `Clear ✕` clears all selections; `↻ Refresh` reloads the pool.
- Each row's `▶` opens a preview popover.
- **Optional shortcut**: skip Search and tick the entire Media Pool, letting Selection evaluate the full set against the editing intent. This means less manual work and one fewer retrieval stage, but Selection must consider more material, so reasoning time may increase.


#### Step 4 · Edit Timeline Generation

1. **Editing mode**:
   - `🎞️ Clip Assembly` — Combine complete clips. The whole clip is the editing unit, original durations are kept, **no trimming**, no target duration. Recommended for **vlogs, travel videos, documentaries, BTS, and montages with short, self-contained clips**.
   - `🎯 Moment Assembly` — Select moments from within clips. Temporal events inside clips are the editing unit, with an **optional target duration**. Recommended for **extracting highlights from longer clips or creating an edit to match a target runtime**.
2. **Output aspect ratio**: `16:9 / 9:16 / 4:3 / 3:4 / 1:1` (default 16:9). This is an explicit **output spec** that carries through to the export: each clip is scaled to **fit** the frame with its own aspect ratio preserved.
3. (Moment Assembly only) **Target duration (seconds)**: range 1–7200. It is an *optimisation target* rather than a hard constraint.
4. **Editing intent**: Describe the **editorial direction** in terms of **style, emotion, pacing, and purpose**, rather than technical operations.
   Example: `Fast-paced tech product promo, showcasing various functions.`
5. Click `🎬 Generate Edit Timeline`.

The result renders inline in the Selection section:

- Summary line: mode · segment count · total duration · target-duration status · compression info;
- `🧭 Ordering` strategy;
- Narrative report: each segment's source in/out points, duration, ordering logic, and editorial reasoning;
- `🗂️ Not used — backup material (N)` expander: the alternatives (**none of these are in the export**).

> Regenerating the timeline **clears the previous export**.


#### Step 5 · Project Export and NLE Review 

1. Enter a **Sequence name** (default `MAPO Edit`).
2. Click `📦 Export Project`. Delivery compiles the structured timeline into **FCP7 XML (xmeml v5) + JSON**, preserving order and in/out points exactly, writes them to the output directory, and shows an export summary.
3. On success a `📂 Show <XML filename> in folder` button appears, which opens the file manager with the file selected (requires the app and the browser to run on the same machine).
4. Drag the XML straight into **Premiere Pro (File ▸ Import)**; after import, make final adjustments in the timeline (e.g. extend or shorten individual clips).


### Debug & Maintenance

- **🛠️ Debug log (N messages)**: collapsible panel at the bottom of the main area; it records the raw exchanges of every agent (the key Selection/Delivery outputs already render in their own sections; this is the full transcript for troubleshooting).
- **🔄 Reset Pipeline**: at the bottom of the sidebar; clears all session state (Media Pool ticks, search results, timeline, export) and restarts.

---

### FAQ

| Symptom | Cause / fix |
| --- | --- |
| `⚠ Folder not found` | Wrong footage-folder path; check the sidebar input |
| Stages still locked after Ingest | Ingest did not actually succeed (failure or 0 clips); check the error and the Debug log |
| `partial_success` warning | Some clips failed analysis (e.g. an undecodable file); the rest are fine |
| More than 200 files rejected | Per-run cap; split the folder into batches or raise `_MAX_INGEST_FILES` |
| Deliver button disabled | No structured timeline yet; generate a Selection timeline first |
| `⚠ Order matches …` | The order matches the order the material was listed; read the report to judge whether it fits the intent |
| `📂 Show in folder` does nothing | The app is not running on this machine (remote deployment); use the displayed absolute path |
| No OpenAI key | The app runs, but Search/Selection/vision analysis are unavailable; Ingest can only do local probing |
| Thumbnails / inline playback unavailable | The codec does not support inline preview; this is a normal graceful degradation |

---

### Third-party Services & Dependencies

| Dependency / service | Type | Purpose | How to connect |
| --- | --- | --- | --- |
| OpenAI API | Paid API (the project's only external model service) | vision tagging, embeddings, agent reasoning | `OPENAI_API_KEY` in `.env` |
| FFmpeg / ffprobe | Local open-source tools | probing, scene detection, frame sampling, proxy generation | install system-wide and add to `PATH` |
| LangGraph / LangChain | Open-source Python libraries | agent graph orchestration, tool routing | `pip` install; no account |
| LangSmith | Optional observability platform | trace tracking | optional `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true` in `.env` |
| SQLite | Local database | catalogue persistence (WAL, foreign keys, connection pool) | local file `METADATA_DB_PATH`; no account |
| Streamlit | Open-source UI framework / optional hosting | UI framework; optional hosting on Streamlit Community Cloud | no account for local runs; platform account for hosting |
| Adobe Premiere Pro | target NLE (provided by the user, not a system dependency) | imports the FCP7 XML project | user's own software |

---

## Project structure

```
main.py                              # entry point → spawns Streamlit
app/
  config.py                          # typed settings (loads .env once)
  orchestrator/
    production_orchestrator.py       # the spine — 4 explicit stage functions
  agents/
    selection_agent.py               # ③ intent-driven edit timeline
  services/
    ingest_service.py                # ① incremental, typed result, no agent loop
    search_service.py                # ② one SearchQuery parse + hybrid retrieval
    delivery_service.py              # ④ validated deterministic export
    plan_contract.py                 # versioned reads + legacy ID/path checks
    workspace_state.py               # project/editor session reconciliation
    openai_service.py                # llm + embeddings
    retrieval_service.py             # hybrid search + query understanding
    catalogue_resolver.py            # project-scoped identifier resolution
    database_service.py              # SQLite catalogue (clips + events)
    ffmpeg_service.py                # ffprobe / FFmpeg wrappers
    vision_service.py                # clip + event tagging
    timeline_service.py              # timeline builders (both modes)
    premiere_export_service.py       # FCP7 XML/JSON compiler
  models/
    schemas.py                       # stage-to-stage data contracts
    state.py                         # shared pipeline state
  ui/
    streamlit_app.py                 # the UI
  utils/logger.py
  data/                              # catalogue, footage, outputs/exports
tests/                               # pytest suite
```

## P0 engineering baseline

The optimization baseline is documented in [docs/P0_BASELINE.md](docs/P0_BASELINE.md).
It includes WAL-consistent local backups, pinned Windows/Python 3.12 dependencies,
offline regressions, local latency/usage measurements, fixed evaluation inputs,
and an opt-in budgeted query-parser trial. No live model evaluation runs by default.

```bash
python -m pytest -q
python -m scripts.baseline capture
python -m scripts.evaluate --budget-usd 0.05 --max-requests 2  # dry-run
```

Known P1/P5 defects are strict expected failures, listed in the baseline document.
The upcoming database/model/form/UI rename is tracked in
[docs/P0_NAMING_INVENTORY.md](docs/P0_NAMING_INVENTORY.md).

## P1a catalogue migration

The catalogue now uses physical `clips/clip_id` tables/fields with stable source
identities, active event sets and per-step analysis status. Importing unchanged
media reuses completed work and preserves both clip and event IDs; importing
another folder appends instead of replacing the project. See
[docs/P1A_CATALOGUE.md](docs/P1A_CATALOGUE.md) for scope, backups and validation.

Fresh databases are created automatically. Deploy an existing legacy database
explicitly before starting/restarting the app:

```bash
python -m scripts.migrate_catalogue app/data/mapo_catalogue.db
```

The command creates a consistent backup and migrates once; repeated execution
does not recreate IDs. This workspace's catalogue has already been migrated.
Public model/state naming and UI navigation remain scheduled for P1b/P1c.

## P1b contracts and services

P1b uses `ClipMetadata`, `clip_id`, `clip_metadata`, and `selected_clips` throughout
production code. `shot_type` remains the photographic shot classification.
New timeline plans and export JSON carry `schema_version: 1`, `project_id`,
`clip_id`, `source_revision`, and event references where relevant.

`read_edit_plan(value, project_id)` accepts historical unversioned `shot_id` plans
and exported JSON at one compatibility boundary. It verifies IDs against original
paths within the project. `validate_edit_plan` additionally checks current source
revisions, disk fingerprints and real event/trim bounds before export. Historical
files without source versions can be inspected, but require regeneration before
export; reading does not invent provenance or overwrite the originals.

Each import/export returns its own typed result and `run_id`. Existing catalogues
load immediately; switching project/editor clears stale candidates and outputs.
Selection tools enforce the curated candidates and the editor's mode, aspect ratio,
and optional duration. P1c will reorganize the pages and remaining visible labels.

See [P1b delivery record](docs/P1B_CONTRACTS.md) and
[development conventions](docs/DEVELOPMENT.md).
