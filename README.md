# 🎬 MAPO — Multi-Agent Production Orchestrator

An AI-assisted film/video **post-production pipeline** built on [LangGraph](https://langchain-ai.github.io/langgraph/). MAPO ingests raw footage, builds a searchable catalogue, helps you find and curate the right clips, arranges them into an edit timeline, and exports a project file **Adobe Premiere Pro imports natively**.

Built for a small company editing team using one workstation. Author: **Yichen Zheng**.

---

## The pipeline

Four pipeline capabilities are available through a [Streamlit](https://streamlit.io/) workspace. Existing media and saved plans can be opened directly. Selection is the reasoning agent; ingestion, search, and export use direct services. You stay in control at two points: which clips participate, and when to export.

```
①  Ingest   →  ②  Search   →  [ Curation ]  →  ③  Selection   →  ④  Deliver
  catalogue     find clips   choose candidates    edit timeline     Premiere XML
```

| Stage | What it does |
|---|---|
| **① Ingest** | Analyses footage with ffprobe, FFmpeg, GPT-5.5 Vision, and embeddings, then builds a local searchable catalogue of clips and temporal events. |
| **② Search** | Retrieves and ranks whole clips against a natural-language query, with relevance scores and retrieval reasons. |
| **Curation** | Lets the editor select candidate clips in Media Library. Only selected clips reach Selection. |
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

The sidebar contains the project picker, workspace settings and **Footage → Import / Update**.
The main area has four equal pages: **Media Library**, **Find Footage**, **Edit Plan**, **Export**.
Only the active page renders. Opening a page never runs a search, analysis, planning or export.

### Project and footage

Choose a project by name. **Workspace settings** contains Editor ID, the diagnostic
Project ID and a **New project name → Create project** form. Creating a project is explicit.
Each browser session has its own workspace; project/editor switches restore that session's
own candidates, input drafts, saved plans and media folder. Editor ID is a workspace label,
not an authentication mechanism.

Set **Media folder** under Footage and click **Import / Update**. The import reports
**Reused / New / Changed / Failed** counts and analysis updates. Existing online media is
available immediately after startup; an unsuccessful import does not erase the library.

Imports reuse unchanged clips and events, preserving IDs and completed analysis. The UI
explicitly processes large folders in pages of at most 200 files. Import options contain
visual/moment analysis, explicit reanalysis and content verification. The UI does not
rescan the media directory on every rerun. Import is synchronous in P1c; background
progress, cancellation and recovery are planned for P2.

### Media Library

Browse paginated clip cards with duration, orientation and analysis coverage. Filter by
filename/tags, orientation, analysis status, or **Candidates only**. Check a clip to add it
to the edit's candidate set; **Select this page / Clear this page** affect only the current
filtered page. **Clear candidates** in the sidebar clears the entire candidate set.

**Preview** opens one source preview. **Show thumbnails** loads frames for the current
page only and defaults off to keep opening the library fast. **Create edit plan** opens
the planning page with the same candidates.

### Find Footage

Describe subjects, actions or atmosphere and submit **Find footage**. An empty query lists
the available catalogue. Search runs only on submission; paging, previews and checkbox
changes reuse the result list. Results preserve retrieval order and relevance, with event
evidence where available.

Result checkboxes and **Add this page / Remove this page** use the same stable clip IDs
as Media Library. Switching pages or changing filters does not lose candidates.

### Edit Plan

Choose **Full Clips** to keep complete source lengths, or **Selected Moments** to choose
events within clips with an optional target duration. If candidates have no analysed
moments, update them using **Analyse moments** before requesting moment editing.

Set the output aspect ratio, describe the editing intent, give the plan a name and click
**Generate edit plan**. Each successful generation saves a separate version in this
browser workspace. The page shows the timeline, duration and editorial explanation.
**Export this plan** opens Export without compiling automatically.

Plans survive page and project/editor switches within the session. Use **Download plan
JSON** to persist a plan beyond the session. Reopen that JSON on Export; source references
are verified before it can be used. Local order/trim editing is planned for P3.

### Export

Choose a version under **Saved plan**, name the sequence and click **Export Premiere
project**. Export preserves the chosen plan's order, trims, source versions and output
aspect ratio. Download the Premiere XML and timeline JSON, then import the XML in
Premiere using **File → Import**.

Opening/revisiting Export does not compile again. Plans with changed sources, unavailable
media or stale events show a regeneration message and cannot be exported. Historical
JSON is accepted only when identity and source provenance can be verified; the original
file is never overwritten.

### Data and verification

The catalogue stays in `METADATA_DB_PATH`; exports are written under
`PROCESSED_OUTPUT_DIR/exports`. Keep SQLite and its WAL on the workstation.
P1c does not change model choices or video analysis algorithms.

Run offline tests with `python -B -m pytest -q`. Real model quality and Premiere import
validation are separate acceptance tasks; see [PLAN.md](PLAN.md).

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
and optional duration. P1c provides four pages with shared stable-ID candidates and project/editor workspaces.

See [P1b delivery record](docs/P1B_CONTRACTS.md) and
[development conventions](docs/DEVELOPMENT.md).



## P1c workspace

See [P1c delivery record](docs/P1C_WORKSPACE.md) for the page flow, state migration and
acceptance coverage. The former path-based checkbox state is migrated once; production
candidate state uses `candidate_clip_ids`. Saved plan versions are session-local, with
explicit JSON download/open for persistence.
