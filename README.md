# 🎬 MAPO — Multi-Agent Production Orchestrator

An AI-assisted film/video **post-production pipeline** built on [LangGraph](https://langchain-ai.github.io/langgraph/). MAPO ingests raw footage, builds a searchable semantic catalogue, helps an editor retrieve and curate the right clips, orchestrates them into an edit timeline, and compiles that timeline into a project file **Adobe Premiere Pro imports natively**.

Ingestion is **temporal event-based**, not just frame-based: alongside each clip's overall tags ("what is *in* the clip"), MAPO segments every clip into ordered **events** with real start/end timecodes and describes "what *happens*" in each ("a player walks on, then scores"). Those events serve the two later stages differently, along a clean **retrieval-vs-editing boundary**: **Search** stays clip-level (it returns whole clips, but recall is *event-aware* — a clip surfaces when a moment inside it matches), and **Selection** is where moments get *cut* — it reads a clip's events and trims the edit to the exact event boundary.

> MSc project — UCL, Computer Graphics, Vision & Imaging (CGVI). Author: **Yichen Zheng**.

---

## What it is

MAPO is **four specialised ReAct agents** arranged as a **strict linear pipeline**:

```
①  Ingest   →  ②  Search   →  [ Curation (editor) ]  →  ③  Selection   →  ④  Delivery
   catalogue     retrieval          UI Human-in-the-Loop        edit timeline      Premiere compile
```

It is driven as a **fixed state machine** — there is **no LLM "supervisor"** routing between agents. An explicit orchestrator invokes exactly one agent per stage, in order. The whole thing is exposed through a [Streamlit](https://streamlit.io/) UI, which is also the **Human-in-the-Loop (HITL)** layer: the editor decides which clips participate (curation checkboxes) and when to export (the Deliver button).

The guiding rule everywhere is **honesty over completeness**: *store only what a tool measured or the vision model actually saw; leave the rest `unclassified`; never fabricate.* No invented filenames, no made-up quality scores, no fake relevance numbers.

### What is real vs. simulated

| Real | Simulated / not implemented |
|---|---|
| ffprobe technical metadata (dimensions, orientation, fps, codec, audio, duration) | PostgreSQL production datastore (SQLite is used instead) |
| FFmpeg scene-cut detection (cut count + timestamps) | Whisper transcription |
| GPT-5.4 Vision frame tagging (shot type, objects, keywords, description, people count, camera motion, lighting, mood, subject position) | Face / speaker **identity** recognition (people are *counted*, never *identified*) |
| GPT-5.4 **temporal event extraction** (per-clip ordered events: action, state change, subjects, keywords, real in/out timecodes) | Audio event detection (loudness/onsets — future work) |
| Semantic embeddings + hybrid vector/lexical retrieval, with **event-aware clip recall** (moments boost clip search; moments are *cut* in Selection) | Automated quality scoring (removed by design — quality is the editor's call) |
| FCP7 XML (`xmeml` v5) Premiere-importable export | |

---

## Architecture

Everything is assembled in [app/orchestrator/production_orchestrator.py](app/orchestrator/production_orchestrator.py) — **read this first**; it's the spine.

### Two layers

**1. Pipeline orchestrator** — [production_orchestrator.py](app/orchestrator/production_orchestrator.py) exposes four explicit stage functions the UI calls in order:

| Function | Stage | Invokes |
|---|---|---|
| `run_ingest(directory, project_id, user_id)` | ① Ingest | `ingest_agent` |
| `run_search(query, project_id)` | ② Search | `retrieval_service.hybrid_search` (event-aware clip recall) |
| `run_selection(intent, selected_paths, project_id, user_id)` | ③ Selection | `selection_agent` |
| `run_delivery(plan, project_id, user_id)` | ④ Delivery | `delivery_agent` |

The code path, the architecture diagram, and the thesis description therefore all match — there is no "documented supervisor, actually bypassed by the UI" discrepancy. The orchestrator also owns `_extract_plan`, which reads the Selection→Delivery timeline plan out of the planning **tool message** (`plan_timeline` or the event-based `plan_moment_timeline`, not the model's prose), so the model reformatting its narration can never corrupt the plan Delivery receives.

**2. Four sub-agents** in [app/agents/](app/agents/), each a self-contained ReAct loop (`assistant → tools → assistant`) compiled by the shared `_build_agent_graph` helper. Every agent follows the same **three-part contract** the orchestrator imports:

- `@tool`-decorated functions + a `ToolNode`
- `<name>_assistant(state, config)` — binds tools, prepends a `SystemMessage`, returns `{"messages": [response]}`
- `should_continue_<name>(state, config)` — `"continue"` if the last message has tool calls, else `"end"`

### The four agents

| Agent | Responsibility | Key tools |
|---|---|---|
| **[Ingest](app/agents/ingest_agent.py)** | Scan → probe → scene-detect → vision-tag → **embed** → **extract temporal events** → catalogue. Builds the searchable knowledge base (clip-level + event-level). **Incremental** (unchanged files reused via an mtime/size fingerprint — events reused too). Records a structured `IngestResult` the UI gates on. Refuses (no destructive rewrite) if a scan exceeds the per-run file cap. | `detect_new_footage`, `ingest_footage`, `scan_footage_directory`, `get_shots_by_type`, `classify_shot_attributes`, `export_metadata_json`, … |
| **[Search](app/agents/search_agent.py)** | **Retrieval only, clip-level.** Returns candidate **clips** + a suggestion marker + grounded relevance %; never moments/timecodes and never a "best" pick. Recall is **event-aware** (a clip surfaces when a moment inside it matches, shown as a "contains: …" hint), but the returned unit is always the clip. The LLM just fills structured query params. | `search_catalogue` (event-aware clip hybrid), `list_all_shots` |
| **[Selection](app/agents/selection_agent.py)** | Intent-aware **edit-timeline orchestrator** (not a score ranker). Chooses a timeline structure that fits the editing intent — **no fixed narrative arc** — and explains each step's placement, connection, and pacing. **Decides for itself**, from the intent, whether to build a whole-clip timeline or a **moment-precise** one (reading a clip's events and trimming to the exact boundaries). This is where moments get *cut*. Consumes the editor's **curated** clips. | `get_candidate_details`, `get_clip_events`, `plan_timeline`, `plan_moment_timeline`, `generate_delivery_summary` |
| **[Delivery](app/agents/delivery_agent.py)** | **Project compiler** (not an editor). Compiles the timeline into an FCP7 XML Premiere imports natively (+ JSON intermediate). Preserves clip order and trims **exactly** — never re-orders, drops, trims, or edits. | `compile_timeline_segments` (preferred), `compile_premiere_project`, `preview_delivery_timeline` |

There is **no automated quality-scoring agent** — quality is the editor's judgement, informed by the Selection Agent's reasoning. (An earlier 6-agent, then 4-agent-two-phase design was refactored: the Metadata Agent became the Ingest Agent, the Quality Agent was removed entirely, and Delivery was (re)introduced as the 4th stage. A test asserts the removed modules are no longer importable.)

### Project isolation & identifier resolution

- `project_id` reaches tools via LangGraph **`InjectedState`** — the model never fills that security-boundary argument; the `ToolNode` injects it from `ProductionState`. Every catalogue query is scoped to the current project, so one project can never see or compile another's media.
- Selection and Delivery resolve shot-id / file-name identifiers through the shared [catalogue_resolver.py](app/services/catalogue_resolver.py), which filters by `project_id` and raises `AmbiguousIdentifier` (listing candidates) instead of silently taking the first match. It never fabricates: an identifier that matches nothing resolves to `None`, and callers **refuse** rather than proceed.

---

## The pipeline in detail

### ① Ingest — build the knowledge base

One call to `ingest_footage` runs the whole pipeline per clip:

1. **Verify** the media is readable (ffprobe).
2. **Technical metadata** — true dimensions, rotation-aware display orientation, fps, codec, audio presence, duration.
3. **Scene detection** — FFmpeg scene-change cut count + timestamps.
4. **Vision tagging** — GPT-5.4 watches sampled frames (one per detected scene, or adaptive even sampling) and returns a description, shot type, objects, searchable keywords, an approximate people count, and semantic dimensions (camera motion, lighting, mood, subject position).
5. **Embedding** — the description + keywords + mood/lighting are embedded into a vector for semantic recall.
6. **Temporal event extraction** — the clip is segmented into event windows (scene cuts are the primary boundaries; a long continuous shot is sub-divided), and GPT-5.4 describes **what happens** in each window (action, state change, subjects, action keywords) from several ordered frames. Each event is stored with its **real in/out timecodes** and its **own embedding** in a `clip_events` child table for moment-level search.
7. (optional) **Proxy** generation per clip.

The merged results **replace** the project's catalogue. Ingest is **incremental**: an unchanged, already-catalogued file (same size + mtime) is reused as-is — no re-probing, no re-spent Vision calls. Pass `force_reanalyze=True` to override.

### ② Search — hybrid retrieval

[retrieval_service.hybrid_search](app/services/retrieval_service.py) combines three layers:

1. **SQL hard filters** — orientation, shot type, people, duration window.
2. **Vector semantic recall** — the query is embedded and ranked by cosine against stored embeddings (calibrated onto a usable 0–1 relevance band).
3. **Lexical fallback** — keyword/description overlap when embeddings are unavailable (no API key, or the demo seed).

The two signals combine with a **reinforcing (noisy-OR) rule**, so a strong hit in either layer drives relevance up rather than one diluting the other. Both the Search Agent's tool *and* the UI's direct search go through this same function, so retrieval is identical LLM-driven or user-driven. Query understanding (`hoist_orientation` + `expand_query`) routes format words to filters and expands free text into English synonyms.

**Event-aware clip recall.** The retrieval unit is always the **clip** — Search browses footage; it never returns moments or timecodes (deciding *which moment* to use is Selection's job). But events are not dormant here: `hybrid_search` also scores each clip's temporal events and folds the best one into the clip's relevance with the same noisy-OR rule, so a clip whose *overall* tags miss the query still surfaces when a **moment inside it** matches (query "celebration" finds a clip whose only celebratory beat is one event). The matched moment rides along as a `matched_event` field used purely as the card's "contains: …" reason — it never turns the result into a moment. (`retrieval_service.search_events` still exists as the moment-level primitive, but it now backs **Selection**, not a Search user mode.)

### Curation — the Human-in-the-Loop layer

Between Search and Selection, the editor ticks which candidate clips participate (the Streamlit **Bin**). This is the single HITL mechanism — there is no LangGraph `interrupt()` approval gate.

### ③ Selection — intent-driven edit timeline

The Selection Agent reads the curated clips' real metadata, decides an **order** and per-clip **importance**, then calls `plan_timeline` — which resolves each clip's real duration and emits structured segments in one of three modes ([timeline_service.py](app/services/timeline_service.py)):

- **TRIM MODE** — drop the first/last *N* seconds off every clip, keep the middle.
- **TIMED MODE** — a total target length ("30s", "1:30") is split across clips proportionally to importance (water-filled, capped at each clip's real length).
- **FULL CLIP MODE** — clips keep their full length, concatenated in order.

For **moment-precise** edits the agent **decides for itself from the editing intent** (no UI switch): when the intent targets specific moments ("a 30s celebration reel"), it calls `get_clip_events` to see each clip's temporal events and `plan_moment_timeline` with the chosen event IDs in order — producing an **EVENTS MODE** plan whose per-segment in/out points are the events' own measured boundaries (via `timeline_service.plan_event_segments`). This is the stage where moments are actually *cut*, keeping retrieval (Search) and editing (Selection) cleanly separated.

A segment with real footage but nothing left after trimming — or an event range that falls outside the source — is flagged `valid: false` so Delivery refuses it, never silently dropped or emitted. Every mode emits the **same structured plan shape**, so Delivery compiles all of them identically.

### ④ Delivery — compile to Premiere

The Delivery Agent hands the ordered, resolved segments to [premiere_export_service.py](app/services/premiere_export_service.py):

- **`build_timeline`** maps clips into a neutral JSON intermediate (frame-accurate, NTSC-aware timebase, order preserved). An explicit illegal trim raises `InvalidSegmentRange` naming the segment.
- **`to_fcp7_xml`** renders valid FCP7 `xmeml` v5 with tracks **V1** (video), **A1** (original audio), and **A2** (only when a clip genuinely has a second audio stream).
- **`compile_project`** validates every clip's media exists on disk (refuses with `MediaValidationError` otherwise), writes versioned, atomic `.xml` + `.json` to `PROCESSED_OUTPUT_DIR/exports`.

Import the resulting `.xml` in Premiere via **File ▸ Import**.

---

## Shared state & data contracts

[app/models/state.py](app/models/state.py) defines `ProductionState` (a `TypedDict`) — the single object threaded through the graph — typed against the Pydantic contracts in [schemas.py](app/models/schemas.py):

- `SearchCandidate` — mirrors one `hybrid_search` row.
- `TimelineSegment` / `EditPlan` — mirror `plan_segments` output (incl. `valid` / `validation_error`).
- `IngestResult` — the ingest outcome the UI gates on (`success` / `partial_success` / `failure` + `indexed_count` + `event_count`).
- `DeliveryResult` — the compile outcome (artefact paths + sequence shape).
- `VisionTags` — clip-level frame tags, enforced by the vision model via `.with_structured_output(...)`.
- `EventTags` — the **action-oriented** per-event output (`action` / `state_change` / `subjects` / `keywords`), and `ClipEvent` — one persisted `clip_events` row.

`messages` uses the `add_messages` reducer; `remaining_steps` guards against infinite ReAct loops; `footage_dir` carries the run's directory on state (so concurrent sessions don't clobber a global). New fields are added **additively**.

---

## Services

| Service | Role |
|---|---|
| [openai_service.py](app/services/openai_service.py) | Module-level `llm` / `llm_fast` + `embed_texts()`. Returns `None` without a key — never a fabricated vector. **Import `llm` / `embed_texts` from here.** |
| [retrieval_service.py](app/services/retrieval_service.py) | Hybrid retrieval: `hybrid_search` (event-aware **clip** recall — the Search stage) and `search_events` (moment-level primitive — backs **Selection**), sharing one event-scoring path; + query understanding (`expand_query`, `hoist_orientation`, `group_size`). |
| [catalogue_resolver.py](app/services/catalogue_resolver.py) | Project-scoped identifier resolution shared by Selection & Delivery (`resolve_one`, `resolve_ordered`, `AmbiguousIdentifier`). |
| [database_service.py](app/services/database_service.py) | SQLite catalogue. Singleton `db` (read) + write helpers `replace_project_shots` / `get_catalogued_paths` / `get_catalogued_shots`, and the event layer `replace_project_events` / `get_catalogued_events` / `get_events_by_ids`. Persists to a file so ingested rows survive restarts; seeds a demo catalogue only when empty. |
| [ffmpeg_service.py](app/services/ffmpeg_service.py) | ffprobe/FFmpeg wrappers: `probe_video_metadata`, `detect_scene_cut_times`, `build_event_windows`, `extract_*_frames_b64` / `extract_frames_in_window_b64`, `count_audio_streams`, proxies. Degrade gracefully when FFmpeg is absent. |
| [vision_service.py](app/services/vision_service.py) | `analyze_frames()` → clip-level `VisionTags`; `analyze_event()` → per-event `EventTags` (action/change). Both return `None` rather than fabricate. |
| [timeline_service.py](app/services/timeline_service.py) | Pure timeline planner: `parse_target_duration`, `allocate_durations`, `plan_segments` (clip modes), `plan_event_segments` (events mode). |
| [premiere_export_service.py](app/services/premiere_export_service.py) | Pure FCP7 XML/JSON compiler: `build_timeline`, `to_fcp7_xml`, `compile_project`. |

The metadata database is **SQLite persisted to a file** (`settings.METADATA_DB_PATH`). The `shots` table carries technical columns (`width/height/orientation/fps/codec/has_audio/scene_count/description/people_count`) plus semantic columns (`camera_motion/lighting/mood/subject_position/embedding`) and a `UNIQUE(project_id, file_path)` constraint. A child **`clip_events`** table holds the temporal events — one `shots` row → many events (`start_seconds`/`end_seconds`, `action`, `state_change`, `subjects`, `keywords`, per-event `embedding`) with an `ON DELETE CASCADE` foreign key, so replacing a project's shots on re-ingest cleanly rebuilds its events. PostgreSQL remains the aspirational production datastore.

---

## Getting started

### Prerequisites

- **Python 3.10+**
- **FFmpeg / ffprobe** on `PATH` — required for real ingest (probing, scene detection, frame sampling, proxies). If absent, those tools return graceful error strings rather than crashing, but ingest cannot build a real catalogue.
- An **OpenAI API key** — for vision tagging, embeddings, and the agents' reasoning.

### Install

```bash
pip install -r requirements.txt
```

### Configure

Create a `.env` in the project root:

```dotenv
OPENAI_API_KEY=sk-...

# Optional — defaults shown
LLM_MODEL=gpt-5.4
LLM_MODEL_FALLBACK=gpt-5.4-mini
VISION_MODEL=gpt-5.4
EMBEDDING_MODEL=text-embedding-3-small
METADATA_DB_PATH=./app/data/mapo_catalogue.db
RAW_FOOTAGE_DIR=./app/data/raw_footage
PROCESSED_OUTPUT_DIR=./app/data/output
```

All env-backed settings live in [app/config.py](app/config.py) via the typed `settings` singleton — add new ones there, not through scattered `os.getenv` calls.

### Run

```bash
python main.py                            # the only supported entry point (spawns Streamlit)
# equivalently:
streamlit run app/ui/streamlit_app.py
```

Then in the browser:

1. **① Ingest** — set the footage folder, click *Run Ingest Analysis*. Later phases stay locked until ingest reports a structured success with `indexed_count > 0`.
2. **② Search** — type a query; matching **clips** are ranked 🟡 suggested / ⚪ neutral / 🔴 low. Recall is event-aware, so a clip can surface because of a moment inside it — shown as the card's "contains: …" reason. ➕ ticks a clip into the Bin.
3. **Curate** — tick/untick clips in the sidebar **Media Pool** (Select-all available).
4. **③ Selection** — state your editing intent; the agent lays your ticked clips into an explained edit timeline, and **cuts to precise event moments on its own** when the intent calls for it (e.g. "a 30s celebration reel").
5. **④ Deliver** — enabled only when a structured timeline exists; exports the FCP7 XML for Premiere.

---

## Testing

```bash
pytest tests/ -v                                    # all tests
pytest tests/test_agents.py::test_shot_query -v     # a single test
```

Tests point `METADATA_DB_PATH` at `:memory:` via [tests/conftest.py](tests/conftest.py), so they run against the deterministic demo seed (5 shots) and never touch the persistent on-disk catalogue. They cover: database round-trips, agent tool composition, the removed-modules assertion, hybrid search, timeline planning (duration parsing, allocation, all modes, invalid-trim flagging), the Premiere compiler (order preservation, NTSC timebase, V1/A1/A2 layout, media validation, versioned exports), project isolation, ambiguous-identifier handling, the orchestrator's fixed-pipeline shape, and the **temporal event layer** (event schema, `clip_events` round-trip + cascade delete, orphan skipping, event-id project isolation, event-window segmentation, `plan_event_segments`, and moment retrieval).

---

## Project structure

```
main.py                              # entry point → spawns Streamlit
app/
  config.py                          # typed settings (loads .env once)
  orchestrator/
    production_orchestrator.py       # the spine — 4 explicit stage functions, no supervisor
  agents/
    ingest_agent.py                  # ① scan/probe/vision-tag/embed/catalogue
    search_agent.py                  # ② hybrid retrieval (retrieval only)
    selection_agent.py               # ③ intent-driven edit-timeline orchestration
    delivery_agent.py                # ④ Premiere FCP7 XML compiler
  services/
    openai_service.py                # llm / llm_fast / embeddings
    retrieval_service.py             # hybrid_search + query understanding
    catalogue_resolver.py            # project-scoped identifier resolution
    database_service.py              # SQLite catalogue (read + write helpers)
    ffmpeg_service.py                # ffprobe/FFmpeg wrappers
    vision_service.py                # GPT-5.4 frame tagging
    timeline_service.py              # pure timeline planner
    premiere_export_service.py       # pure FCP7 XML/JSON compiler
  models/
    schemas.py                       # Pydantic stage-to-stage contracts
    state.py                         # ProductionState (LangGraph shared state)
  ui/
    streamlit_app.py                 # the UI + Human-in-the-Loop layer
  utils/logger.py                    # get_logger("<module>")
  data/                              # SQLite catalogue, footage, outputs/exports
tests/                               # pytest suite (:memory: DB)
```

---

## Design principles

- **Never fabricate catalogue data.** Tools return real measurements or `None`/`unknown`; agent prompts forbid inventing filenames, scores, or tags.
- **Hard responsibility boundaries.** Search *retrieves* (no "best" pick); Selection *orchestrates* the curated clips into an intent-driven timeline (no ranking-by-score); Delivery is a *pure compiler* (order in = order out — no re-ordering, dropping, or trimming).
- **Fixed pipeline, no supervisor.** One explicit orchestration path shared by the UI and the agents.
- **The editor stays in control.** HITL is the UI — curation checkboxes decide what participates; the Export button decides when to compile.

Tool docstrings are the agents' API contract — the LLM selects and fills tools from them, so they are written precisely. See [CLAUDE.md](CLAUDE.md) for the full contributor guide.
