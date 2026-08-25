# 🎬 MAPO — Multi-Agent Production Orchestrator

An AI-assisted film/video **post-production pipeline** built on [LangGraph](https://langchain-ai.github.io/langgraph/). MAPO ingests raw footage, builds a searchable catalogue, helps you find and curate the right clips, arranges them into an edit timeline, and exports a project file **Adobe Premiere Pro imports natively**.

> MSc project — UCL, Computer Graphics, Vision & Imaging (CGVI). Author: **Yichen Zheng**.

---

## The pipeline

Four agents run in a fixed order, driven from a [Streamlit](https://streamlit.io/) UI. You stay in control at two points: which clips participate, and when to export.

```
①  Ingest   →  ②  Search   →  [ Curation ]  →  ③  Selection   →  ④  Deliver
  catalogue     find clips    tick the Bin      edit timeline      Premiere XML
```

| Stage | What it does |
|---|---|
| **① Ingest** | Probes each file with ffprobe (resolution, orientation, fps, codec, audio, duration), detects scene cuts with FFmpeg, and sends sampled frames to GPT-5.5 Vision for tags (shot type, objects, keywords, description, people count, mood). It also splits every clip into ordered **events** — "what happens", with real start/end timecodes. Everything lands in a local catalogue. Re-running skips unchanged files. |
| **② Search** | Type what you're looking for; you get back **whole clips**, ranked with a relevance % and a suggestion marker (🟡 suggested / ⚪ neutral / 🔴 low). A clip can surface because of a single moment inside it — the card then shows a "contains: …" reason. |
| **Curation** | Tick the clips you want to use in the sidebar **Bin** (Select-all available). Only ticked clips reach Selection. |
| **③ Selection** | An AI assistant editor: it takes your ticked clips as *candidates*, drops what doesn't serve your intent (and tells you why, as backup material), and lays the rest out as an explained timeline. |
| **④ Deliver** | Compiles the timeline into an FCP7 XML (`xmeml` v5) + JSON, written to `app/data/output/exports`. Order and trims are preserved exactly — nothing is re-edited here. |

### Two editing modes

Selection has exactly two modes. They differ only in the **unit of editing**:

| Mode | Unit | Target Duration |
|---|---|---|
| 🎞️ **Clip Assembly** — vlog, documentary, travel, BTS, montage | the whole clip, at its original length | not applicable (no trimming) |
| 🎯 **Moment Assembly** — build an edit from moments inside longer clips | a temporal event | optional |

In Moment Assembly, a target duration is an *optimisation*: the agent first picks enough meaningful moments, then shortens the least important ones to fit. Nothing is cut proportionally, and no moment is silently removed — if the overrun can't be absorbed, it's reported back to you.

### Output aspect ratio

You pick the output frame explicitly — **16:9 / 9:16 / 4:3 / 3:4 / 1:1** — and it carries through to the export. Source media is never modified: each clip is scaled to fit with its own aspect preserved, so footage is **never stretched and never cropped**. Where the ratios differ you get letterboxing/pillarboxing. (Selection may *prefer* footage that fits the frame; it never resizes anything.)

---

## Getting started

### Prerequisites

- **Python 3.10+**
- **FFmpeg / ffprobe** on `PATH` — required for ingest (probing, scene detection, frame sampling). Without it, ingest cannot build a real catalogue.
- An **OpenAI API key** — for vision tagging, embeddings, and the agents.

### Install

```bash
pip install -r requirements.txt
```

### Configure

Create a `.env` in the project root:

```dotenv
OPENAI_API_KEY=sk-...

# Optional — defaults shown
LLM_MODEL=gpt-5.5
LLM_MODEL_FALLBACK=gpt-5.5-mini
VISION_MODEL=gpt-5.5
EMBEDDING_MODEL=text-embedding-3-small
METADATA_DB_PATH=./app/data/mapo_catalogue.db
RAW_FOOTAGE_DIR=./app/data/raw_footage
PROCESSED_OUTPUT_DIR=./app/data/output
```

### Run

```bash
python main.py                            # the only supported entry point (spawns Streamlit)
# equivalently:
streamlit run app/ui/streamlit_app.py
```

### Then, in the browser

1. **① Ingest** — set the footage folder and click *Run Ingest Analysis*. Later phases stay locked until ingest actually indexes something.
2. **② Search** — type a query, review the ranked clips, ➕ the ones you like into the Bin.
3. **Curate** — fine-tune the Bin in the sidebar.
4. **③ Selection** — choose **Editing Mode**, **Output aspect ratio**, an optional **Target Duration** (Moment Assembly only), and describe your **Editing Intent** (style, emotion, pacing, purpose — not technical operations). The agent returns an explained timeline.
5. **④ Deliver** — enabled once a timeline exists. Exports the XML, then offers a **📂 Show in folder** button so you can drag it into Premiere (**File ▸ Import**).

A running transcript of what the agents did is available in the collapsed **🛠️ Debug log** at the bottom. **Reset Pipeline** clears everything.

---

## Project structure

```
main.py                              # entry point → spawns Streamlit
app/
  config.py                          # typed settings (loads .env once)
  orchestrator/
    production_orchestrator.py       # the spine — 4 explicit stage functions
  agents/
    ingest_agent.py                  # ① probe / scene-detect / vision-tag / catalogue
    search_agent.py                  # ② retrieval only
    selection_agent.py               # ③ intent-driven edit timeline
    delivery_agent.py                # ④ Premiere FCP7 XML compiler
  services/
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

