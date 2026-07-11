"""MAPO — Streamlit UI (the only operating entry point).

Layout: the Bin (media pool) lives in the LEFT SIDEBAR, under the footage folder,
because it is the shared data pool for the whole workflow; the workflow runs full-width
in the main area:

    SIDEBAR  ⚙️ settings · 📁 footage folder · 🗂️ Bin (Select-all · [checkbox | ▶] per clip)
    MAIN     ① Ingest → ② Search → ③ Selection → ④ Deliver → 🛠️ Debug log (collapsed)

Each stage renders its OWN output inline (the timeline under ③, the export result under
④); the running agent transcript is demoted to a collapsed debug log at the bottom.

Selection model:
    - The Bin's checkboxes ARE the selection (single source of truth); "Select all"
      ticks/unticks the whole pool in one click.
    - Search is a decision aid: it ranks matches into 🟡 suggested / ⚪ neutral /
      🔴 low and shows a reason + thumbnail; its ➕/➖ ticks/unticks the clip in the Bin.
    - Selection orchestrates ONLY the ticked Bin clips into an edit timeline.
    - Deliver compiles that timeline into a Premiere Pro–importable project (FCP7 XML),
      preserving clip order exactly — a pure export step, no re-editing.

Launch:
    python main.py   (or)   streamlit run app/ui/streamlit_app.py
"""

import base64
import json
import re
import streamlit as st
from pathlib import Path
from langchain_core.messages import HumanMessage

# Must be first Streamlit command
st.set_page_config(page_title="MAPO", page_icon="🎬", layout="wide")

SUGGESTION_MARK = {"suggested": "🟡", "neutral": "⚪", "low": "🔴"}
_TIERS = [("suggested", "🟡 Suggested"), ("neutral", "⚪ Neutral"), ("low", "🔴 Low")]


# ── Selection state (Bin checkbox is the single source of truth) ────────────────


def _bin_key(path: str) -> str:
    return f"bin_{path}"


def _set_bin(path: str, value: bool):
    """on_click callback: set a Bin clip's selected state before the rerun.

    Makes the Search ➕/➖ toggle persistent — it flips to ➖ and stays until the user
    clicks ➖, unticks the clip in the Bin, or clears the search marks.
    """
    st.session_state[_bin_key(path)] = value


def _clear_all_selections():
    """on_click callback: untick every Bin clip.

    Must run in a callback (not the script body): callbacks execute BEFORE the widgets
    are instantiated on the rerun, so assigning to the checkbox keys is allowed — doing
    it inline after the checkboxes render raises a StreamlitAPIException.
    """
    for k in [k for k in st.session_state if k.startswith("bin_")]:
        st.session_state[k] = False


def _toggle_select_all():
    """on_change callback for the 'Select all' checkbox in the Media Pool.

    Sets every Bin clip to the checkbox's new value. It fires ONLY when the editor
    toggles it (not on every rerun), so it never fights the editor's per-clip ticks —
    after a Select-all / Deselect-all they can freely adjust individual clips.
    """
    value = bool(st.session_state.get("select_all_bin", False))
    for c in st.session_state.get("bin_shots", []):
        st.session_state[_bin_key(c["file_path"])] = value


# ── Graph / service loaders ─────────────────────────────────────────────────


def get_mapo_agent():
    from app.orchestrator.production_orchestrator import mapo_agent
    return mapo_agent


def get_selection_agent():
    from app.orchestrator.production_orchestrator import selection_agent
    return selection_agent


def get_delivery_agent():
    from app.orchestrator.production_orchestrator import delivery_agent
    return delivery_agent


def _base_state(project_id, extra=None):
    state = {
        "project_id": project_id, "messages": [], "loaded_preferences": "",
        "ingested_files": [], "shot_metadata": [], "search_results": [],
        "search_candidates": [], "selected_candidates": [], "selected_shots": [],
        "recommendations": [], "edit_timeline": [], "delivery_output": [],
    }
    if extra:
        state.update(extra)
    return state


def _pid(project_id):
    try:
        return int(project_id)
    except (TypeError, ValueError):
        return project_id


# ── Backend calls ─────────────────────────────────────────────────────────────


def run_query(query: str, project_id: str, user_id: str, footage_dir: str):
    from app.config import settings
    settings.RAW_FOOTAGE_DIR = Path(footage_dir)
    mapo_agent = get_mapo_agent()
    config = {"configurable": {"thread_id": f"streamlit-{user_id}-{project_id}", "user_id": user_id}}
    augmented = query
    if any(kw in query.lower() for kw in ["scan", "index", "ingest", "catalogue", "import", "footage", "folder", "directory"]):
        augmented = f"{query}[System context: footage directory is {footage_dir}]"
    result = mapo_agent.invoke(
        _base_state(project_id, {"messages": [HumanMessage(content=augmented)]}), config=config)
    return result["messages"][-1].content


def _extract_plan(messages) -> dict | None:
    """Pull the structured timeline plan (from plan_timeline) out of the agent messages.

    plan_timeline embeds the plan as a ```json fenced block; we take the most recent one
    that parses and carries a "segments" list. Returns None if there is none.
    """
    for m in reversed(messages):
        content = getattr(m, "content", "") or ""
        if not isinstance(content, str):
            continue
        for match in re.findall(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL):
            try:
                obj = json.loads(match)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "segments" in obj:
                return obj
    return None


def run_selection(intent: str, selected_paths: list[str], project_id: str, user_id: str):
    """Run the Selection Agent. Returns (narration_text, structured_plan_or_None)."""
    selection_agent = get_selection_agent()
    config = {"configurable": {"thread_id": f"streamlit-select-{user_id}-{project_id}", "user_id": user_id}}
    clip_list = "\n".join(f"- {p}" for p in selected_paths)
    message = (
        f"My editing intent: {intent}\n\n"
        f"The editor has selected these clips for the edit (work ONLY with these):\n{clip_list}\n\n"
        "Fetch their details, decide the order and each clip's importance, then call "
        "plan_timeline (passing my editing-intent text so it can detect any target "
        "duration). The timeline's structure, pacing and number of steps are driven by my "
        "intent — do NOT assume a fixed narrative arc. For each step explain why the clip "
        "sits there, how it connects to the previous clip, and what it does for the pacing."
    )
    result = selection_agent.invoke(
        _base_state(project_id, {"messages": [HumanMessage(content=message)], "selected_candidates": selected_paths}),
        config=config)
    return result["messages"][-1].content, _extract_plan(result["messages"])


def run_delivery(plan: dict | None, timeline_text: str, ordered_paths: list[str],
                 project_id: str, user_id: str, sequence_name: str = "MAPO Edit"):
    """Compile the Selection timeline into a Premiere-importable project.

    Prefers the STRUCTURED plan (segments with in/out trims) — the Delivery Agent calls
    `compile_timeline_segments` and honours the trims + order exactly. Falls back to the
    ordered clip list (full clips) when no structured plan is available. Delivery
    re-orders NOTHING — it compiles what it is handed.
    """
    delivery_agent = get_delivery_agent()
    config = {"configurable": {"thread_id": f"streamlit-deliver-{user_id}-{project_id}", "user_id": user_id}}
    if plan and plan.get("segments"):
        message = (
            f"Compile the timeline into a Premiere Pro project named '{sequence_name}'.\n\n"
            "The Selection Agent produced these STRUCTURED timeline segments. Call "
            "compile_timeline_segments with segments_json set to EXACTLY this JSON — do "
            "not re-order, add, drop, or re-time any segment:\n\n"
            f"{json.dumps(plan)}"
        )
    else:
        clip_list = "\n".join(f"- {p}" for p in ordered_paths)
        message = (
            f"Compile the edit timeline below into a Premiere Pro project named "
            f"'{sequence_name}'.\n\n"
            f"The Selection Agent laid out this timeline (PRESERVE THIS ORDER EXACTLY):\n"
            f"{timeline_text}\n\n"
            f"The curated clip paths (ground-truth list, same intended order):\n{clip_list}\n\n"
            "Preview the timeline, then compile the FCP7 XML. Do not re-order or drop clips."
        )
    result = delivery_agent.invoke(
        _base_state(project_id, {"messages": [HumanMessage(content=message)],
                                 "selected_candidates": ordered_paths}),
        config=config)
    return result["messages"][-1].content


def run_and_record(query: str, project_id: str, user_id: str, footage_path: str):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("MAPO is processing..."):
        try:
            response = run_query(query, project_id, user_id, footage_path)
            st.session_state.messages.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {e}"})
            return None


def load_bin(project_id):
    """Load the full media pool (every catalogued clip, fixed file-name order)."""
    from app.services.retrieval_service import hybrid_search
    st.session_state.bin_shots = hybrid_search(_pid(project_id), top_k=1000)


def do_search(query: str, project_id: str):
    """Search = rank/mark matches. Sets 🟡 suggested only; never touches selection."""
    from app.services.retrieval_service import hybrid_search, expand_query, hoist_orientation
    # Pull any orientation word out of the raw query FIRST, so "find all horizontal
    # shots" runs as a pure orientation filter (no misleading %) and only genuine
    # content is sent through synonym expansion.
    residual, orientation = hoist_orientation(query, None)
    expanded = expand_query(residual) if residual and residual.strip() else None
    candidates = hybrid_search(_pid(project_id), keywords=expanded or None,
                               orientation=orientation)
    st.session_state.suggested = {
        c["file_path"]: c["relevance"]
        for c in candidates if c.get("suggestion") in ("suggested", "neutral")
    }
    st.session_state.search_results = candidates


# ── Preview / thumbnail helpers ─────────────────────────────────────────────


def _safe_mtime(path: str) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False)
def _thumbnail_bytes(path: str, mtime: float, duration: float):
    """Extract one representative JPEG frame (cached per file). None if unavailable."""
    try:
        if not Path(path).exists():
            return None
        from app.services.ffmpeg_service import extract_sample_frames_b64
        frames = extract_sample_frames_b64(path, duration or 0.0, count=1)
        return base64.b64decode(frames[0]) if frames else None
    except Exception:
        return None


def _thumb(c):
    return _thumbnail_bytes(c["file_path"], _safe_mtime(c["file_path"]), c.get("duration_seconds") or 0.0)


def _reason(c, tier: str) -> str:
    """A short 'why' for a search card: the clip's own vision description, so the editor
    reads what the shot actually SHOWS rather than a bag of keywords. Falls back to the
    keyword tags, then a generic note, when no description was recorded."""
    if tier == "low":
        return "weak semantic match"
    desc = (c.get("description") or "").strip()
    if desc:
        return desc
    kws = [k.strip() for k in (c.get("keywords") or "").split(",") if k.strip()]
    if kws:
        return " / ".join(kws[:3])
    return "semantic match"


def _meta_line(c: dict) -> str:
    bits = []
    if c.get("shot_type") and c["shot_type"] != "unclassified":
        bits.append(f"`{c['shot_type']}`")
    if c.get("duration_seconds"):
        bits.append(f"{c['duration_seconds']:.1f}s")
    if c.get("orientation"):
        bits.append(c["orientation"])
    if c.get("mood"):
        bits.append(f"mood: {c['mood']}")
    if c.get("group_size") and c["group_size"] not in ("unknown", "none"):
        bits.append(f"{c['group_size']} ({c.get('people_count')})")
    if c.get("keywords"):
        bits.append(str(c["keywords"])[:80])
    return " · ".join(bits)


def _render_preview(c):
    """Popover body: a frame, an inline player if possible, and full metadata."""
    path = c["file_path"]
    st.markdown(f"**{Path(path).name}**")
    if Path(path).exists():
        try:
            st.video(path)
        except Exception:
            st.caption("(inline playback unavailable for this codec)")
    else:
        st.caption("⚠ File not found on disk — preview unavailable.")
    st.caption(_meta_line(c))


# ── Section renderers ────────────────────────────────────────────────────────


def render_bin(project_id, locked):
    """Render the Bin section in the sidebar: checkbox + ▶ preview per clip."""
    st.subheader("🎞️ Media Pool")
    st.caption("☑ add to edit · ▶ preview")
    if locked:
        st.info("🔒 Run Ingest to populate the Bin.")
        return

    only_matches = st.checkbox("Only search matches", value=False,
                               help="Show only the last search's matches (and ticked clips)")

    shots = st.session_state.bin_shots
    if not shots:
        st.info("Bin is empty. Run Ingest to populate it.")
        return

    # Select-all toggle: ticks/unticks every clip when changed, then the editor can still
    # adjust individual clips (it only acts on toggle, never on rerun).
    st.checkbox("Select all", key="select_all_bin", on_change=_toggle_select_all,
                help="Tick or untick every clip in the pool")

    sugg = st.session_state.get("suggested", {})

    # FIXED file-name / ingestion order — ticking never moves a row.
    for c in shots:
        path = c["file_path"]
        name = Path(path).name
        if only_matches and path not in sugg and not st.session_state.get(_bin_key(path)):
            continue
        col_name, col_play = st.columns([0.8, 0.2])
        with col_name:
            st.checkbox(name, key=_bin_key(path))
        with col_play:
            with st.popover("▶"):
                _render_preview(c)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("↻ Refresh", use_container_width=True):
            load_bin(project_id)
            st.rerun()
    with b2:
        st.button("Clear ✕", use_container_width=True, help="Clear all selections",
                  on_click=_clear_all_selections)


def render_search_results():
    if not st.session_state.get("search_results"):
        return

    grouped = {"suggested": [], "neutral": [], "low": []}
    for c in st.session_state.search_results:
        grouped.setdefault(c.get("suggestion", "low"), []).append(c)

    # Each tier collapses into its own expander. Only 🟡 Suggested is open by default —
    # it holds the highest-confidence matches; ⚪ Neutral and 🔴 Low start collapsed.
    _EXPANDED = {"suggested": True, "neutral": False, "low": False}

    for tier, header in _TIERS:
        items = grouped.get(tier) or []
        if not items:
            continue
        with st.expander(f"{header} ({len(items)})", expanded=_EXPANDED.get(tier, False)):
            for c in items:
                path = c["file_path"]
                name = Path(path).name
                in_bin = bool(st.session_state.get(_bin_key(path)))
                rel = c.get("relevance")
                rel_str = f"{rel*100:.0f}%" if rel is not None else "—"
                dur = c.get("duration_seconds") or 0.0
                mark = SUGGESTION_MARK.get(tier, "⚪")

                card, action = st.columns([5, 1])
                with card:
                    st.markdown(f"{mark} **{name}** · {rel_str} · {dur:.1f}s")
                    st.caption(f"💡 {_reason(c, tier)}")
                    # Thumbnail evidence only for strong/medium matches.
                    if tier in ("suggested", "neutral"):
                        thumb = _thumb(c)
                        if thumb:
                            st.image(thumb, width=200)
                with action:
                    if in_bin:
                        st.button("➖", key=f"srch_tgl_{path}", help="Untick in Bin",
                                  on_click=_set_bin, args=(path, False))
                    else:
                        st.button("➕", key=f"srch_tgl_{path}", help="Tick in Bin",
                                  on_click=_set_bin, args=(path, True))
        st.markdown("")


# ── App ────────────────────────────────────────────────────────────────────


def main():
    st.title("🎬 MAPO — Multi-Agent Production Orchestrator")
    st.caption("MSc Computer Graphics, Vision & Imaging — UCL | Yichen Zheng")

    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("ingest_done", False)
    st.session_state.setdefault("bin_shots", [])
    st.session_state.setdefault("suggested", {})
    st.session_state.setdefault("search_results", [])
    st.session_state.setdefault("selection_output", "")
    st.session_state.setdefault("delivery_output_text", "")
    st.session_state.setdefault("last_timeline_plan", None)

    locked = not st.session_state.ingest_done

    # ── Sidebar: settings + the shared Bin (the workflow's data pool) ──────────
    with st.sidebar:
        st.header("⚙️ Project Settings")
        project_id = st.text_input("Project ID", value="1")
        user_id = st.text_input("Editor ID", value="editor_01")

        st.divider()

        st.subheader("📁 Footage")
        footage_path = st.text_input(
            "Full path to the folder containing your video files", value="./app/data/raw_footage")
        if footage_path:
            fdir = Path(footage_path)
            if fdir.exists():
                vids = []
                for ext in ["mp4", "mov", "avi", "mxf", "r3d", "braw"]:
                    vids.extend(fdir.rglob(f"*.{ext}"))
                st.success(f"✓ Found {len(vids)} video files")
            else:
                st.warning("⚠ Folder not found — check the path")

        st.divider()

        # Bin — its own section, the shared media pool.
        if st.session_state.ingest_done and not st.session_state.bin_shots:
            load_bin(project_id)
        render_bin(project_id, locked)

        st.divider()
        if st.button("🔄 Reset Pipeline", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # ── Main area (full width): the workflow ──────────────────────────────────
    # ① Ingest
    st.subheader("① Ingest")
    st.caption("Scan the footage directory, extract semantic metadata, build the catalogue.")
    if st.session_state.ingest_done:
        st.success("✅ Ingest completed. All clips are in the Bin (left sidebar).")
    run_ingest = st.button("🎬 Run Ingest Analysis", use_container_width=True,
                           help="Ingest Agent — scan, vision-tag, embed, catalogue")
    st.divider()

    # ② Search — decision cards
    st.subheader("② Search")
    st.caption("Ranks matches (🟡 suggested · ⚪ neutral · 🔴 low). ➕ ticks the clip in the Media Pool.")

    if locked:
        st.info("🔒 Locked — run Ingest first.")

    search_text = st.text_input("Search query", placeholder="e.g. energetic crowd celebration",
                                disabled=locked, key="search_query")
    s1, s2 = st.columns(2)
    with s1:
        if st.button("🔍 Search", use_container_width=True, disabled=locked):
            if search_text.strip():
                do_search(search_text, project_id)
            else:
                st.warning("Enter a search query first.")
    with s2:
        if st.button("✖ Clear marks", use_container_width=True, disabled=locked):
            st.session_state.suggested = {}
            st.session_state.search_results = []
    render_search_results()
    st.divider()

    # ③ Selection
    st.subheader("③ Selection")
    st.caption("State your editing intent. Orchestrates your ticked clips into a timeline.")
    selected_paths = [c["file_path"] for c in st.session_state.bin_shots
                      if st.session_state.get(_bin_key(c["file_path"]))]
    have_sel = bool(selected_paths)
    if locked:
        st.info("🔒 Locked — run Ingest first.")
    elif not have_sel:
        st.info("Tick at least one clip in the Media Pool (left sidebar).")
    else:
        st.caption(f"{len(selected_paths)} clip(s) selected.")
    intent_text = st.text_input("Editing intent", placeholder="e.g. Fast-paced 30s football promo",
                                disabled=locked or not have_sel, key="intent")
    run_select = st.button("🎬 Generate Edit Timeline", use_container_width=True,
                           disabled=locked or not have_sel,
                           help="Selection Agent — lays out an intent-driven timeline (no fixed arc)")

    # Selection output renders HERE, inside the Selection section.
    if st.session_state.get("selection_output"):
        plan = st.session_state.get("last_timeline_plan")
        if plan:
            n_steps = len(plan.get("segments", []))
            if plan.get("mode") == "timed":
                st.caption(f"⏱️ TIMED MODE · target {plan.get('target_seconds')}s · "
                           f"actual {plan.get('total_seconds')}s · {n_steps} steps")
            elif plan.get("mode") == "trim":
                st.caption(f"✂️ TRIM MODE · first {plan.get('head_trim')}s + last "
                           f"{plan.get('tail_trim')}s off each clip · "
                           f"{plan.get('total_seconds')}s · {n_steps} steps")
            else:
                st.caption(f"🎞️ FULL CLIP MODE · {plan.get('total_seconds')}s · "
                           f"{n_steps} steps (no trimming)")
        with st.container(border=True):
            st.markdown(st.session_state.selection_output)
    st.divider()

    # ④ Deliver — compile the timeline into a Premiere-importable project
    st.subheader("④ Deliver")
    st.caption("Compiles the edit timeline into a Premiere Pro–importable project "
               "(FCP7 XML + JSON). Preserves clip order exactly — no re-editing.")
    have_timeline = bool(st.session_state.get("last_timeline"))
    if locked:
        st.info("🔒 Locked — run Ingest first.")
    elif not have_timeline:
        st.info("Generate an edit timeline in ③ Selection first.")
    else:
        st.caption(f"Timeline ready · {len(st.session_state.get('last_timeline_paths', []))} clip(s).")
    seq_name = st.text_input("Sequence name", value="MAPO Edit",
                             disabled=locked or not have_timeline, key="seq_name")
    run_deliver = st.button("📦 Export to Premiere (FCP7 XML)", use_container_width=True,
                            disabled=locked or not have_timeline,
                            help="Delivery Agent — compiles the timeline into an importable project file")

    # Delivery output renders HERE, inside the Deliver section.
    if st.session_state.get("delivery_output_text"):
        with st.container(border=True):
            st.markdown(st.session_state.delivery_output_text)
    st.divider()

    # ── Resolve actions ──────────────────────────────────────────────────────
    quick_query = None
    if run_ingest:
        quick_query = f"Run ingest analysis on the footage directory at {footage_path} "
    elif run_select:
        if intent_text.strip():
            with st.spinner("Selection Agent is orchestrating the edit..."):
                try:
                    response, plan = run_selection(intent_text.strip(), selected_paths, project_id, user_id)
                    # Render inline in the Selection section + keep a debug-log copy.
                    st.session_state.selection_output = response
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"[Selection] intent: {intent_text.strip()} · {len(selected_paths)} clip(s)"})
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    # Capture the timeline + structured plan so Delivery can compile it
                    # (trims + order preserved) without re-ordering anything.
                    st.session_state.last_timeline = response
                    st.session_state.last_timeline_plan = plan
                    st.session_state.last_timeline_paths = (
                        [s["file_path"] for s in plan["segments"]] if plan and plan.get("segments")
                        else selected_paths)
                    # A fresh timeline invalidates any previous export.
                    st.session_state.delivery_output_text = ""
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {e}"})
            st.rerun()
        else:
            st.warning("Enter your editing intent first.")
    elif run_deliver:
        with st.spinner("Delivery Agent is compiling the Premiere project..."):
            try:
                response = run_delivery(
                    st.session_state.get("last_timeline_plan"),
                    st.session_state.get("last_timeline", ""),
                    st.session_state.get("last_timeline_paths", []),
                    project_id, user_id, (seq_name.strip() or "MAPO Edit"))
                st.session_state.delivery_output_text = response
                st.session_state.messages.append({
                    "role": "user", "content": f"[Deliver] compile '{seq_name.strip() or 'MAPO Edit'}'"})
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {e}"})
        st.rerun()

    # ── Debug log (collapsed) ─────────────────────────────────────────────────
    # The primary Selection/Delivery outputs render inside their own sections above;
    # this is a demoted, collapsed raw log of every agent exchange for debugging.
    with st.expander(f"🛠️ Debug log ({len(st.session_state.messages)} messages)", expanded=False):
        if not st.session_state.messages:
            st.caption("No agent activity yet.")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if quick_query:
        response = run_and_record(quick_query, project_id, user_id, footage_path)
        if run_ingest and response is not None:
            st.session_state.ingest_done = True
            load_bin(project_id)
        st.rerun()

    # Chat-bar style input: a wide text box + a compact ➤ send button on one row.
    # A form is used (not st.chat_input) because st.chat_input is pinned to the bottom
    # and auto-focuses, which scrolls the whole page down on load; a form does not.
    with st.form("followup", clear_on_submit=True, border=False):
        c_in, c_send = st.columns([8, 1])
        with c_in:
            prompt = st.text_input("Ask MAPO a follow-up...", label_visibility="collapsed",
                                   placeholder="Ask MAPO a follow-up...")
        with c_send:
            sent = st.form_submit_button("➤", use_container_width=True)
        if sent and prompt.strip():
            run_and_record(prompt, project_id, user_id, footage_path)
            st.rerun()


if __name__ == "__main__":
    main()
