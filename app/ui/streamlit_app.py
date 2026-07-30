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
import streamlit as st
from pathlib import Path

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


# ── Orchestrator (the single, explicit pipeline path — audit H-06) ─────────────
#
# The UI is a thin presentation layer: every stage goes through the pipeline
# orchestrator's explicit stage functions (no LLM supervisor, no bypass). These are
# lazy-imported so Streamlit reruns don't recompile the graphs each time.


def _orch():
    from app.orchestrator import production_orchestrator as orch
    return orch


def _pid(project_id):
    try:
        return int(project_id)
    except (TypeError, ValueError):
        return project_id


# ── Backend calls (thin wrappers over the orchestrator stages) ─────────────────


def run_selection(intent: str, selected_paths: list[str], project_id: str, user_id: str):
    """③ Selection — returns (narration_text, structured_plan_or_None).

    The Selection Agent decides for itself, from the editing intent, whether to build a
    whole-clip timeline or a moment-precise one (trimmed to detected event boundaries).
    """
    return _orch().run_selection(intent, selected_paths, project_id, user_id)


def run_delivery(plan: dict, project_id: str, user_id: str, sequence_name: str = "MAPO Edit"):
    """④ Delivery — compile the STRUCTURED plan (required; no Bin-order fallback, H-04)."""
    return _orch().run_delivery(plan, project_id, user_id, sequence_name=sequence_name)


def load_bin(project_id):
    """Load the full media pool (every catalogued clip, fixed file-name order)."""
    from app.services.retrieval_service import hybrid_search
    st.session_state.bin_shots = hybrid_search(_pid(project_id), top_k=1000)


def do_search(query: str, project_id: str, user_id: str):
    """② Search — rank/mark matching CLIPS via the orchestrator's Search stage.

    Delegates to ``run_search``, which invokes the Search Agent (falling back to direct
    hybrid retrieval when the LLM is unavailable), then marks 🟡 suggested / ⚪ neutral
    clips in the Bin. The unit is always the whole clip; a match driven by a moment inside
    a clip carries a ``matched_event`` hint shown as the card's reason. Never touches
    selection.
    """
    candidates = _orch().run_search(query, project_id, user_id)
    st.session_state.suggested = {
        c["file_path"]: c.get("relevance")
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
    """A short 'why' for a search card. When a MOMENT inside the clip drove the match
    (event-aware recall), lead with that moment — it explains why this clip surfaced.
    Otherwise fall back to the clip's own vision description, then keyword tags, so the
    editor reads what the shot actually SHOWS rather than a bag of keywords."""
    if tier == "low":
        return "weak semantic match"
    ev = c.get("matched_event")
    if ev and (ev.get("action") or "").strip():
        span = f"{ev.get('start_seconds', 0):.0f}–{ev.get('end_seconds', 0):.0f}s"
        return f"contains: {ev['action'].strip()} (~{span})"
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

    # ② Search — decision cards (always clip-level; choosing a moment is Selection's job)
    st.subheader("② Search")
    st.caption("Finds matching clips (🟡 suggested · ⚪ neutral · 🔴 low). Recall is "
               "event-aware — a clip surfaces when a moment inside it matches, shown as the "
               "card's reason. ➕ ticks the clip in the Media Pool.")

    if locked:
        st.info("🔒 Locked — run Ingest first.")

    search_text = st.text_input("Search query", placeholder="e.g. energetic crowd celebration",
                                disabled=locked, key="search_query")
    s1, s2 = st.columns(2)
    with s1:
        if st.button("🔍 Search", use_container_width=True, disabled=locked):
            if search_text.strip():
                do_search(search_text, project_id, user_id)
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
    st.caption("State your editing intent. Orchestrates your ticked clips into a timeline — "
               "the agent cuts to precise event moments when your intent calls for it.")
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
            if plan.get("mode") == "events":
                st.caption(f"🎯 EVENTS MODE · trimmed to {n_steps} detected moment(s) · "
                           f"{plan.get('total_seconds')}s total")
            elif plan.get("mode") == "timed":
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
    # H-04: Delivery needs the STRUCTURED plan (ordered segments) from Selection — not
    # merely some timeline text. Without a structured plan there is no defined edit order
    # to compile and NO media-pool-order fallback, so Deliver stays disabled.
    _plan = st.session_state.get("last_timeline_plan")
    have_plan = bool(_plan and _plan.get("segments"))
    if locked:
        st.info("🔒 Locked — run Ingest first.")
    elif not have_plan:
        st.info("Generate a structured edit timeline in ③ Selection first "
                "(Delivery compiles the ordered segments — there is no media-pool fallback).")
    else:
        st.caption(f"Timeline ready · {len(_plan['segments'])} segment(s) · "
                   f"mode: {_plan.get('mode', 'full')}.")
    seq_name = st.text_input("Sequence name", value="MAPO Edit",
                             disabled=locked or not have_plan, key="seq_name")
    run_deliver = st.button("📦 Export to Premiere (FCP7 XML)", use_container_width=True,
                            disabled=locked or not have_plan,
                            help="Delivery Agent — compiles the ordered timeline segments")

    # Delivery output renders HERE, inside the Deliver section.
    if st.session_state.get("delivery_output_text"):
        with st.container(border=True):
            st.markdown(st.session_state.delivery_output_text)
    st.divider()

    # ── Resolve actions (each button runs exactly one explicit pipeline stage) ──
    if run_ingest:
        with st.spinner("Ingest Agent is scanning, tagging and cataloguing footage..."):
            try:
                from app.services.database_service import get_catalogued_paths
                # run_ingest passes the footage directory on state (no global mutation,
                # H-07) and returns a structured IngestResult.
                response, res = _orch().run_ingest(footage_path, project_id, user_id)
                st.session_state.messages.append(
                    {"role": "user", "content": f"[Ingest] {footage_path}"})
                st.session_state.messages.append({"role": "assistant", "content": response})
                # C-08: unlock later phases ONLY on a real structured success — status
                # success/partial_success AND clips indexed AND an independent catalogue
                # check. A non-empty agent message (e.g. "Directory not found") never unlocks.
                catalogued = len(get_catalogued_paths(_pid(project_id)))
                if (res is not None and res.status in ("success", "partial_success")
                        and res.indexed_count > 0 and catalogued > 0):
                    st.session_state.ingest_done = True
                    load_bin(project_id)
                    if res.status == "partial_success":
                        warn = res.message + ("\n" + "; ".join(res.warnings) if res.warnings else "")
                        st.warning(warn)
                else:
                    reason = (res.message if res is not None else
                              "the agent never completed an ingest (no structured result)")
                    st.error("Ingest did not succeed — Search/Selection/Delivery stay "
                             f"locked: {reason}")
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {e}"})
        st.rerun()
    elif run_select:
        if intent_text.strip():
            with st.spinner("Selection Agent is orchestrating the edit..."):
                try:
                    response, plan = run_selection(
                        intent_text.strip(), selected_paths, project_id, user_id)
                    # Render inline in the Selection section + keep a debug-log copy.
                    st.session_state.selection_output = response
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"[Selection] intent: {intent_text.strip()} · {len(selected_paths)} clip(s)"})
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    # The STRUCTURED plan is the ONLY thing Delivery consumes (audit
                    # H-04): there is no Bin-order fallback. If Selection produced no
                    # structured plan, Delivery stays disabled and asks for a re-run.
                    st.session_state.last_timeline = response
                    st.session_state.last_timeline_plan = plan
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
                # H-04: Delivery is driven ONLY by the structured plan's ordered segments.
                response = run_delivery(
                    st.session_state.get("last_timeline_plan"),
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


if __name__ == "__main__":
    main()
