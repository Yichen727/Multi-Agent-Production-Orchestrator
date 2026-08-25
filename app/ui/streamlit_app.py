"""MAPO — Streamlit UI.

The UI is a thin presentation layer over the four-stage pipeline:

    Sidebar: Settings · Footage · Media Pool
    Main:    ① Ingest → ② Search → ③ Selection → ④ Deliver

The Media Pool is the single source of truth for selected clips.
Selection produces a structured timeline, and Delivery deterministically
compiles that plan into a Premiere Pro–compatible FCP7 XML project.

Launch:
    python main.py
    streamlit run app/ui/streamlit_app.py
"""

import base64
import subprocess
import sys
import streamlit as st
from pathlib import Path

# Must be first Streamlit command
st.set_page_config(page_title="MAPO", page_icon="🎬", layout="wide")

SUGGESTION_MARK = {"suggested": "🟡", "neutral": "⚪", "low": "🔴"}
_TIERS = [("suggested", "🟡 Suggested"), ("neutral", "⚪ Neutral"), ("low", "🔴 Low")]

# The ONLY two editing modes the Selection stage exposes: label → (mode id, caption).
# They differ in the UNIT of editing — whole clips vs moments inside clips. Duration
# control is a property of Moment Assembly alone, not a mode of its own.
EDITING_MODES = {
    "🎞️ Clip Assembly": ("clip_assembly",
                          "Combine complete clips."),
    "🎯 Moment Assembly": ("moment_assembly",
                           "Select moments from within clips."),
}

_DURATION_STATUS = {
    "on_target": "on target",
    "under_target": "under target",
    "over_target": "over target — kept for content",
}

# Output aspect ratio — an explicit OUTPUT SPEC the editor picks, never inferred from the
# editing prompt. All five propagate identically (Selection → plan → Delivery). Delivery
# scales clips to FIT the frame: aspect preserved, no cropping, no stretching —
# letterbox/pillarbox where the source and target ratios differ.
ASPECT_CHOICES = ["16:9", "9:16", "4:3", "3:4", "1:1"]


# ── Selection state (Bin checkbox is the single source of truth) ────────────────


def _bin_key(path: str) -> str:
    return f"bin_{path}"


def _set_bin(path: str, value: bool):
    """on_click callback: set a Bin clip's selected state before the rerun.

    Makes the Search ➕/➖ toggle persistent — it flips to ➖ and stays until the user
    clicks ➖ or unticks the clip in the Bin.
    """
    st.session_state[_bin_key(path)] = value


def _set_bin_many(paths: list[str], value: bool):
    """on_click callback: tick/untick a WHOLE tier of search results at once.

    Same mechanism as ``_set_bin``, applied to every path in one go — that is what makes
    "add all 🟡 Suggested to the Media Pool" a single click instead of one ➕ per card.
    """
    for path in paths:
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


def _backup_headline(item: dict) -> str:
    """``name (start–end)`` for one backup-material item — an editor's clip reference.

    The timecodes come from the catalogue (the Selection tool resolved them); an item that
    matched nothing simply shows its raw reference rather than an invented range.
    """
    name = item.get("name") or item.get("ref") or "(unidentified)"
    start, end = item.get("start_seconds"), item.get("end_seconds")
    if start is None or end is None:
        return name
    return f"{name} ({start:.2f}s–{end:.2f}s)"


def _backup_span(item: dict) -> str:
    """The MEASURED extent of one backup item, phrased for whichever unit it is.

    A rejected moment reads as ``in → out · length``; a rejected whole clip reads as
    ``full clip · length``; an item whose length was never measured says so rather than
    display a 0.00s–0.00s range that looks like a real trim.
    """
    start, end = item.get("start_seconds"), item.get("end_seconds")
    if end is None:
        return "length unmeasured"
    if not item.get("event_id") and not start:
        return f"full clip · {end:.1f}s"
    start = start or 0.0
    return f"{start:.2f}s → {end:.2f}s · {end - start:.1f}s"


def _render_backup_item(index: int, item: dict) -> None:
    """One backup-material entry as its own card, in a fixed scannable order.

    Header (rank · unit icon · file name · measured timecodes) → the moment's own action
    label → the two editorial lines the agent owns: why it is not in the edit, and what it
    could still serve. Grouped near-duplicates hang off the card as one line instead of
    each claiming a slot, so the shortlist stays a shortlist.
    """
    with st.container(border=True):
        kind = "🎯" if item.get("event_id") else "🎞️"
        name = item.get("name") or item.get("ref") or "(unidentified)"
        st.markdown(f"{kind} **{index}. {name}**  ·  {_backup_span(item)}")
        if item.get("label"):
            st.caption(item["label"])
        if not item.get("file_path"):
            st.caption("⚠ No catalogued clip matched this reference — shown as given.")
        rows = [f"**Why not used** · {item.get('reason') or '*not stated*'}"]
        if item.get("suggested_use"):
            rows.append(f"**Could be used for** · {item['suggested_use']}")
        st.markdown("  \n".join(rows))       # two spaces = one tight line break
        group = item.get("also_details") or []
        if group:
            st.caption(f"⧉ Same call for {len(group)} near-duplicate(s): "
                       + "; ".join(_backup_headline(g) for g in group))


def _strip_backup_section(report: str) -> str:
    """Drop any prose "not used / backup material" block from the agent's report.

    Backup material is rendered ONCE, from the plan (resolved names + measured timecodes)
    in its own expander. The agent is told not to repeat it in prose, but a model can
    always drift, so the duplicate is removed at render time rather than shown twice.
    Anything from a later ``Notes:``/heading line onwards is kept — only the list goes.
    """
    def _is_heading(ln: str) -> bool:
        low = ln.strip().lstrip("#*_ ").lower()
        return ("🗂" in ln or "backup material" in low
                or low.startswith(("not used", "unused", "alternative material")))

    lines = report.splitlines()
    start = next((i for i, ln in enumerate(lines) if _is_heading(ln)), None)
    if start is None:
        return report
    end = next((j for j in range(start + 1, len(lines))
                if lines[j].lstrip().lower().startswith(("notes:", "🎬", "###", "## "))),
               len(lines))
    # Also swallow a "---" separator that only existed to introduce the dropped block.
    while start > 0 and lines[start - 1].strip() in ("", "---", "***", "___"):
        start -= 1
    return "\n".join(lines[:start] + lines[end:]).strip()


def _reveal_in_file_manager(target: str) -> str | None:
    """Open the OS file manager with ``target`` selected, ready to drag into an NLE.

    Returns ``None`` on success, or a short reason it could not be opened. This works only
    when the browser and the Streamlit server are the SAME machine — which is the supported
    ``python main.py`` local setup. A remote server cannot open a window on the viewer's
    machine, so the caller reports the path instead of pretending it worked.
    """
    path = Path(target)
    if not path.exists():
        return "the file is no longer on disk"
    try:
        if sys.platform.startswith("win"):
            # Explorer wants the flag and the path glued together; it also exits non-zero
            # even on success, so fire-and-forget rather than check the return code.
            subprocess.Popen(["explorer", f"/select,{path}"])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path.parent)])   # no "select" equivalent
    except OSError as e:
        return str(e)
    return None


def _render_export_actions(result) -> None:
    """One click to reveal the exported project file, so it can be dragged into Premiere.

    Gated on the COMPILER's own structured ``DeliveryResult`` (status success + a real path
    that still exists), never on the agent's prose: a refused or failed compile must not
    offer to open a file that was never written.
    """
    xml_path = getattr(result, "xml_path", None) if result else None
    if not (xml_path and getattr(result, "status", None) == "success"):
        return
    if not Path(xml_path).exists():
        st.caption(f"⚠ The exported file is no longer at its recorded path: {xml_path}")
        return
    name = Path(xml_path).name
    if st.button(f"📂 Show {name} in folder", key="reveal_export",
                 use_container_width=True,
                 help="Opens the export folder with the file selected — drag it straight "
                      "into Premiere Pro (works when the app runs on this machine)"):
        problem = _reveal_in_file_manager(xml_path)
        if problem:
            st.warning(f"Could not open the folder ({problem}). The file is at: {xml_path}")
        else:
            st.caption(f"Opened {Path(xml_path).parent}")


# ── Backend calls (thin wrappers over the orchestrator stages) ─────────────────


def run_selection(intent: str, selected_paths: list[str], project_id: str, user_id: str,
                  editing_mode: str = "clip_assembly", target_seconds: float | None = None,
                  aspect_ratio: str = ""):
    """③ Selection — returns (narration_text, structured_plan_or_None).

    The editor picks the editing mode: CLIP ASSEMBLY (complete clips, original durations,
    no target duration) or MOMENT ASSEMBLY (moments inside clips, optional target
    duration), plus the OUTPUT ASPECT RATIO (a delivery spec that rides along the plan to
    Delivery). Everything else — which candidates belong, the order, the pacing — is the
    Selection Agent's editorial reasoning.
    """
    return _orch().run_selection(intent, selected_paths, project_id, user_id,
                                 editing_mode=editing_mode, target_seconds=target_seconds,
                                 aspect_ratio=aspect_ratio)


def run_delivery(plan: dict, project_id: str, user_id: str, sequence_name: str = "MAPO Edit"):
    """④ Delivery — compile the STRUCTURED plan (required; no Bin-order fallback, H-04).

    Returns ``(agent_text, DeliveryResult | None)``; the structured result carries the
    written artefact paths so the UI can reveal the file on disk.
    """
    return _orch().run_delivery(plan, project_id, user_id, sequence_name=sequence_name)


def load_bin(project_id):
    """Load the full media pool (every catalogued clip, fixed file-name order)."""
    from app.services.retrieval_service import hybrid_search
    st.session_state.bin_shots = hybrid_search(_pid(project_id), top_k=1000)


def do_search(query: str, project_id: str, user_id: str):
    """② Search — rank/mark matching CLIPS via the orchestrator's Search stage.

    Delegates to ``run_search``, which invokes the Search Agent (falling back to direct
    hybrid retrieval when the LLM is unavailable); each candidate carries its own
    🟡 suggested / ⚪ neutral / 🔴 low marker, which is what the result cards group by. The
    unit is always the whole clip; a match driven by a moment inside a clip carries a
    ``matched_event`` hint shown as the card's reason. Never touches selection.
    """
    st.session_state.search_results = _orch().run_search(query, project_id, user_id)


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

    shots = st.session_state.bin_shots
    if not shots:
        st.info("Bin is empty. Run Ingest to populate it.")
        return

    # Select-all toggle: ticks/unticks every clip when changed, then the editor can still
    # adjust individual clips (it only acts on toggle, never on rerun).
    st.checkbox("Select all", key="select_all_bin", on_change=_toggle_select_all,
                help="Tick or untick every clip in the pool")

    # FIXED file-name / ingestion order — ticking never moves a row. The pool always
    # shows the WHOLE catalogue: filtering it by the last search would hide clips the
    # editor may still want, and Search already has its own tiered result list.
    for c in shots:
        path = c["file_path"]
        name = Path(path).name
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
            # Bulk controls for the whole tier — one click instead of one ➕ per card,
            # which matters when a query returns dozens of matches. They write the SAME
            # bin_<path> keys as the per-card ➕/➖, so the sidebar Bin, the card icons and
            # ③ Selection's count all stay in sync automatically.
            paths_in_tier = [c["file_path"] for c in items]
            bulk_add, bulk_del, _ = st.columns([1.4, 1.4, 3.2])
            with bulk_add:
                st.button(f"➕ Add all ({len(items)})", key=f"srch_add_{tier}",
                          use_container_width=True,
                          help="Tick every clip in this tier in the Media Pool",
                          on_click=_set_bin_many, args=(paths_in_tier, True))
            with bulk_del:
                st.button(f"➖ Remove all ({len(items)})", key=f"srch_del_{tier}",
                          use_container_width=True,
                          help="Untick every clip in this tier in the Media Pool",
                          on_click=_set_bin_many, args=(paths_in_tier, False))

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
    st.session_state.setdefault("search_results", [])
    st.session_state.setdefault("selection_output", "")
    st.session_state.setdefault("delivery_output_text", "")
    st.session_state.setdefault("delivery_result", None)
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
    st.caption("Scan and catalogue the footage.")
    if st.session_state.ingest_done:
        st.success("✅ Ingest completed. All clips are in the Media Pool.")
    run_ingest = st.button("🎬 Run Ingest Analysis", use_container_width=True,
                           help="Ingest Agent — scan, vision-tag, embed, catalogue")
    st.divider()

    # ② Search — decision cards (always clip-level; choosing a moment is Selection's job)
    st.subheader("② Search")
    st.caption("Find matching clips and add them to the Media Pool.")

    if locked:
        st.info("🔒 Locked — run Ingest first.")

    search_text = st.text_input("Search query", placeholder="e.g. Tech products",
                                disabled=locked, key="search_query")
    s1, s2 = st.columns(2)
    with s1:
        if st.button("🔍 Search", use_container_width=True, disabled=locked):
            if search_text.strip():
                do_search(search_text, project_id, user_id)
            else:
                st.warning("Enter a search query first.")
    with s2:
        if st.button("✖ Clear results", use_container_width=True, disabled=locked,
                     help="Drop the last search's result cards (Media Pool ticks stay)"):
            st.session_state.search_results = []
    render_search_results()
    st.divider()

    # ③ Selection — two editing modes, nothing else to configure
    st.subheader("③ Selection")
    st.caption("Choose an editing mode and describe your intent.")
    selected_paths = [c["file_path"] for c in st.session_state.bin_shots
                      if st.session_state.get(_bin_key(c["file_path"]))]
    have_sel = bool(selected_paths)
    if locked:
        st.info("🔒 Locked — run Ingest first.")
    elif not have_sel:
        st.info("Tick at least one clip in the Media Pool (left sidebar).")
    else:
        st.caption(f"{len(selected_paths)} clip(s) selected.")

    sel_disabled = locked or not have_sel
    mode_label = st.radio(
        "Editing mode",
        options=list(EDITING_MODES),
        captions=[EDITING_MODES[k][1] for k in EDITING_MODES],
        horizontal=True, disabled=sel_disabled, key="editing_mode_label")
    is_moment = EDITING_MODES[mode_label][0] == "moment_assembly"

    o1, o2 = st.columns([2, 2])
    with o1:
        aspect_ratio = st.selectbox(
            "Output aspect ratio", options=ASPECT_CHOICES, index=0,
            disabled=sel_disabled, key="aspect_ratio",
            )
    with o2:
        # Target Duration applies to MOMENT ASSEMBLY only — Clip Assembly keeps every
        # clip's original length, so there is nothing to optimise. Default is N/A.
        target_seconds = st.number_input(
            "Target duration (seconds)", min_value=1.0, max_value=7200.0, step=5.0,
            value=None, placeholder="N/A", format="%.0f",
            disabled=sel_disabled or not is_moment, key="target_seconds",
            help=("Optional. The agent selects the moments the intent needs first, then "
                  "compresses the lower-value ones to fit — content is never dropped just "
                  "to hit the number." if is_moment
                  else "Not available in Clip Assembly Mode."))
    if not is_moment:
        target_seconds = None

    intent_text = st.text_input(
        "Editing intent", key="intent", disabled=sel_disabled,
        placeholder="e.g. Fast-paced tech product promo, showcasing various functions",
        help="Describe style, emotion, pacing and purpose.")
    run_select = st.button("🎬 Generate Edit Timeline", use_container_width=True,
                           disabled=sel_disabled,
                           help="Selection Agent — an assistant editor: it curates, orders "
                                "and explains (no fixed narrative arc)")

    # Selection output renders HERE, inside the Selection section.
    if st.session_state.get("selection_output"):
        plan = st.session_state.get("last_timeline_plan")
        if plan:
            n_steps = len(plan.get("segments", []))
            if plan.get("mode") == "moment_assembly":
                head = (f"🎯 MOMENT ASSEMBLY · {n_steps} moment(s) · "
                        f"{plan.get('total_seconds')}s")
                if plan.get("target_seconds"):
                    head += (f" · target {plan.get('target_seconds')}s "
                             f"({_DURATION_STATUS.get(plan.get('duration_status'), '')})")
                comp = plan.get("compression") or {}
                if comp.get("applied"):
                    head += (f" · optimised −{comp.get('absorbed_seconds')}s across "
                             f"{comp.get('trimmed_count')} lower-value moment(s)")
                st.caption(head)
            else:
                st.caption(f"🎞️ CLIP ASSEMBLY · {n_steps} complete clip(s) · "
                           f"{plan.get('total_seconds')}s (no trimming)")
            if plan.get("aspect_ratio"):
                st.caption(f"🖼️ Output frame: {plan['aspect_ratio']}")
            # Ordering is the stage's core editorial act — surface the declared shape, and
            # flag an order that merely reproduces how the material was listed.
            if plan.get("ordering_strategy"):
                st.caption(f"🧭 Ordering: {plan['ordering_strategy']}")
            if (plan.get("order_check") or {}).get("unchanged"):
                st.caption(f"⚠ Order matches {plan['order_check']['reference']} — "
                           "check the agent's reasoning below for why that shape fits.")
        with st.container(border=True):
            st.markdown(_strip_backup_section(st.session_state.selection_output))
        # Backup material, identified the way an editor reads it — source file + real
        # timecodes, never a bare event id. This expander is the SINGLE place it appears.
        dropped = (plan or {}).get("excluded") or []
        if dropped:
            with st.expander(f"🗂️ Not used — backup material ({len(dropped)})",
                             expanded=False):
                st.caption("The alternatives the agent weighed and rejected, strongest "
                           "first — not every unused clip. None of these is in the export.")
                for i, x in enumerate(dropped, 1):
                    _render_backup_item(i, x)
                omitted = (plan or {}).get("excluded_omitted", 0)
                if omitted:
                    st.caption(f"＋{omitted} further item(s) were trimmed — the shortlist is "
                               "capped at the strongest alternatives.")
    st.divider()

    # ④ Deliver — compile the timeline into a Premiere-importable project
    st.subheader("④ Deliver")
    st.caption("Export the generated timeline as a Premiere Pro–compatible project.")
    # H-04: Delivery needs the STRUCTURED plan (ordered segments) from Selection — not
    # merely some timeline text. Without a structured plan there is no defined edit order
    # to compile and NO media-pool-order fallback, so Deliver stays disabled.
    _plan = st.session_state.get("last_timeline_plan")
    have_plan = bool(_plan and _plan.get("segments"))
    if locked:
        st.info("🔒 Locked — run Ingest first.")
    elif not have_plan:
        st.info("Generate a structured edit timeline first.")
    else:
        st.caption(f"Timeline ready · {len(_plan['segments'])} segment(s) · "
                   f"built by {_plan.get('mode', 'clip_assembly').replace('_', ' ')} · "
                   f"{_plan.get('total_seconds')}s"
                   + (f" · output frame {_plan['aspect_ratio']}"
                      if _plan.get("aspect_ratio") else "")
                   )
    seq_name = st.text_input("Sequence name", value="MAPO Edit",
                             disabled=locked or not have_plan, key="seq_name")
    run_deliver = st.button("📦 Export Project", use_container_width=True,
                            disabled=locked or not have_plan,
                            help="Delivery Agent — compiles the ordered timeline segments")

    # Delivery output renders HERE, inside the Deliver section.
    if st.session_state.get("delivery_output_text"):
        with st.container(border=True):
            st.markdown(st.session_state.delivery_output_text)
        _render_export_actions(st.session_state.get("delivery_result"))
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
            mode_id = EDITING_MODES[mode_label][0]
            with st.spinner("Selection Agent is orchestrating the edit..."):
                try:
                    response, plan = run_selection(
                        intent_text.strip(), selected_paths, project_id, user_id,
                        editing_mode=mode_id, target_seconds=target_seconds,
                        aspect_ratio=aspect_ratio)
                    # Render inline in the Selection section + keep a debug-log copy.
                    st.session_state.selection_output = response
                    st.session_state.messages.append({
                        "role": "user",
                        "content": (f"[Selection] {mode_label} · {aspect_ratio} · target "
                                    f"{target_seconds or 'N/A'} · intent: {intent_text.strip()} "
                                    f"· {len(selected_paths)} candidate clip(s)")})
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
                # The DeliveryResult is the compiler's own record of what it wrote — it is
                # what gates the "show in folder" action, not the agent's narration.
                response, res = run_delivery(
                    st.session_state.get("last_timeline_plan"),
                    project_id, user_id, (seq_name.strip() or "MAPO Edit"))
                st.session_state.delivery_output_text = response
                st.session_state.delivery_result = res
                st.session_state.messages.append({
                    "role": "user", "content": f"[Deliver] compile '{seq_name.strip() or 'MAPO Edit'}'"})
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                # No artefact was written — drop any earlier one so the reveal button can't
                # point at a stale export.
                st.session_state.delivery_result = None
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
