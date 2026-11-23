"""Atomized editing workspace. Only the active page renders or performs its work."""

import base64
import json
import math
from pathlib import Path

import streamlit as st

from app.branding import PRODUCT_NAME, ACCENT_COLOR, DEFAULT_SEQUENCE_NAME, EXPORT_PREFIX
from app.services import database_service as catalogue
from app.services.workspace_state import reconcile_workspace, set_candidates, save_edit_plan
from app.services.plan_contract import validate_edit_plan

st.set_page_config(page_title=PRODUCT_NAME, page_icon="🎬", layout="wide")

st.html(f"""
<style>
  .atomized-wordmark {{ color: {ACCENT_COLOR}; font-size: 2rem; font-weight: 750; }}
  [data-testid="stPopoverBody"] {{
    min-width: min(520px, calc(100vw - 2rem));
    max-width: min(720px, calc(100vw - 2rem));
  }}
</style>
""")

PAGES = ["Media Library", "Find Footage", "Edit Plan", "Export"]
MODES = {"Full Clips": "clip_assembly", "Selected Moments": "moment_assembly"}
ASPECTS = ["16:9", "9:16", "4:3", "3:4", "1:1"]


def _orch():
    from app.orchestrator import production_orchestrator
    return production_orchestrator


def _go(page):
    st.session_state["workspace_page"] = page


def _remember(key):
    st.session_state["ui_values"][key] = st.session_state[key]


def _seed(key, default):
    st.session_state.setdefault(key, st.session_state["ui_values"].get(key, default))


def _choose(ids, selected):
    set_candidates(st.session_state, ids, selected)
    # Checkbox keys are views of the durable candidate IDs, never the source of truth.
    for key in list(st.session_state):
        if key.startswith("candidate:") and int(key.rsplit(":", 1)[1]) in ids:
            st.session_state[key] = selected


def _toggle_candidate(clip_id, key):
    set_candidates(st.session_state, [clip_id], bool(st.session_state[key]))


def _candidate_checkbox(clip, page):
    key = f"candidate:{page}:{clip['clip_id']}"
    st.session_state[key] = clip["clip_id"] in st.session_state["candidate_clip_ids"]
    st.checkbox(Path(clip["file_path"]).name, key=key, on_change=_toggle_candidate,
                args=(clip["clip_id"], key), help="Include in the candidate set for this edit")


def _analysis_label(clip):
    count = clip.get("event_count", 0)
    if count:
        return f"Moments analysed {clip.get('analysed_event_count', 0)}/{count}"
    if clip.get("description") or clip.get("keywords"):
        return "Clip analysed · no moments"
    return "Metadata only"


def _meta(clip):
    orientation = clip.get("orientation") or "orientation unknown"
    return f"{clip.get('duration_seconds') or 0:.1f}s · {orientation} · {_analysis_label(clip)}"


@st.cache_data(show_spinner=False)
def _thumbnail(path, mtime_ns, source_revision, duration_seconds):
    try:
        from app.services.ffmpeg_service import extract_sample_frames_b64
        frames = extract_sample_frames_b64(path, duration_seconds=duration_seconds, count=1)
        return base64.b64decode(frames[0]) if frames else None
    except Exception:
        return None


def _render_clip(clip, page, thumbnails=False):
    with st.container(border=True):
        main, action = st.columns([5, 1])
        with main:
            _candidate_checkbox(clip, page)
            st.caption(_meta(clip))
            if page == "find":
                event = clip.get("matched_event")
                reason = ((event.get("action") if event else None)
                          or clip.get("description") or clip.get("keywords"))
                if reason:
                    st.caption(reason)
                if clip.get("relevance") is not None:
                    st.caption(f"Retrieval relevance {clip['relevance']:.0%}")
            if thumbnails:
                try:
                    stamp = Path(clip["file_path"]).stat().st_mtime_ns
                    frame = _thumbnail(clip["file_path"], stamp, clip["source_revision"], clip.get("duration_seconds") or 0)
                    if frame:
                        st.image(frame, width=200)
                except OSError:
                    st.caption("Source file unavailable")
        with action:
            preview = st.popover("Preview", key=f"preview:{page}:{clip['clip_id']}",
                                 on_change="rerun", width="stretch")
            if preview.open:
                with preview:
                    _render_preview(clip)


def _render_preview(clip):
    """Render only inside the open popover anchored to this clip's button."""
    st.markdown(f"**{Path(clip['file_path']).name}**")
    if Path(clip["file_path"]).is_file():
        st.video(clip["file_path"])
    else:
        st.warning("Source file unavailable. Reconnect the media folder and update the library.")
    st.caption(_meta(clip))
    st.caption(clip.get("description") or "No visual description yet.")


def _backup_headline(item: dict) -> str:
    """Format a backup item's name and measured time range."""
    name = item.get("name") or item.get("ref") or "(unidentified)"
    start, end = item.get("start_seconds"), item.get("end_seconds")
    if start is None or end is None:
        return name
    return f"{name} ({start:.2f}s–{end:.2f}s)"


def _backup_span(item: dict) -> str:
    """Format the measured span of a backup item."""
    start, end = item.get("start_seconds"), item.get("end_seconds")
    if end is None:
        return "length unmeasured"
    if not item.get("event_id") and not start:
        return f"full clip · {end:.1f}s"
    start = start or 0.0
    return f"{start:.2f}s → {end:.2f}s · {end - start:.1f}s"


def _render_backup_item(index: int, item: dict) -> None:
    """Render one backup-material card."""
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
        st.markdown("  \n".join(rows))      
        group = item.get("also_details") or []
        if group:
            st.caption(f"⧉ Same call for {len(group)} near-duplicate(s): "
                       + "; ".join(_backup_headline(g) for g in group))


def _strip_backup_section(report: str) -> str:
    """Remove the backup-material section from agent prose.
    Backup items are rendered separately from the structured plan.
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




def _sidebar():
    projects = catalogue.list_projects()
    labels = {p["project_id"]: p["project_name"] for p in projects}
    options = list(labels) or [1]
    if "pending_project_id" in st.session_state:
        st.session_state["project_picker"] = st.session_state.pop("pending_project_id")
    if st.session_state.get("project_picker") not in options:
        st.session_state["project_picker"] = options[0]
    with st.sidebar:
        st.html('<div class="atomized-wordmark">Atomized</div>')
        st.caption("Post Production Helper")
        pid = st.selectbox("Project", options, format_func=lambda p: labels.get(p, "Untitled project"),
                           key="project_picker")
        with st.expander("Workspace settings"):
            user = st.text_input("Editor ID", value="editor_01", key="workspace_editor_id",
                                 help="Separates working selections within this browser session")
            st.caption(f"Project ID: {pid}")
            with st.form("create_project_form"):
                name = st.text_input("New project name", key="new_project_name")
                create = st.form_submit_button("Create project")
            if create:
                try:
                    st.session_state["pending_project_id"] = catalogue.create_project(name)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        clips = catalogue.get_library_clips(pid)
        reconcile_workspace(st.session_state, pid, user, clips,
                            verify_plan=st.session_state.get("workspace_page") in ("Edit Plan", "Export"))
        st.subheader("Footage")
        _seed("footage_folder", "./app/data/raw_footage")
        directory = st.text_input("Media folder", key="footage_folder",
                                  on_change=_remember, args=("footage_folder",))
        with st.expander("Import options"):
            for key, default in (("import_visual", True), ("import_moments", True),
                                 ("import_force", False), ("import_verify", False)):
                _seed(key, default)
            visual = st.checkbox("Analyse visual content", key="import_visual",
                                 on_change=_remember, args=("import_visual",))
            moments = st.checkbox("Analyse moments", key="import_moments", disabled=not visual,
                                  on_change=_remember, args=("import_moments",))
            force = st.checkbox("Reanalyse existing media", key="import_force",
                                on_change=_remember, args=("import_force",),
                                help="Runs completed analysis again; ordinary updates reuse it")
            verify = st.checkbox("Verify file contents", key="import_verify",
                                 on_change=_remember, args=("import_verify",),
                                 help="Also detects replacements that kept the same size and timestamp")
        if st.button("Import / Update", key="import_update", type="primary",
                     use_container_width=True, disabled=not directory.strip()):
            _import_media(directory, pid, user, visual, moments, force, verify)
        result = st.session_state.get("ingest_result")
        if result:
            show = st.error if result["status"] == "failure" else (
                st.warning if result["status"] == "partial_success" else st.success)
            show(f"Reused {result['reused']} · New {result['new']} · Changed {result['changed']} "
                 f"· Failed {result['failed']}")
            with st.expander("Import details"):
                st.write(f"Updated analysis: {result.get('analysed', 0)}")
                for detail in result["details"]:
                    st.text(detail)
        st.divider()
        st.metric("Current candidates", len(st.session_state["candidate_clip_ids"]))
        online = st.session_state["clip_metadata"]
        st.caption(f"{len(online)} online clips · {sum(bool(c.get('event_count')) for c in online)} with moments")
        if len(clips) > len(online):
            st.caption(f"{len(clips) - len(online)} offline clips")
        st.button("Clear candidates", key="clear_candidates", on_click=_choose,
                  args=(list(st.session_state["candidate_clip_ids"]), False),
                  disabled=not st.session_state["candidate_clip_ids"])
    return pid, user


def _import_media(directory, pid, user, visual, moments, force, verify):
    before = {c["clip_id"]: c for c in catalogue.get_library_clips(pid)}
    totals = {"status": "success", "reused": 0, "new": 0, "changed": 0, "failed": 0,
              "analysed": 0, "details": []}
    offset, indexed = 0, 0
    try:
        with st.status("Updating media library…", expanded=True) as progress:
            while True:
                st.write(f"Processing batch {offset // 200 + 1}")
                _, result = _orch().run_ingest(directory, pid, user, batch_offset=offset,
                    analyze_content=visual, analyze_events=moments, force_reanalyze=force,
                    verify_content=verify)
                totals["reused"] += result.reused_count
                totals["failed"] += result.unreadable_count
                indexed += result.indexed_count
                totals["details"].extend([result.message, *result.warnings, *result.errors])
                if result.status != "success":
                    totals["status"] = result.status
                if result.status == "failure" or result.next_offset is None:
                    break
                offset = result.next_offset
            progress.update(label="Import finished", state="error" if totals["status"] == "failure" else "complete",
                            expanded=False)
    except Exception as exc:
        totals["status"] = "failure"
        totals["details"].append(str(exc))
    after = catalogue.get_library_clips(pid)
    totals["new"] = sum(c["clip_id"] not in before for c in after)
    totals["changed"] = sum(c["clip_id"] in before and c["source_revision"] != before[c["clip_id"]]["source_revision"]
                            for c in after)
    totals["analysed"] = max(0, indexed - totals["reused"] - totals["new"] - totals["changed"])
    reconcile_workspace(st.session_state, pid, user, after)
    st.session_state["ingest_result"] = totals
    st.rerun()


def _library():
    st.header("Media Library")
    st.caption("Browse project media and choose candidates for your edit.")
    clips = st.session_state["clip_metadata"]
    if not clips:
        st.info("Import your media using Footage in the sidebar. Existing project media appears here automatically.")
        return
    for key, default in (("library_filter", ""), ("library_orientation", "Any orientation"),
                         ("library_analysis", "Any analysis"), ("library_selected", False),
                         ("library_thumbnails", False), ("library_size", 20)):
        _seed(key, default)
    query = st.text_input("Filter media", key="library_filter", on_change=_remember, args=("library_filter",))
    a, b, c = st.columns(3)
    orientation = a.selectbox("Orientation", ["Any orientation", "Landscape", "Portrait", "Square"],
                             key="library_orientation", on_change=_remember, args=("library_orientation",))
    analysis = b.selectbox("Analysis", ["Any analysis", "Metadata only", "Has moments", "No moments"],
                          key="library_analysis", on_change=_remember, args=("library_analysis",))
    selected_only = c.checkbox("Candidates only", key="library_selected",
                              on_change=_remember, args=("library_selected",))
    selected = set(st.session_state["candidate_clip_ids"])
    filtered = [clip for clip in clips if (
        (not query or query.casefold() in (clip["file_path"] + " " + (clip.get("keywords") or "")).casefold())
        and (orientation == "Any orientation" or clip.get("orientation") == orientation.lower())
        and (not selected_only or clip["clip_id"] in selected)
        and (analysis != "Metadata only" or not (clip.get("description") or clip.get("keywords")))
        and (analysis != "Has moments" or clip.get("event_count"))
        and (analysis != "No moments" or not clip.get("event_count")))]
    size = st.selectbox("Clips per page", [20, 50], key="library_size",
                        on_change=_remember, args=("library_size",))
    pages = max(1, math.ceil(len(filtered) / size))
    _seed("library_page", 1)
    st.session_state["library_page"] = min(st.session_state["library_page"], pages)
    page = st.number_input("Page", min_value=1, max_value=pages, step=1, key="library_page",
                           on_change=_remember, args=("library_page",))
    shown = filtered[(page - 1) * size:page * size]
    st.caption(f"{len(filtered)} matching clips · page {page}/{pages}")
    left, middle, right = st.columns(3)
    left.button("Select this page", key="library_select_page", on_click=_choose,
                args=([c["clip_id"] for c in shown], True), disabled=not shown)
    middle.button("Clear this page", key="library_clear_page", on_click=_choose,
                  args=([c["clip_id"] for c in shown], False), disabled=not shown)
    right.button("Create edit plan", key="library_to_plan", on_click=_go, args=("Edit Plan",),
                 disabled=not selected)
    thumbnails = st.checkbox("Show thumbnails", key="library_thumbnails",
                             on_change=_remember, args=("library_thumbnails",),
                             help="Loads thumbnails for this page only")
    if not shown:
        st.info("No media matches these filters. Broaden the filters to see more clips.")
    for clip in shown:
        _render_clip(clip, "library", thumbnails)


def _find(pid, user):
    st.header("Find Footage")
    st.caption("Search for subjects, actions or atmosphere. Add matches to the same candidate set.")
    if not st.session_state["clip_metadata"]:
        st.info("Import media in the sidebar, or switch to a project with an existing library.")
        return
    _seed("find_query", "")
    with st.form("find_form"):
        query = st.text_input("What are you looking for?", key="find_query",
                              placeholder="e.g. People celebrating outdoors")
        submit = st.form_submit_button("Find footage", type="primary")
    if submit:
        _remember("find_query")
        try:
            with st.spinner("Finding footage…"):
                found = _orch().run_search(query, pid, user)
                by_id = {c["clip_id"]: c for c in st.session_state["clip_metadata"]}
                st.session_state["search_results"] = [{**by_id.get(c["clip_id"], {}), **c} for c in found]
            st.session_state["ui_values"]["find_has_run"] = True
            st.session_state["find_page"] = 1
        except Exception as exc:
            st.error(f"Search could not finish: {exc}")
    rows = st.session_state["search_results"]
    if not rows:
        if st.session_state["ui_values"].get("find_has_run"):
            st.info("No matching footage. Try broader content terms.")
        return
    st.caption(f"{len(rows)} results")
    # Retrieval stays untouched; the UI renders one window of its result list.
    _seed("find_page", 1)
    pages = max(1, math.ceil(len(rows) / 20))
    st.session_state["find_page"] = min(st.session_state["find_page"], pages)
    page = st.number_input("Results page", min_value=1, max_value=pages, step=1, key="find_page",
                           on_change=_remember, args=("find_page",))
    shown = rows[(page - 1) * 20:page * 20]
    a, b, c = st.columns(3)
    a.button("Add this page", key="find_add_page", on_click=_choose,
             args=([c["clip_id"] for c in shown], True))
    b.button("Remove this page", key="find_remove_page", on_click=_choose,
             args=([c["clip_id"] for c in shown], False))
    c.button("Create edit plan", key="find_to_plan", on_click=_go, args=("Edit Plan",))
    for clip in shown:
        _render_clip(clip, "find")


def _plan_summary(plan):
    mode = "Full Clips" if plan.get("mode") == "clip_assembly" else "Selected Moments"
    st.caption(f"{mode} · {len(plan['segments'])} segments · {plan.get('total_seconds', 0):g}s "
               + (f"· {plan['aspect_ratio']}" if plan.get("aspect_ratio") else ""))
    if plan.get("target_seconds"):
        st.caption(f"Target {plan['target_seconds']:g}s · {plan.get('duration_status', '').replace('_', ' ')}")
    if plan.get("ordering_strategy"):
        st.write(plan["ordering_strategy"])
    st.dataframe([{"Order": s["order"], "Clip": s.get("name") or Path(s["file_path"]).name,
                   "In (s)": s["in_point"], "Out (s)": s["out_point"],
                   "Duration (s)": s["duration"], "Role": s.get("label", "")}
                  for s in plan["segments"]], hide_index=True, use_container_width=True)


def _edit(pid, user):
    st.header("Edit Plan")
    candidates = [c for c in st.session_state["clip_metadata"]
                  if c["clip_id"] in st.session_state["candidate_clip_ids"]]
    st.caption(f"{len(candidates)} candidate clips")
    if not candidates:
        st.info("Choose candidates in Media Library or Find Footage to create an edit plan.")
        st.button("Choose media", key="plan_to_library", on_click=_go, args=("Media Library",))
    for key, default in (("plan_mode", "Full Clips"), ("plan_aspect", "16:9"),
                         ("plan_target", None), ("plan_intent", ""), ("plan_name", "Edit plan")):
        _seed(key, default)
    mode = st.radio("Editing mode", list(MODES), horizontal=True, key="plan_mode",
                    captions=["Keep complete video clips.", "Choose moments within clips."],
                    on_change=_remember, args=("plan_mode",))
    a, b = st.columns(2)
    aspect = a.selectbox("Output aspect ratio", ASPECTS, key="plan_aspect",
                         on_change=_remember, args=("plan_aspect",))
    target = b.number_input("Target duration (seconds)", min_value=1.0, max_value=7200.0,
                            value=None, key="plan_target", disabled=mode == "Full Clips",
                            placeholder="Optional", on_change=_remember, args=("plan_target",))
    if mode == "Full Clips":
        st.caption("Full Clips preserves the complete source lengths; target duration does not apply.")
    elif candidates and not any(c.get("analysed_event_count") for c in candidates):
        st.warning("These candidates have no analysed moments. Enable Analyse moments in Import options and update the media.")
    intent = st.text_area("Editing intent", key="plan_intent", on_change=_remember, args=("plan_intent",),
                          placeholder="Describe the purpose, style, pace and feeling of the edit.")
    name = st.text_input("Plan name", key="plan_name", on_change=_remember, args=("plan_name",))
    if st.button("Generate edit plan", key="plan_generate", type="primary", disabled=not candidates):
        if not intent.strip():
            st.warning("Describe the editing intent first.")
        else:
            try:
                with st.spinner("Building your edit plan…"):
                    report, plan = _orch().run_selection(intent.strip(), [c["file_path"] for c in candidates],
                        pid, user, editing_mode=MODES[mode], target_seconds=target if mode == "Selected Moments" else None,
                        aspect_ratio=aspect)
                    if not plan:
                        raise ValueError("No usable timeline was generated. Refine the intent and try again.")
                    save_edit_plan(st.session_state, plan, report, name)
                st.success("Plan saved in this workspace.")
            except Exception as exc:
                st.error(f"The edit plan could not be generated: {exc}")
    plan = st.session_state["edit_plan"]
    if plan:
        _plan_summary(plan)
        st.markdown(_strip_backup_section(st.session_state["edit_report"]))
        for i, item in enumerate(plan.get("excluded") or [], 1):
            _render_backup_item(i, item)
        st.button("Export this plan", key="plan_to_export", on_click=_go, args=("Export",))
        st.download_button("Download plan JSON", json.dumps(plan, ensure_ascii=False, indent=2),
                           file_name=f"{EXPORT_PREFIX}_edit_plan.json", mime="application/json", key="plan_download")


def _export(pid, user):
    st.header("Export")
    st.caption("Choose a saved plan and export it for Adobe Premiere Pro.")
    with st.expander("Open a saved plan JSON"):
        upload = st.file_uploader("Plan JSON", type=["json"], key="export_upload")
        if st.button("Open plan", key="export_open", disabled=upload is None):
            try:
                plan = validate_edit_plan(upload.getvalue(), pid)
                opened = save_edit_plan(st.session_state, plan, "", Path(upload.name).stem)
                st.session_state["export_plan"] = opened["id"]
                st.success("Verified plan opened.")
            except Exception as exc:
                st.error(f"This plan cannot be used: {exc}")
    records = st.session_state["saved_edit_plans"]
    if not records:
        st.info("Create an edit plan first, or open a previously downloaded plan JSON.")
        st.button("Create edit plan", key="export_to_plan", on_click=_go, args=("Edit Plan",))
        return
    by_id = {r["id"]: r for r in records}
    current = st.session_state.get("active_plan_id")
    if st.session_state.get("export_plan") not in by_id:
        st.session_state["export_plan"] = current if current in by_id else records[-1]["id"]
    selected = st.selectbox("Saved plan", list(by_id), key="export_plan",
                            format_func=lambda rid: f"{by_id[rid]['name']} · version {list(by_id).index(rid) + 1}")
    st.session_state["active_plan_id"] = selected
    record = by_id[selected]
    _plan_summary(record["plan"])
    valid = True
    try:
        validate_edit_plan(record["plan"], pid)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        valid = False
        st.warning(f"This plan needs to be regenerated: {exc}")
    _seed("export_name", DEFAULT_SEQUENCE_NAME)
    name = st.text_input("Sequence name", key="export_name", on_change=_remember, args=("export_name",))
    if st.button("Export Premiere project", key="export_compile", type="primary", disabled=not valid):
        try:
            with st.spinner("Preparing Premiere project…"):
                report, result = _orch().run_delivery(record["plan"], pid, user, sequence_name=name.strip() or DEFAULT_SEQUENCE_NAME)
            st.session_state.update(export_report=report, export_result=result, exported_plan_id=selected)
        except Exception as exc:
            st.session_state.update(export_result=None, export_report=str(exc), exported_plan_id=selected)
    result = st.session_state["export_result"]
    if result and st.session_state.get("exported_plan_id") == selected and result.status == "success":
        st.success(f"Exported {result.clip_count} segments · {result.width} × {result.height}")
        for path, label, mime in ((result.xml_path, "Download Premiere XML", "application/xml"),
                                  (result.json_path, "Download timeline JSON", "application/json")):
            if path and Path(path).is_file():
                st.download_button(label, Path(path).read_bytes(), file_name=Path(path).name, mime=mime,
                                   key="download:" + label)
        st.caption("Import the XML in Premiere Pro using File → Import.")
    elif st.session_state.get("export_report") and st.session_state.get("exported_plan_id") == selected:
        st.error(st.session_state["export_report"])


def main():
    pid, user = _sidebar()
    st.title(PRODUCT_NAME)
    page = st.radio("Workspace", PAGES, horizontal=True, key="workspace_page", label_visibility="collapsed")
    if page == "Media Library":
        _library()
    elif page == "Find Footage":
        _find(pid, user)
    elif page == "Edit Plan":
        _edit(pid, user)
    else:
        _export(pid, user)


if __name__ == "__main__":
    main()
