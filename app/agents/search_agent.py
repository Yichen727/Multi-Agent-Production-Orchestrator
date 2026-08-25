"""Search Agent — unified hybrid retrieval over the production catalogue."""

import json
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, InjectedState
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from app.models.state import ProductionState
from app.services.openai_service import llm
from app.services.retrieval_service import hybrid_search
from app.utils.logger import get_logger

logger = get_logger("search_agent")


def _pid_from_state(state) -> int:
    """Get the current project ID."""
    try:
        return int((state or {}).get("project_id"))
    except (TypeError, ValueError):
        return 1


# ── Formatting ─────────────────────────────────────────────────────────────────

_SUGGESTION_MARK = {"suggested": "🟡", "neutral": "⚪", "low": "🔴"}


def _format_candidates(candidates: list[dict]) -> str:
    """Format candidates as a numbered list."""
    if not candidates:
        return ("No clips matched. Broaden the keywords or relax the filters, or use "
                "list_all_shots to see what is catalogued.")

    lines = ["Candidates (retrieval only — final selection happens downstream):"]
    for i, c in enumerate(candidates, 1):
        name = Path(c["file_path"]).name
        mark = _SUGGESTION_MARK.get(c.get("suggestion"), "")
        bits = []
        if c.get("orientation"):
            bits.append(c["orientation"])
        if c.get("duration_seconds"):
            bits.append(f"{c['duration_seconds']:.1f}s")
        if c.get("shot_type") and c["shot_type"] != "unclassified":
            bits.append(c["shot_type"])
        if c.get("mood"):
            bits.append(f"mood={c['mood']}")
        if c.get("group_size") and c["group_size"] not in ("unknown", "none"):
            bits.append(c["group_size"])
        rel = c.get("relevance")
        rel_str = f" — relevance {rel*100:.0f}%" if rel is not None else ""
        lines.append(f"  {i}. {mark} {name} — {', '.join(bits)}{rel_str}")
        # Show the matching event as context; selection handles the actual cut.
        ev = c.get("matched_event")
        if ev and ev.get("action"):
            lines.append(f"      ↳ contains: {ev['action']} "
                         f"(~{ev.get('start_seconds', 0):.0f}-{ev.get('end_seconds', 0):.0f}s)")
    return "\n".join(lines)


def _render(candidates: list[dict]) -> str:
    """Format candidates for the LLM and preserve the raw results as JSON."""
    text = _format_candidates(candidates)
    return text + "\n\n```json\n" + json.dumps(candidates, default=str) + "\n```"


# ── Tools ────────────────────────────────────────────────────────────────────


@tool
def search_catalogue(keywords: str = None, core_keywords: str = None,
                     shot_type: str = None, orientation: str = None, people: int = None,
                     min_duration: float = None, max_duration: float = None,
                     state: Annotated[dict, InjectedState] = None) -> str:
    """Search the catalogue using hybrid semantic and structured retrieval.

    Args:
        core_keywords: User's literal content terms, without synonyms.
        keywords: Core terms plus controlled synonym expansion.
        shot_type: Shot type filter.
        orientation: Portrait, landscape, or square.
        people: Minimum visible people count.
        min_duration: Minimum duration in seconds.
        max_duration: Maximum duration in seconds.

    Returns:
        Matching clips with metadata and relevance information.
    """
    candidates = hybrid_search(
        _pid_from_state(state), keywords=keywords or core_keywords,
        core_keywords=core_keywords, shot_type=shot_type,
        orientation=orientation, people=people,
        min_duration=min_duration, max_duration=max_duration,
    )
    return _render(candidates)


@tool
def list_all_shots(state: Annotated[dict, InjectedState] = None) -> str:
    """List all clips in the current project's catalogue."""
    return _render(hybrid_search(_pid_from_state(state), top_k=200))


# ── Agent Assembly ───────────────────────────────────────────────────────────

search_tools = [
    search_catalogue,
    list_all_shots,
]

llm_with_search = llm.bind_tools(search_tools)
search_tool_node = ToolNode(search_tools)

SEARCH_PROMPT = """You are the SEARCH AGENT in the MAPO system.
Your role is to retrieve candidate clips from the catalogue that match a query.

You retrieve whole clips. Final moment selection is handled by the Selection Agent.
Translate the user's request into a structured query and call `search_catalogue` once.
Use `list_all_shots` only for "all clips" or catalogue checks.
Report only tool results. The catalogue is the source of truth.

EVENT-AWARE RETRIEVAL:
A clip may match through its overall content or a temporal event inside it.
When `matched_event` is returned, relay its "contains" information as context.
The returned unit remains the whole clip.

QUERY:
- `core_keywords`: user's own content terms, translated to English, without synonyms.
- `keywords`: `core_keywords` plus controlled synonym expansion.
- Use dedicated fields for `orientation`, `shot_type`, `people`,
  `min_duration`, and `max_duration`.
- Keep orientation and shot-type terms in their dedicated fields.
- For format-only queries, use only the relevant structured filters.
- Content relevance is model-based and should not be presented as certainty.

KEYWORD EXPANSION:
- Concrete objects, people, and places: synonyms, variants, and specific sub-types.
- Abstract concepts such as mood, atmosphere, weather, scenery, and activity:
  closely related terms are allowed.
- Keep expansions semantically close and useful for recall.

Examples:
- "phone" → core: phone | keywords: phone, mobile phone, smartphone, cellphone
- "celebration" → core: celebration | keywords: celebration, party, cheering,
  applause, festival

GROUNDING:
- Report only filenames, IDs, metadata, and scores returned by tools.
- If no results are returned, report no matches.
- Treat missing or `unclassified` fields as unknown.

OUTPUT:
Relay the numbered candidate list and preserve the 🟡/⚪/🔴 suggestion markers.
Do not make editorial recommendations.

Prior user preferences: {memory}"""


def search_assistant(state: ProductionState, config: RunnableConfig):
    """Run the Search Agent."""
    memory = state.get("loaded_preferences", "None")
    prompt = SEARCH_PROMPT.format(memory=memory)
    response = llm_with_search.invoke(
        [SystemMessage(prompt)] + state["messages"]
    )
    return {"messages": [response]}


def should_continue_search(state: ProductionState, config: RunnableConfig) -> str:
    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return "end"
    return "continue"
