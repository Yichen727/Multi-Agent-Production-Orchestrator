"""Search Agent — unified hybrid retrieval over the production catalogue.

The SECOND stage of the MAPO pipeline (Ingest → Search → Selection). Given a
natural-language query, it retrieves matching candidate clips via ONE unified tool
(`search_catalogue`) backed by the hybrid retrieval service: SQL hard filters +
vector semantic recall (with a lexical fallback). It NEVER ranks or recommends a
"best" clip — that is the Selection Agent's responsibility.

The LLM's only job here is to translate the user's request into a STRUCTURED query
(keywords + optional filters); it no longer picks between many tools.
"""

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
    """Current project id from the injected graph state (defaults to 1).

    Injected by the ToolNode from ``ProductionState`` — the model never fills it — so a
    search is always confined to the current project's catalogue (audit C-02).
    """
    try:
        return int((state or {}).get("project_id"))
    except (TypeError, ValueError):
        return 1


# ── Formatting ─────────────────────────────────────────────────────────────────

_SUGGESTION_MARK = {"suggested": "🟡", "neutral": "⚪", "low": "🔴"}


def _format_candidates(candidates: list[dict]) -> str:
    """Render candidate dicts as a grounded, numbered text list for the agent."""
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
    return "\n".join(lines)


# ── Tools ────────────────────────────────────────────────────────────────────


@tool
def search_catalogue(keywords: str = None, shot_type: str = None,
                     orientation: str = None, people: int = None,
                     min_duration: float = None, max_duration: float = None,
                     state: Annotated[dict, InjectedState] = None) -> str:
    """Unified hybrid search over the catalogue — the ONE search tool.

    Combines structured SQL filters with semantic vector recall (falling back to
    lexical matching when embeddings are unavailable). Fill only the arguments the
    query implies; leave the rest as None.

    Args:
        keywords: Free-text semantic query — the subjects/actions/mood/scenery the
            user is after (e.g. "energetic crowd celebration"). Comma- or space-
            separated. This drives semantic recall, so expand the user's term into a
            generous set of English synonyms and related words.
        shot_type: e.g. wide_shot, close_up, establishing, medium_shot, aerial.
        orientation: 'portrait' | 'landscape' | 'square' (vertical/horizontal ok).
        people: Minimum number of people visible (use for "clips with people").
        min_duration: Minimum clip length in seconds.
        max_duration: Maximum clip length in seconds.

    Returns:
        A numbered list of matching clips with real metadata, a suggestion marker,
        and a grounded relevance %. Retrieval only — never a "best" pick.
    """
    candidates = hybrid_search(
        _pid_from_state(state), keywords=keywords, shot_type=shot_type,
        orientation=orientation, people=people,
        min_duration=min_duration, max_duration=max_duration,
    )
    return _format_candidates(candidates)


@tool
def list_all_shots(state: Annotated[dict, InjectedState] = None) -> str:
    """List every catalogued shot with its metadata (ground-truth check).

    Useful when the query asks for "all clips" or to see exactly what is catalogued.
    Scoped to the current project.
    """
    return _format_candidates(hybrid_search(_pid_from_state(state), top_k=200))


# ── Agent Assembly ───────────────────────────────────────────────────────────

search_tools = [
    search_catalogue,
    list_all_shots,
]

llm_with_search = llm.bind_tools(search_tools)
search_tool_node = ToolNode(search_tools)

SEARCH_PROMPT = """You are the SEARCH AGENT in the MAPO system.
Your role: RETRIEVE candidate clips from the catalogue that match a query.

You have ONE search tool: `search_catalogue`. You do NOT choose between many tools —
you translate the user's request into a STRUCTURED query and call it once (use
`list_all_shots` only for an "all clips" / ground-truth check). Report ONLY what the
tool returns. The catalogue is the single source of truth.

HOW TO BUILD THE QUERY:
- Put the semantic idea (subjects, actions, mood, scenery) into `keywords`.
- Put hard constraints into their OWN arguments: orientation, shot_type, people,
  min_duration / max_duration. Leave everything else as None.
- STRUCTURED WORDS ARE NOT KEYWORDS. Orientation words (horizontal/landscape,
  vertical/portrait, square) and shot-type words (wide, close-up, aerial, medium,
  establishing) describe FORMAT, not content — route them to `orientation` / `shot_type`
  and NEVER also place them in `keywords`. Putting them in `keywords` produces a
  misleadingly low relevance %, because they don't appear in the clip's content tags.
- If the request is ONLY a format constraint (e.g. "find all horizontal shots"), call
  the tool with just that argument (orientation="horizontal") and NO `keywords`. Those
  results have no relevance % — that is correct; every returned clip fully matches.

KEYWORDS EXPANSION — MANDATORY, FOR ANY TOPIC:
Retrieval matches semantics AND text, but the catalogue is tagged in English with one
wording out of many. So NEVER pass the user's raw word alone. For any concept the user
names — any category — expand `keywords` into a comma-separated set that includes:
  1. the term translated to English,
  2. close synonyms,
  3. broader/related terms and typical co-occurring words,
  4. common singular/plural and spelling variants.
Examples of the FORMAT only (not an allow-list):
  - "海" / "sea" → sea,ocean,water,beach,coast,waves,shore
  - "城市" / "city" → city,urban,buildings,street,skyline,downtown
A generous expansion only helps both the vector and lexical layers.

ANTI-HALLUCINATION RULES:
- NEVER invent file names, shot IDs, numbers, or relevance figures. Every clip you list
  must come from a tool result in this conversation.
- If the tool returns no rows, report that NO clips matched (after trying a wider
  synonym set). Do not pad the answer.
- Fields may be 'unclassified' / missing when the system has not analysed content; say
  so plainly rather than guessing.

OUTPUT: relay the tool's numbered candidate list. The 🟡/⚪/🔴 markers are the system's
relevance suggestion, not a verdict.

CRITICAL BOUNDARY: you RETRIEVE only. You do NOT rank clips as "best", make editorial
judgements, or recommend which clip to use — that is the Selection Agent's job. When a
query implies a creative choice, return all reasonable candidates and note that final
selection happens downstream.

Prior user preferences: {memory}"""


def search_assistant(state: ProductionState, config: RunnableConfig):
    """Search Agent reasoning node."""
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
