"""Search Agent — unified hybrid retrieval over the production catalogue.

The SECOND stage of the MAPO pipeline (Ingest → Search → Selection). Given a
natural-language query, it retrieves matching candidate clips via ONE unified tool
(`search_catalogue`) backed by the hybrid retrieval service: SQL hard filters +
vector semantic recall (with a lexical fallback). It NEVER ranks or recommends a
"best" clip — that is the Selection Agent's responsibility.

The LLM's only job here is to translate the user's request into a STRUCTURED query
(keywords + optional filters); it no longer picks between many tools.
"""

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
        # Event-aware recall: when a MOMENT inside the clip drove the match, note it as
        # context (the result is still the whole clip — Selection decides the cut).
        ev = c.get("matched_event")
        if ev and ev.get("action"):
            lines.append(f"      ↳ contains: {ev['action']} "
                         f"(~{ev.get('start_seconds', 0):.0f}-{ev.get('end_seconds', 0):.0f}s)")
    return "\n".join(lines)


def _render(candidates: list[dict]) -> str:
    """Return the readable candidate list for the LLM PLUS a machine-readable JSON block.

    The fenced ```json array carries the exact ``hybrid_search`` rows. The orchestrator
    reads the candidates back from THIS tool output (via ``_extract_candidates``), not from
    the model's prose — so the UI receives the real structured candidates (relevance,
    suggestion, matched_event, ...) it renders, and the model reformatting its narration
    can never corrupt them. Mirrors how the Selection planners emit their structured plan.
    """
    text = _format_candidates(candidates)
    return text + "\n\n```json\n" + json.dumps(candidates, default=str) + "\n```"


# ── Tools ────────────────────────────────────────────────────────────────────


@tool
def search_catalogue(keywords: str = None, core_keywords: str = None,
                     shot_type: str = None, orientation: str = None, people: int = None,
                     min_duration: float = None, max_duration: float = None,
                     state: Annotated[dict, InjectedState] = None) -> str:
    """Unified hybrid search over the catalogue — the ONE search tool.

    Combines structured SQL filters with semantic vector recall (falling back to
    lexical matching when embeddings are unavailable). Fill only the arguments the
    query implies; leave the rest as None.

    Args:
        core_keywords: The things the USER LITERALLY ASKED FOR, translated to English —
            the entities/subjects/actions themselves, comma-separated, NO synonyms
            (user says "手机" or "phone" → "phone"). Matching these counts as strong
            evidence, so putting a synonym here would overstate a match. Always fill
            this whenever the request has any content component.
        keywords: The FULL retrieval term set — ``core_keywords`` PLUS your synonym
            expansion (e.g. "phone, mobile phone, smartphone, cellphone"). Comma-
            separated. Expansion widens recall only; it can never certify a match.
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
        _pid_from_state(state), keywords=keywords or core_keywords,
        core_keywords=core_keywords, shot_type=shot_type,
        orientation=orientation, people=people,
        min_duration=min_duration, max_duration=max_duration,
    )
    return _render(candidates)


@tool
def list_all_shots(state: Annotated[dict, InjectedState] = None) -> str:
    """List every catalogued shot with its metadata (ground-truth check).

    Useful when the query asks for "all clips" or to see exactly what is catalogued.
    Scoped to the current project.
    """
    return _render(hybrid_search(_pid_from_state(state), top_k=200))


# ── Agent Assembly ───────────────────────────────────────────────────────────

search_tools = [
    search_catalogue,
    list_all_shots,
]

llm_with_search = llm.bind_tools(search_tools)
search_tool_node = ToolNode(search_tools)

SEARCH_PROMPT = """You are the SEARCH AGENT in the MAPO system.
Your role: RETRIEVE candidate clips from the catalogue that match a query.

You retrieve CLIPS (whole files), never moments or timecodes. Deciding WHICH MOMENT of a
clip to use is the Selection stage's job, not yours — you help the editor browse and
shortlist footage. You translate the user's request into a STRUCTURED query and call
`search_catalogue` once (use `list_all_shots` only for an "all clips" / ground-truth
check). Report ONLY what the tool returns. The catalogue is the single source of truth.

`search_catalogue` recall is EVENT-AWARE under the hood: a clip is returned when its
overall content OR a specific moment inside it matches your query, so a query like
"celebration" finds a clip even if only one beat of it is celebratory. When a moment drove
the match, the result carries a "contains: ..." note — relay it as CONTEXT, but the unit
you return is always the whole clip.

HOW TO BUILD THE QUERY:
- Put the user's OWN terms (translated to English, no synonyms) into `core_keywords`,
  and those same terms PLUS your expansion into `keywords`. Both, every time there is a
  content component — the system scores a core-term match far higher than a synonym
  match, so mislabelling a synonym as core inflates relevance.
- Put hard constraints into their OWN arguments: orientation, shot_type, people,
  min_duration / max_duration. Leave everything else as None.
- STRUCTURED WORDS ARE NOT KEYWORDS. Orientation words (horizontal/landscape,
  vertical/portrait, square) and shot-type words (wide, close-up, aerial, medium,
  establishing) describe FORMAT, not content — route them to `orientation` / `shot_type`
  and NEVER also place them in `keywords`. Putting them in `keywords` produces a
  misleadingly low relevance %, because they don't appear in the clip's content tags.
- If the request is ONLY a format constraint (e.g. "find all horizontal shots"), call
  the tool with just that argument (orientation="horizontal") and NO keywords. Those
  results come back at 100% — correct, because orientation and duration are MEASURED
  facts, so every returned clip matches exactly.
- Content relevance never reaches 100%: tags come from a vision model that can be wrong,
  so a strong content match tops out around 95%. Never present a % as certainty.

KEYWORDS EXPANSION — MANDATORY, BUT DISCIPLINED:
The catalogue is tagged in English with one wording out of many, so never search the
user's raw word alone — but expansion is a RECALL device, not a licence to drift.
Every term you add to `keywords` must be able to REPLACE a core term: a synonym, an
alternative name, a spelling/singular-plural variant, or a specific type of it.
  - CONCRETE things (objects, people, places): synonyms and sub-types ONLY. Never widen
    them to a category, a context, or things that merely appear nearby.
      "手机" / "phone" → core: phone | keywords: phone, mobile phone, smartphone,
      cellphone, telephone
      WRONG: device, gadget, electronics, technology, communication, screen — and NEVER
      a word that merely CONTAINS the core word with another meaning (microphone,
      headphone, earphone).
  - ABSTRACT ideas (mood, atmosphere, weather, scenery, activity) MAY widen to closely
    related terms:
      "celebration" → core: celebration | keywords: celebration, party, cheering,
      applause, festival
An over-broad expansion does real damage: it pulls in unrelated footage and makes it
look like a strong match.

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
