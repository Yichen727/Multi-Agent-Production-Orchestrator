"""Parse a query once, then retrieve without an agent loop or result narration."""

from app.config import settings
from app.models.schemas import SearchQuery
from app.services.contracts import project_id_value
from app.services.openai_service import llm_fast
from app.services.retrieval_service import hybrid_search, hoist_orientation
from app.utils.logger import get_logger

logger = get_logger("search_service")

QUERY_PROMPT = """Translate this footage search to SearchQuery.
Use core_keywords for the user's literal subjects/actions translated to English, with
no synonyms. keywords contains core terms plus closely related synonyms for recall.
Use dedicated fields for orientation, shot_type, minimum people and duration bounds.
Do not add content terms to format-only queries. Empty/all-clips requests have no
filters. Do not invent filters or broaden the requested subjects."""


def parse_search_query(query: str) -> SearchQuery:
    text = (query or "").strip()
    if not text:
        return SearchQuery()
    if settings.OPENAI_API_KEY:
        try:
            result = llm_fast.with_structured_output(SearchQuery).invoke([
                ("system", QUERY_PROMPT), ("human", text)])
            parsed = result if isinstance(result, SearchQuery) else SearchQuery.model_validate(result)
            if (parsed.min_duration is not None and parsed.max_duration is not None
                    and parsed.min_duration > parsed.max_duration):
                raise ValueError("Reversed duration bounds")
            return parsed
        except Exception as exc:
            logger.warning("Query parsing failed (%s); using literal terms", type(exc).__name__)
    residual, orientation = hoist_orientation(text, None)
    return SearchQuery(core_keywords=residual or "", keywords=residual or "",
                       orientation=orientation)


def search_clips(query: str | SearchQuery, project_id, *, top_k: int = 200) -> list[dict]:
    pid = project_id_value(project_id)
    parsed = parse_search_query(query) if isinstance(query, str) else query
    parsed = SearchQuery.model_validate(parsed)
    if (parsed.min_duration is not None and parsed.max_duration is not None
            and parsed.min_duration > parsed.max_duration):
        raise ValueError("Minimum duration cannot exceed maximum duration")
    filters = parsed.model_dump()
    filters["keywords"] = parsed.keywords or parsed.core_keywords or None
    filters["core_keywords"] = parsed.core_keywords or None
    return hybrid_search(pid, top_k=top_k, **filters)
