"""OpenAI LLM and embedding utilities."""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config import settings
from app.utils.logger import get_logger
from app.services.metrics_service import UsageCallback, emit
import time

logger = get_logger("openai_service")


def get_llm(model: str = None, temperature: float = 0) -> ChatOpenAI:
    """Create a configured ChatOpenAI instance."""
    model = model or settings.LLM_MODEL
    llm = ChatOpenAI(model_name=model, temperature=temperature, callbacks=[UsageCallback(model)])
    logger.info(f"LLM initialized: {model}")
    return llm


def get_llm_fast() -> ChatOpenAI:
    """Return the configured fallback model."""
    return get_llm(model=settings.LLM_MODEL_FALLBACK)

llm = get_llm()
llm_fast = get_llm_fast()

# Embeddings (semantic retrieval layer)
_embeddings_client = None

def get_embeddings() -> OpenAIEmbeddings | None:
    """Return the cached embedding client, or None if unavailable."""
    global _embeddings_client
    if not settings.OPENAI_API_KEY:
        return None
    if _embeddings_client is None:
        _embeddings_client = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        logger.info(f"Embeddings initialized: {settings.EMBEDDING_MODEL}")
    return _embeddings_client


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Embed texts for semantic retrieval, returning None on failure."""
    if not texts:
        return None
    client = get_embeddings()
    if client is None:
        logger.warning("No OPENAI_API_KEY — skipping embeddings.")
        return None
    start = time.perf_counter()
    try:
        vectors = client.embed_documents(texts)
        emit("embedding_call", model=settings.EMBEDDING_MODEL, status="success", items=len(texts),
             elapsed_ms=round((time.perf_counter() - start) * 1000, 3),
             input_tokens=None, estimated_usd=None, usage_source="not_exposed_by_embed_documents")
        return vectors
    except Exception as e: 
        emit("embedding_call", model=settings.EMBEDDING_MODEL, status="error", items=len(texts),
             elapsed_ms=round((time.perf_counter() - start) * 1000, 3),
             error_type=type(e).__name__, input_tokens=None, estimated_usd=None)
        logger.error(f"Embedding failed: {e}")
        return None
