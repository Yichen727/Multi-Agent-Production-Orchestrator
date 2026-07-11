"""LLM initialization and management."""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("openai_service")


def get_llm(model: str = None, temperature: float = 0) -> ChatOpenAI:
    """Create a ChatOpenAI instance.

    Args:
        model: Model name override. Defaults to settings.LLM_MODEL.
        temperature: Sampling temperature.

    Returns:
        Configured ChatOpenAI instance.
    """
    model = model or settings.LLM_MODEL
    llm = ChatOpenAI(model_name=model, temperature=temperature)
    logger.info(f"LLM initialized: {model}")
    return llm


def get_llm_fast() -> ChatOpenAI:
    """Get the cost-optimized fallback model for simpler tasks."""
    return get_llm(model=settings.LLM_MODEL_FALLBACK)


# Pre-initialized instances for import convenience
llm = get_llm()
llm_fast = get_llm_fast()


# ── Embeddings (semantic retrieval layer) ─────────────────────────────────────

_embeddings_client = None


def get_embeddings() -> OpenAIEmbeddings | None:
    """Return a cached OpenAIEmbeddings client, or None without an API key.

    Used to build the semantic vector space that powers hybrid search. Kept behind
    this single accessor so callers never construct an embeddings client directly.
    """
    global _embeddings_client
    if not settings.OPENAI_API_KEY:
        return None
    if _embeddings_client is None:
        _embeddings_client = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        logger.info(f"Embeddings initialized: {settings.EMBEDDING_MODEL}")
    return _embeddings_client


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Embed a batch of texts, or return None if embeddings are unavailable.

    Returns one vector per input text. Returns ``None`` (never a fabricated vector)
    when there is no API key, the input is empty, or the call fails — callers should
    then fall back to lexical matching rather than pretend a vector exists.
    """
    if not texts:
        return None
    client = get_embeddings()
    if client is None:
        logger.warning("No OPENAI_API_KEY — skipping embeddings.")
        return None
    try:
        return client.embed_documents(texts)
    except Exception as e:  # network / model / quota — degrade gracefully
        logger.error(f"Embedding failed: {e}")
        return None