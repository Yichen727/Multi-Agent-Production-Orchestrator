"""Local latency/usage measurements. Missing usage or prices remain unknown."""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone

from langchain_core.callbacks import BaseCallbackHandler

from app.config import settings
from app.services.pricing import PRICE_VERSION, PRICES, estimate_cost

logger = logging.getLogger(__name__)
_context = contextvars.ContextVar("mapo_measurement", default={})
_write_lock = threading.Lock()


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def normalise_usage(raw: dict | None) -> dict:
    raw = raw or {}
    inputs = _count(raw.get("input_tokens", raw.get("prompt_tokens")))
    outputs = _count(raw.get("output_tokens", raw.get("completion_tokens")))
    details = raw.get("input_token_details") or raw.get("prompt_tokens_details") or {}
    output_details = raw.get("output_token_details") or raw.get("completion_tokens_details") or {}
    return {
        "input_tokens": inputs, "output_tokens": outputs,
        "cached_input_tokens": _count(details.get("cache_read", details.get("cached_tokens"))),
        "reasoning_tokens": _count(output_details.get("reasoning", output_details.get("reasoning_tokens"))),
    }


def emit(event: str, **values) -> None:
    if not settings.METRICS_ENABLED:
        return
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              **_context.get(), "event": event, **values}
    try:
        directory = settings.METRICS_OUTPUT_DIR
        directory.mkdir(parents=True, exist_ok=True)
        # One file per process, so a future worker cannot interleave a JSON line.
        with _write_lock:
            with (directory / f"records-{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    except (OSError, TypeError, ValueError):
        logger.warning("Could not write local measurement; application result is unaffected.")


def measure_stage(stage: str):
    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def measured(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            token = _context.set({"run_id": uuid.uuid4().hex, "stage": stage,
                                  "project_id": str(bound.arguments.get("project_id", ""))})
            start = time.perf_counter()
            emit("stage_start")
            try:
                result = fn(*args, **kwargs)
                outcome = getattr(result[-1], "status", None) if isinstance(result, tuple) and result else None
                emit("stage_end", status="completed", outcome=outcome,
                     elapsed_ms=round((time.perf_counter() - start) * 1000, 3))
                return result
            except Exception as exc:
                emit("stage_end", status="error", error_type=type(exc).__name__,
                     elapsed_ms=round((time.perf_counter() - start) * 1000, 3))
                raise
            finally:
                _context.reset(token)
        return measured
    return decorate


def measure_operation(operation: str):
    """Measure local/Vision work without recording paths, frames or result bodies."""
    def decorate(fn):
        @functools.wraps(fn)
        def measured(*args, **kwargs):
            if not settings.METRICS_ENABLED:
                return fn(*args, **kwargs)
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                emit("operation", operation=operation, status="completed",
                     result_available=result is not None,
                     elapsed_ms=round((time.perf_counter() - start) * 1000, 3))
                return result
            except Exception as exc:
                emit("operation", operation=operation, status="error", error_type=type(exc).__name__,
                     elapsed_ms=round((time.perf_counter() - start) * 1000, 3))
                raise
        return measured
    return decorate


class UsageCallback(BaseCallbackHandler):
    """Observe model responses without persisting prompts or generated content."""

    def __init__(self, model: str):
        self.model = model
        self._pending = {}
        self._lock = threading.Lock()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._start(run_id)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._start(run_id)

    def _start(self, run_id):
        with self._lock:
            self._pending[str(run_id)] = (time.perf_counter(), dict(_context.get()))

    def _finish(self, run_id):
        with self._lock:
            start, context = self._pending.pop(str(run_id), (time.perf_counter(), dict(_context.get())))
        return round((time.perf_counter() - start) * 1000, 3), context

    def on_llm_end(self, response, *, run_id, **kwargs):
        elapsed, context = self._finish(run_id)
        output = response.llm_output or {}
        raw = output.get("token_usage") or output.get("usage")
        model = output.get("model_name") or self.model
        if raw is None:
            # Current application uses one generation per model call.
            for group in response.generations:
                for generation in group:
                    message = getattr(generation, "message", None)
                    raw = getattr(message, "usage_metadata", None)
                    if raw:
                        model = (getattr(message, "response_metadata", {}) or {}).get("model_name", model)
                        break
                if raw:
                    break
        usage = normalise_usage(raw)
        emit("model_call", **context, call_id=str(run_id), model=model, status="success",
             elapsed_ms=elapsed, **usage, estimated_usd=estimate_cost(model, usage),
             price_version=PRICE_VERSION, sdk_retry_count=None)

    def on_llm_error(self, error, *, run_id, **kwargs):
        elapsed, context = self._finish(run_id)
        emit("model_call", **context, call_id=str(run_id), model=self.model, status="error",
             elapsed_ms=elapsed, error_type=type(error).__name__,
             **normalise_usage(None), estimated_usd=None, price_version=PRICE_VERSION,
             sdk_retry_count=None)

def current_run_id() -> str:
    """Read the invocation ID without sharing mutable results between callers."""
    return _context.get().get("run_id", "")
