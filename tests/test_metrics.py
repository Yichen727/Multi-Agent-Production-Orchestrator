import json
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.config import settings
from app.services.metrics_service import UsageCallback, estimate_cost, measure_stage, normalise_usage


def records():
    return [json.loads(line) for p in settings.METRICS_OUTPUT_DIR.glob("*.jsonl")
            for line in p.read_text(encoding="utf-8").splitlines()]


def test_cost_counts_cache_and_reasoning_once():
    usage = normalise_usage({"prompt_tokens": 1000, "completion_tokens": 100,
                             "prompt_tokens_details": {"cached_tokens": 800},
                             "completion_tokens_details": {"reasoning_tokens": 40}})
    assert estimate_cost("gpt-5.5", usage) == pytest.approx(0.0044)
    assert estimate_cost("unknown-model", usage) is None
    assert estimate_cost("gpt-5.5", normalise_usage(None)) is None


def test_callback_logs_usage_without_prompt_response_or_error_text(monkeypatch):
    monkeypatch.setattr(settings, "METRICS_ENABLED", True)
    callback = UsageCallback("gpt-5.5")
    rid = uuid.uuid4()
    callback.on_chat_model_start({}, [["secret prompt"]], run_id=rid)
    message = AIMessage(content="secret output", usage_metadata={"input_tokens": 10,
                        "output_tokens": 3, "total_tokens": 13})
    callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=rid)
    callback.on_llm_error(RuntimeError("secret failure"), run_id=uuid.uuid4())
    data = records()
    assert data[0]["input_tokens"] == 10
    assert data[1]["estimated_usd"] is None
    assert "secret" not in json.dumps(data)


def test_stage_runs_are_isolated_under_concurrency_and_on_error(monkeypatch):
    monkeypatch.setattr(settings, "METRICS_ENABLED", True)

    @measure_stage("test")
    def task(project_id):
        if project_id == 3:
            raise ValueError("do not log this")
        return project_id

    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(task, [1, 2])) == [1, 2]
    with pytest.raises(ValueError):
        task(3)
    data = records()
    starts = [r for r in data if r["event"] == "stage_start"]
    assert len({r["run_id"] for r in starts}) == 3
    assert {r["project_id"] for r in starts} == {"1", "2", "3"}
    assert len([r for r in data if r.get("status") == "error"]) == 1


def test_measurement_io_failure_does_not_change_application_result(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "METRICS_ENABLED", True)
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("x")
    monkeypatch.setattr(settings, "METRICS_OUTPUT_DIR", not_a_directory)

    @measure_stage("test")
    def task(project_id):
        return "unchanged"

    assert task(1) == "unchanged"


def test_real_langchain_callback_path_with_mock_http(monkeypatch):
    import httpx
    from langchain_openai import ChatOpenAI

    monkeypatch.setattr(settings, "METRICS_ENABLED", True)
    def respond(request):
        return httpx.Response(200, json={
            "id": "offline-completion", "object": "chat.completion", "created": 1,
            "model": "gpt-5.5-2026-04-23",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        })
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        llm = ChatOpenAI(model="gpt-5.5", http_client=transport, max_retries=0,
                         callbacks=[UsageCallback("gpt-5.5")])
        assert llm.invoke("private text").content == "ok"
    event = next(r for r in records() if r["event"] == "model_call")
    assert event["input_tokens"] == 100
    assert event["estimated_usd"] == pytest.approx(.0011)
    assert "private text" not in str(records())
