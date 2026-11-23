from types import SimpleNamespace
from unittest.mock import Mock
import json

import pytest

from scripts.evaluate import FIELDS, TrialBudget, run_trial

CASES = [{"id": "q1", "query": "Portrait football clips"}, {"id": "q2", "query": "Phones"}]


def test_no_request_without_budget_reservation():
    call = Mock(side_effect=AssertionError("Must not send"))
    report = run_trial(CASES, "gpt-5.5", .000001, 2, call=call)
    assert report["requests"][0]["status"] == "stopped_budget"
    call.assert_not_called()


@pytest.mark.parametrize("budget", [0, -1, float("inf"), float("nan")])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError):
        TrialBudget(budget)


def test_dry_run_reserves_but_never_calls_a_provider():
    report = run_trial(CASES, "gpt-5.4-mini", .05, 1)
    assert report["mode"] == "dry_run"
    assert [r["status"] for r in report["requests"]] == ["dry_run"]


def test_unknown_charges_stop_without_retry():
    call = Mock(side_effect=TimeoutError("secret response"))
    report = run_trial(CASES, "gpt-5.4-mini", 1, 2, call=call)
    assert call.call_count == 1
    assert report["requests"][0]["status"] == "error_charge_unknown"
    assert report["accounted_or_reserved_usd"] > 0
    assert "secret" not in str(report)


def test_trial_settles_usage_and_stops_if_actual_cost_exceeds_budget():
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=100,
                               input_tokens_details=SimpleNamespace(cached_tokens=0)),
                               output_text=json.dumps({field: None for field in FIELDS}))
    call = Mock(return_value=response)
    report = run_trial(CASES, "gpt-5.4-mini", .1, 2, call=call)
    assert call.call_count == 1
    assert report["requests"][0]["status"] == "success"
    assert report["accounted_or_reserved_usd"] == pytest.approx(.75045)
    assert report["provider_hard_cap"] is False
