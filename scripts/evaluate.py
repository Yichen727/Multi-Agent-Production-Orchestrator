"""Budgeted text query-parser trial. Default is dry-run; never uploads footage."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from app.services.pricing import PRICE_VERSION, PRICES, estimate_cost
from scripts.baseline import ROOT, dump_json

INSTRUCTION = (
    "Translate a footage search query into structured filters. core_keywords are the user's literal "
    "subjects/actions in English, without synonyms. Keep orientation, people and duration in dedicated "
    "fields. Return null for unspecified filters. Do not invent catalogue results."
)
FIELDS = {"core_keywords": {"type": ["string", "null"]},
          "orientation": {"type": ["string", "null"], "enum": ["portrait", "landscape", "square", None]},
          "shot_type": {"type": ["string", "null"]}, "people": {"type": ["integer", "null"]},
          "min_duration": {"type": ["number", "null"]}, "max_duration": {"type": ["number", "null"]}}
SCHEMA = {"type": "object", "properties": FIELDS, "required": list(FIELDS), "additionalProperties": False}


class TrialBudget:
    """Reserve conservative text-token estimates; retain reservations after uncertain errors."""

    def __init__(self, usd: float):
        if not math.isfinite(usd) or usd <= 0:
            raise ValueError("A finite, positive budget in USD is required")
        self.limit = usd
        self.accounted = 0.0

    def reserve(self, model: str, payload: dict, max_output_tokens: int) -> float | None:
        if model not in PRICES:
            raise ValueError("Model has no verified price; add a reviewed price before running")
        # Text only. UTF-8 bytes + generous schema/message overhead are a conservative
        # preflight allowance, NOT a provider-enforced absolute billing cap.
        bound = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 4096
        amount = estimate_cost(model, {"input_tokens": bound, "output_tokens": max_output_tokens})
        if self.accounted + amount > self.limit:
            return None
        self.accounted += amount
        return amount

    def settle(self, reserved: float, actual: float | None):
        if actual is not None:
            self.accounted = max(0.0, self.accounted - reserved + actual)


def run_trial(cases, model, budget, max_requests, max_output_tokens=512, call=None):
    if max_requests < 1 or not 16 <= max_output_tokens <= 2048:
        raise ValueError("Requests must be positive; output limit must be 16..2048")
    ledger = TrialBudget(budget)
    records = []
    for case in cases[:max_requests]:
        payload = {"instructions": INSTRUCTION, "input": case["query"],
                   "text": {"format": {"type": "json_schema", "name": "search_query", "strict": True,
                                       "schema": SCHEMA}}}
        reservation = ledger.reserve(model, payload, max_output_tokens)
        if reservation is None:
            records.append({"case_id": case["id"], "status": "stopped_budget"})
            break
        if call is None:
            records.append({"case_id": case["id"], "status": "dry_run",
                            "reserved_usd": reservation})
            continue
        start = time.perf_counter()
        try:
            response = call(model=model, **payload, max_output_tokens=max_output_tokens,
                            reasoning={"effort": "none"})
            usage = response.usage
            normalised = {"input_tokens": getattr(usage, "input_tokens", None),
                          "output_tokens": getattr(usage, "output_tokens", None),
                          "cached_input_tokens": getattr(getattr(usage, "input_tokens_details", None),
                                                         "cached_tokens", None)}
            cost = estimate_cost(model, normalised)
            ledger.settle(reservation, cost)
            item = {"case_id": case["id"], "status": "success", "usage": normalised,
                    "estimated_usd": cost, "elapsed_ms": round((time.perf_counter() - start) * 1000, 3)}
            try:
                parsed = json.loads(response.output_text)
                if set(parsed) != set(FIELDS):
                    raise ValueError("Incomplete structured response")
                item["parsed"] = parsed
            except (ValueError, TypeError):
                item["status"] = "invalid_output"
            records.append(item)
            if cost is None or ledger.accounted >= ledger.limit:
                break
        except Exception as exc:
            # No retries: the server may have processed a timed-out request already.
            records.append({"case_id": case["id"], "status": "error_charge_unknown",
                            "error_type": type(exc).__name__, "reserved_usd": reservation})
            break
    return {"mode": "live" if call else "dry_run", "model": model, "price_version": PRICE_VERSION,
            "budget_usd": ledger.limit, "accounted_or_reserved_usd": round(ledger.accounted, 10),
            "provider_hard_cap": False, "requests": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly permit paid requests")
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--max-requests", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--model", choices=sorted(PRICES), default="gpt-5.4-mini")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/query-trial.json")
    args = parser.parse_args()
    cases = json.loads((ROOT / "benchmarks/quality_cases.json").read_text(encoding="utf-8"))["search_cases"]
    # Validate and reserve BEFORE reading a real key or constructing an API client.
    preview = run_trial(cases, args.model, args.budget_usd, args.max_requests, args.max_output_tokens)
    if args.live:
        if not preview["requests"] or preview["requests"][0]["status"] == "stopped_budget":
            parser.error("Budget cannot reserve even the first request; no API request sent")
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(ROOT / ".env", override=False)
        with OpenAI(max_retries=0, timeout=30) as client:
            result = run_trial(cases, args.model, args.budget_usd, args.max_requests,
                               args.max_output_tokens, client.responses.create)
    else:
        result = preview
    dump_json(args.output, result)
    print(json.dumps({"output": str(args.output), "mode": result["mode"],
                      "requests": len(result["requests"]),
                      "accounted_or_reserved_usd": result["accounted_or_reserved_usd"]}))


if __name__ == "__main__":
    main()
