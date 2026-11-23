"""Versioned P0 standard text-token estimates, shared by telemetry and evaluation."""

PRICE_VERSION = "2026-11-23-standard-text-v1"
# USD / 1M tokens; source links and limitations are in docs/P0_BASELINE.md.
PRICES = {
    "gpt-5.5": (5.0, 0.50, 30.0),
    "gpt-5.5-2026-04-23": (5.0, 0.50, 30.0),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-mini-2026-03-17": (0.75, 0.075, 4.50),
}


def estimate_cost(model: str, usage: dict) -> float | None:
    prices = PRICES.get(model)
    inputs, outputs = usage.get("input_tokens"), usage.get("output_tokens")
    if prices is None or inputs is None or outputs is None:
        return None
    cached = usage.get("cached_input_tokens") or 0
    if cached > inputs or min(inputs, outputs, cached) < 0:
        return None
    # Reasoning tokens are included in output_tokens, never charged twice.
    return round(((inputs - cached) * prices[0] + cached * prices[1] + outputs * prices[2]) / 1e6, 10)
