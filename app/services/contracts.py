"""Shared boundary validation. Project scope never comes from model output."""

import math


def project_id_value(value) -> int:
    if isinstance(value, bool) or not str(value).strip().isdigit() or int(value) < 1:
        raise ValueError("Project ID must be a positive integer")
    return int(value)


def target_duration(value) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Target duration must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Target duration must be a finite positive number")
    return result
