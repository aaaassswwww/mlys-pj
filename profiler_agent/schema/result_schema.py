from __future__ import annotations

import math
from typing import Any

from profiler_agent.schema.metric_specs import METRIC_SPECS


def ensure_numeric_results(results: dict[str, Any], expected_targets: list[str]) -> dict[str, float | int]:
    normalized: dict[str, float | int] = {}
    for target in expected_targets:
        if target not in results:
            raise ValueError(f"Missing result for target: {target}")
        value = results[target]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Result for {target} must be numeric")
        normalized[target] = value
    return normalized


def normalize_results_with_specs(
    results: dict[str, Any], expected_targets: list[str]
) -> tuple[dict[str, float | int], dict[str, Any]]:
    numeric = ensure_numeric_results(results, expected_targets)
    normalized: dict[str, float | int] = {}
    issues: list[dict[str, Any]] = []
    units: dict[str, str] = {}

    for target in expected_targets:
        value = float(numeric[target])
        spec = METRIC_SPECS.get(target)
        if spec is None:
            normalized[target] = value
            continue

        units[target] = spec.unit
        original = value
        if not math.isfinite(value):
            value = spec.min_value
            issues.append(
                {
                    "target": target,
                    "code": "non_finite",
                    "message": "Result was NaN/Inf and was reset to min_value.",
                    "original_value": original,
                    "normalized_value": value,
                }
            )

        if value < spec.min_value:
            issues.append(
                {
                    "target": target,
                    "code": "clamped_min",
                    "message": f"Result below minimum {spec.min_value} {spec.unit}; clamped.",
                    "original_value": value,
                    "normalized_value": spec.min_value,
                }
            )
            value = spec.min_value
        elif value > spec.max_value:
            issues.append(
                {
                    "target": target,
                    "code": "clamped_max",
                    "message": f"Result above maximum {spec.max_value} {spec.unit}; clamped.",
                    "original_value": value,
                    "normalized_value": spec.max_value,
                }
            )
            value = spec.max_value

        if spec.integer_like:
            value = int(round(value))
        else:
            value = round(value, spec.round_digits)
        normalized[target] = value

    quality = {
        "units": units,
        "issue_count": len(issues),
        "issues": issues,
        "normalized_target_count": len(normalized),
    }
    return normalized, quality
