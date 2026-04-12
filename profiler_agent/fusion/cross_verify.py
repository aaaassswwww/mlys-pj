from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Candidate:
    source: str
    value: float
    reliability: float = 1.0


@dataclass(frozen=True)
class FusionResult:
    value: float
    selected_source: str
    method: str
    confidence: float
    retained_sources: list[str]
    dropped_sources: list[str]
    source_reliability: dict[str, float]


def _clamp_reliability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _weighted_median(items: list[Candidate]) -> float:
    ordered = sorted(items, key=lambda c: c.value)
    total_weight = sum(max(0.01, _clamp_reliability(c.reliability)) for c in ordered)
    threshold = total_weight / 2.0
    running = 0.0
    for idx, item in enumerate(ordered):
        running += max(0.01, _clamp_reliability(item.reliability))
        if abs(running - threshold) <= 1e-12 and idx + 1 < len(ordered):
            return float((item.value + ordered[idx + 1].value) / 2.0)
        if running >= threshold:
            return float(item.value)
    return float(ordered[-1].value)


def fuse_candidates(candidates: Iterable[Candidate], default_value: float = 0.0) -> FusionResult:
    items = list(candidates)
    if not items:
        return FusionResult(
            value=default_value,
            selected_source="fallback_default",
            method="no_signal_default",
            confidence=0.0,
            retained_sources=[],
            dropped_sources=[],
            source_reliability={},
        )

    if len(items) == 1:
        item = items[0]
        return FusionResult(
            value=item.value,
            selected_source=item.source,
            method="single_source",
            confidence=round(max(0.2, 0.6 * _clamp_reliability(item.reliability)), 3),
            retained_sources=[item.source],
            dropped_sources=[],
            source_reliability={item.source: _clamp_reliability(item.reliability)},
        )

    values = [c.value for c in items]
    median_value = statistics.median(values)
    deviations = [abs(v - median_value) for v in values]
    mad = statistics.median(deviations)

    retained: list[Candidate] = []
    dropped: list[Candidate] = []
    if mad == 0:
        retained = items
    else:
        threshold = 3.5 * mad
        for item in items:
            if abs(item.value - median_value) <= threshold:
                retained.append(item)
            else:
                dropped.append(item)

    if not retained:
        retained = items
        dropped = []

    fused_value = _weighted_median(retained)
    avg_reliability = statistics.mean(_clamp_reliability(c.reliability) for c in retained)
    confidence = 0.7 + 0.1 * min(len(retained), 3) - 0.1 * len(dropped)
    confidence *= max(0.3, avg_reliability)
    confidence = max(0.0, min(0.95, confidence))
    selected = min(
        retained,
        key=lambda c: (
            -_clamp_reliability(c.reliability),
            abs(c.value - fused_value),
            c.source,
        ),
    )
    source_reliability = {c.source: _clamp_reliability(c.reliability) for c in items}

    return FusionResult(
        value=fused_value,
        selected_source=selected.source,
        method="robust_weighted_median",
        confidence=round(confidence, 3),
        retained_sources=[c.source for c in retained],
        dropped_sources=[c.source for c in dropped],
        source_reliability=source_reliability,
    )


def pick_best(candidates: Iterable[Candidate], default_value: float = 0.0) -> tuple[float, str]:
    # Backward-compatible wrapper.
    result = fuse_candidates(candidates, default_value=default_value)
    return result.value, result.selected_source
