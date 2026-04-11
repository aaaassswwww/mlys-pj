from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Candidate:
    source: str
    value: float


@dataclass(frozen=True)
class FusionResult:
    value: float
    selected_source: str
    method: str
    confidence: float
    retained_sources: list[str]
    dropped_sources: list[str]


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
        )

    if len(items) == 1:
        item = items[0]
        return FusionResult(
            value=item.value,
            selected_source=item.source,
            method="single_source",
            confidence=0.6,
            retained_sources=[item.source],
            dropped_sources=[],
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

    fused_value = float(statistics.median([c.value for c in retained]))
    confidence = 0.7 + 0.1 * min(len(retained), 3) - 0.1 * len(dropped)
    confidence = max(0.0, min(0.95, confidence))

    return FusionResult(
        value=fused_value,
        selected_source=retained[0].source,
        method="robust_median",
        confidence=confidence,
        retained_sources=[c.source for c in retained],
        dropped_sources=[c.source for c in dropped],
    )


def pick_best(candidates: Iterable[Candidate], default_value: float = 0.0) -> tuple[float, str]:
    # Backward-compatible wrapper.
    result = fuse_candidates(candidates, default_value=default_value)
    return result.value, result.selected_source
