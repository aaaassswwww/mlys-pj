from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetSpec:
    targets: list[str]
    run: str


def validate_target_spec(raw: object) -> TargetSpec:
    if not isinstance(raw, dict):
        raise ValueError("target_spec must be a JSON object")

    targets = raw.get("targets")
    run = raw.get("run")

    if not isinstance(targets, list) or not targets:
        raise ValueError("target_spec.targets must be a non-empty list")
    if not all(isinstance(t, str) and t.strip() for t in targets):
        raise ValueError("target_spec.targets must contain non-empty strings")
    if not isinstance(run, str) or not run.strip():
        raise ValueError("target_spec.run must be a non-empty string")

    unique_targets: list[str] = []
    seen: set[str] = set()
    for target in targets:
        if target not in seen:
            unique_targets.append(target)
            seen.add(target)

    return TargetSpec(targets=unique_targets, run=run.strip())

