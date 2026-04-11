from __future__ import annotations


def build_task_plan(targets: list[str]) -> list[str]:
    # Keep target order from target_spec; dedupe is already handled in schema validation.
    return list(targets)

