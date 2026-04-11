from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from profiler_agent.schema.result_schema import ensure_numeric_results


def write_results(out_dir: Path, results: dict[str, Any], expected_targets: list[str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized = ensure_numeric_results(results, expected_targets)
    path = out_dir / "results.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, sort_keys=True)
    return path


def write_evidence(out_dir: Path, evidence: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "evidence.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)
    return path


def write_analysis(out_dir: Path, analysis: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "analysis.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, sort_keys=True)
    return path
