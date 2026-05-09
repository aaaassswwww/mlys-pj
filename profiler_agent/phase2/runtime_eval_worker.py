from __future__ import annotations

import argparse
import json
from pathlib import Path

from profiler_agent.phase2.evaluator import (
    CandidateArtifactPaths,
    build_harness_runtime_evaluator,
    build_torch_extension_candidate_runner,
)
from profiler_agent.phase2.models import GeneratedCandidate, LoraProblemSpec
from profiler_agent.phase2.models import LoadResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 isolated runtime evaluator worker")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--probe-import-only", action="store_true")
    return parser.parse_args()


def _load_request(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    if args.probe_import_only:
        return 0
    if args.request is None or args.response is None:
        raise SystemExit("--request and --response are required unless --probe-import-only is set")
    request = _load_request(args.request)

    candidate_id = str(request.get("candidate_id", "candidate"))
    library_path = Path(str(request.get("library_path", "")))
    problem_specs = [
        LoraProblemSpec(
            hidden_dim=int(item["hidden_dim"]),
            low_rank=int(item.get("low_rank", 16)),
            output_dim=(None if item.get("output_dim") is None else int(item["output_dim"])),
            num_tokens=int(item.get("num_tokens", 32)),
            dtype=str(item.get("dtype", "float32")),
            device=str(item.get("device", "cuda")),
        )
        for item in request.get("problem_specs", [])
    ]
    warmup_runs = int(request.get("warmup_runs", 3))
    measured_runs = int(request.get("measured_runs", 7))
    rtol = float(request.get("rtol", 1e-4))
    atol = float(request.get("atol", 1e-4))

    candidate = GeneratedCandidate(candidate_id=candidate_id, source_code="", source="subprocess_runtime")
    paths = CandidateArtifactPaths(
        source_path=args.request.parent / "optimized_lora.cu",
        library_path=library_path,
    )
    runtime = build_harness_runtime_evaluator(
        problem_specs=problem_specs,
        candidate_runner=build_torch_extension_candidate_runner(),
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        rtol=rtol,
        atol=atol,
    )
    evaluation = runtime(candidate, paths, LoadResult(ok=True, library_path=str(library_path), symbol_name="launch_optimized_lora"))
    args.response.write_text(json.dumps(evaluation.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
