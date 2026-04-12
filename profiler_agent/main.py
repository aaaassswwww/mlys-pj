from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from profiler_agent.io.load_target_spec import load_target_spec
from profiler_agent.multi_agent import MultiAgentCoordinator, MultiAgentRequest
from profiler_agent.orchestrator.pipeline import execute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU profiling agent")
    parser.add_argument("--spec", type=Path, default=Path("inputs/target_spec.json"), help="path to target_spec.json")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="output directory")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multi"],
        default="single",
        help="execution mode: single pipeline or multi-agent coordinator",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="",
        help="optional high-level objective used by multi-agent routing/planning",
    )
    return parser.parse_args()


def _write_multi_agent_artifacts(out_dir: Path, plan: object, trace: object) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "multi_agent_plan.json"
    trace_path = out_dir / "multi_agent_trace.json"
    with plan_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(plan), f, indent=2, sort_keys=True)
    with trace_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(item) for item in trace], f, indent=2, sort_keys=True)
    return plan_path, trace_path


def main() -> int:
    args = parse_args()
    spec = load_target_spec(args.spec)

    if args.mode == "single":
        result = execute(spec=spec, out_dir=args.out)
        print(f"results.json: {result.results_path}")
        print(f"evidence.json: {result.evidence_path}")
        print(f"analysis.json: {result.analysis_path}")
        return 0

    coordinator = MultiAgentCoordinator()
    request = MultiAgentRequest(
        targets=spec.targets,
        run=spec.run,
        objective=args.objective,
        out_dir=args.out,
    )
    result = coordinator.run(request=request)
    pipeline_outputs = result.outputs.get("pipeline", {})
    print(f"results.json: {pipeline_outputs.get('results_path', args.out / 'results.json')}")
    print(f"evidence.json: {pipeline_outputs.get('evidence_path', args.out / 'evidence.json')}")
    print(f"analysis.json: {pipeline_outputs.get('analysis_path', args.out / 'analysis.json')}")
    plan_path, trace_path = _write_multi_agent_artifacts(out_dir=args.out, plan=result.plan, trace=result.trace)
    print(f"multi_agent_plan.json: {plan_path}")
    print(f"multi_agent_trace.json: {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
