from __future__ import annotations

import argparse
from pathlib import Path

from profiler_agent.io.load_target_spec import load_target_spec
from profiler_agent.orchestrator.pipeline import execute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU profiling agent")
    parser.add_argument("--spec", type=Path, default=Path("inputs/target_spec.json"), help="path to target_spec.json")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = load_target_spec(args.spec)
    result = execute(spec=spec, out_dir=args.out)
    print(f"results.json: {result.results_path}")
    print(f"evidence.json: {result.evidence_path}")
    print(f"analysis.json: {result.analysis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
