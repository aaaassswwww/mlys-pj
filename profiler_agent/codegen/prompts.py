from __future__ import annotations

import json


def build_probe_generation_system_prompt() -> str:
    return (
        "You generate CUDA C++ microbenchmarks for GPU intrinsic profiling. "
        "Hard constraints: "
        "1) No external benchmarks, no internet downloads, no third-party benchmark imports. "
        "2) Output must be self-contained CUDA C++ source code compilable by nvcc. "
        "3) Program must print one final line using exact protocol: "
        "'metric=<name> value=<num> samples=<n> median=<num> best=<num> std=<num>'. "
        "4) Include a minimal microbenchmark logic targeted to the requested metric. "
        "5) Return JSON only with keys: metric, filename, code, rationale."
    )


def build_probe_generation_user_prompt(*, metric: str, prior_error: str | None = None) -> str:
    payload = {
        "metric": metric,
        "requirements": [
            "self_contained_cuda_cpp",
            "no_external_benchmark_or_download",
            "print_structured_metric_line",
            "avoid hardcoded vendor spec tables",
        ],
    }
    if prior_error:
        payload["prior_error"] = prior_error[-1500:]
        payload["task"] = "Fix compile/runtime issues while keeping metric objective."
    else:
        payload["task"] = "Generate first valid probe."
    return json.dumps(payload, ensure_ascii=True)


def build_probe_repair_system_prompt() -> str:
    return (
        "You repair CUDA C++ microbenchmark source code using compiler/runtime feedback. "
        "Do not introduce external dependencies. "
        "Return JSON only with keys: metric, filename, code, rationale."
    )

