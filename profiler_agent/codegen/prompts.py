from __future__ import annotations

import json


def build_probe_generation_system_prompt() -> str:
    return (
        "You generate CUDA C++ microbenchmarks for GPU intrinsic profiling. "
        "Hard constraints: "
        "1) No external benchmarks, no internet downloads, no third-party benchmark imports. "
        "2) Output must be self-contained CUDA C++ source code compilable by nvcc. "
        "3) Use a minimal, conservative style to maximize Windows nvcc+cl compile success: "
        "prefer headers <cstdio>, <cmath>, <cuda_runtime.h>; avoid <iostream>, <vector>, <algorithm>, <string>, streams, templates, and STL containers. "
        "If statistics are needed, compute them with plain loops and fixed-size arrays. "
        "Avoid C++ exceptions, advanced STL usage, and host-side abstractions that require heavy template instantiation. "
        "4) Inside device code use clock64() or clock(), never __clock64. "
        "5) Program must print one final line using exact protocol: "
        "'metric=<name> value=<num> samples=<n> median=<num> best=<num> std=<num>'. "
        "6) Print with printf in one line, and ensure numeric placeholders are filled. "
        "7) Include a minimal microbenchmark logic targeted to the requested metric. "
        "8) Return JSON only with keys: metric, filename, code, rationale."
    )


def build_probe_generation_user_prompt(*, metric: str, prior_error: str | None = None) -> str:
    payload = {
        "metric": metric,
        "requirements": [
            "self_contained_cuda_cpp",
            "no_external_benchmark_or_download",
            "print_structured_metric_line",
            "avoid hardcoded vendor spec tables",
            "prefer_minimal_headers_stdio_style",
            "avoid_iostream_vector_algorithm_and_other_stl_containers",
            "use_clock64_or_clock_not___clock64",
        ],
        "output_protocol_template": f"metric={metric} value=%f samples=%d median=%f best=%f std=%f",
        "compile_target": "nvcc -O3 -std=c++14",
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
        "Focus on compile success first. "
        "Keep headers minimal and avoid STL-heavy constructs. "
        "For Windows nvcc+cl compatibility, replace streams/containers with printf and fixed arrays when needed, "
        "and use clock64() or clock() instead of __clock64. "
        "Preserve the exact output protocol line format. "
        "Do not introduce external dependencies. "
        "Return JSON only with keys: metric, filename, code, rationale."
    )

