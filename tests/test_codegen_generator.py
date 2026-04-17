from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from profiler_agent.codegen.generator import ProbeCodeGenerator


class _FakeLLM:
    def __init__(self, payload: dict | None) -> None:
        self.payload = payload

    def is_enabled(self) -> bool:
        return True

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict | None:
        _ = system_prompt, user_prompt
        return self.payload


class CodegenGeneratorTests(unittest.TestCase):
    def test_generate_probe_writes_source_when_valid(self) -> None:
        out_dir = Path("tests/.tmp") / f"codegen_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            payload = {
                "metric": "dram_latency_cycles",
                "filename": "probe.cu",
                "rationale": "pointer chasing",
                "code": (
                    "__global__ void k(){}\n"
                    "int main(){\n"
                    "printf(\"metric=dram_latency_cycles value=1 samples=1 median=1 best=1 std=0\\n\");\n"
                    "return 0;\n"
                    "}\n"
                ),
            }
            gen = ProbeCodeGenerator(llm_client=_FakeLLM(payload))
            result = gen.generate_probe(metric="dram_latency_cycles", out_dir=out_dir)
            self.assertTrue(result.ok)
            self.assertTrue(result.source_path.exists())
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_generate_probe_fails_for_invalid_payload(self) -> None:
        gen = ProbeCodeGenerator(llm_client=_FakeLLM({"code": "int main(){return 0;}"}))
        result = gen.generate_probe(metric="dram_latency_cycles", out_dir=Path("tests/.tmp"))
        self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()

