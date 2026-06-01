from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from profiler_agent.phase3.generator import Phase3CandidateGenerator
from profiler_agent.phase3.models import Phase3OptimizerState


class _StubLLMClient:
    def __init__(self, payload: dict | None) -> None:
        self.payload = payload
        self.calls: list[dict[str, str]] = []

    def is_enabled(self) -> bool:
        return self.payload is not None

    def complete_json(self, *, system_prompt: str, user_prompt: str):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.payload


class Phase3GeneratorTests(unittest.TestCase):
    def test_generator_prefers_llm_candidate_when_valid(self) -> None:
        llm = _StubLLMClient(
            {
                "candidate_id": "llm-fast-path",
                "rationale": "use a cleaner decode grouping path",
                "source_code": (
                    "def create_engine(model_config, weight_dir, device='cuda'):\n"
                    "    class Engine:\n"
                    "        def prefill(self, request_ids, input_ids):\n"
                    "            return None\n"
                    "        def decode(self, request_ids, token_ids):\n"
                    "            return None\n"
                    "        def remove(self, request_ids):\n"
                    "            return None\n"
                    "    return Engine()\n"
                ),
            }
        )
        generator = Phase3CandidateGenerator(llm_client=llm)
        state = Phase3OptimizerState(iteration=3)

        candidate = generator.generate_candidate(state=state, feedback=None)

        self.assertEqual(candidate.source, "llm_runtime_revision")
        self.assertEqual(candidate.candidate_id, "llm-fast-path-v03")
        self.assertIn("def create_engine(", candidate.source_code)
        self.assertEqual(len(llm.calls), 1)

    def test_generator_falls_back_when_llm_payload_invalid(self) -> None:
        llm = _StubLLMClient({"candidate_id": "bad", "rationale": "missing source"})
        generator = Phase3CandidateGenerator(llm_client=llm)
        state = Phase3OptimizerState(iteration=2)

        candidate = generator.generate_candidate(state=state, feedback=None)

        self.assertEqual(candidate.source, "deterministic_runtime_fallback")
        self.assertTrue(candidate.candidate_id.endswith("-v02"))
        self.assertIn("def create_engine(", candidate.source_code)
        self.assertEqual(len(llm.calls), 1)

    def test_generator_writes_debug_payload_with_prompts(self) -> None:
        debug_dir = Path("tests/.tmp") / f"phase3_debug_{uuid4().hex}"
        llm = _StubLLMClient(
            {
                "candidate_id": "prompt-debug",
                "rationale": "debug",
                "source_code": (
                    "def create_engine(model_config, weight_dir, device='cuda'):\n"
                    "    class Engine:\n"
                    "        def prefill(self, request_ids, input_ids):\n"
                    "            return None\n"
                    "        def decode(self, request_ids, token_ids):\n"
                    "            return None\n"
                    "        def remove(self, request_ids):\n"
                    "            return None\n"
                    "    return Engine()\n"
                ),
            }
        )
        try:
            generator = Phase3CandidateGenerator(llm_client=llm, debug_dir=debug_dir)
            state = Phase3OptimizerState(iteration=1)
            generator.generate_candidate(state=state, feedback=None)
            debug_path = debug_dir / "phase3_codegen_iter_01.json"
            self.assertTrue(debug_path.exists())
            payload = json.loads(debug_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["candidate_source"], "llm_runtime_revision")
            self.assertTrue(payload["system_prompt"])
            self.assertTrue(payload["user_prompt"])
        finally:
            shutil.rmtree(debug_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
