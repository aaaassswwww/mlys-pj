from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from profiler_agent.agent_state import AgentStateRecord, load_agent_state, save_agent_state


class AgentStateTests(unittest.TestCase):
    def test_load_missing_agent_state_returns_default_record(self) -> None:
        path = Path("tests/.tmp") / f"missing_state_{uuid4().hex}.json"
        record = load_agent_state(path)
        self.assertEqual(record.iteration, 0)
        self.assertEqual(record.selected_tools_history, [])

    def test_save_and_load_agent_state_roundtrip(self) -> None:
        out_dir = Path("tests/.tmp") / f"agent_state_{uuid4().hex}"
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = out_dir / "agent_state.json"
            original = AgentStateRecord(
                iteration=2,
                request_targets=["dram_latency_cycles"],
                request_run="python -c \"print(1)\"",
                selected_tools_history=[["executor", "microbench"]],
                current_bottleneck="memory_bound",
            )
            save_agent_state(path, original)
            loaded = load_agent_state(path)
            self.assertEqual(loaded.iteration, 2)
            self.assertEqual(loaded.request_targets, ["dram_latency_cycles"])
            self.assertEqual(loaded.current_bottleneck, "memory_bound")
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
