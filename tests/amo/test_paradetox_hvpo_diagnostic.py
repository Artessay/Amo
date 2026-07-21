"""CPU-only regression test for the offline ParaDetox HVPO diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "paradetox" / "diagnose_hvpo_rollouts.py"
SPEC = importlib.util.spec_from_file_location("paradetox_hvpo_diagnostic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


def test_offline_diagnostic_self_test() -> None:
    DIAGNOSTIC._self_test()
