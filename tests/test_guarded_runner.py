from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_guard(tmp_path: Path, timeout: float, command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "scripts/run_guarded.py",
            "--timeout-seconds",
            str(timeout),
            "--log",
            str(tmp_path / "run.log"),
            "--pid-file",
            str(tmp_path / "pid.txt"),
            "--status-file",
            str(tmp_path / "status.json"),
            "--",
            *command,
        ],
        check=False,
    )


def test_guarded_runner_records_success(tmp_path: Path) -> None:
    result = _run_guard(tmp_path, 5, [sys.executable, "-c", "print('complete')"])
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))

    assert result.returncode == 0
    assert status["returncode"] == 0
    assert status["timed_out"] is False
    assert (tmp_path / "run.log").read_text(encoding="utf-8").strip() == "complete"


def test_guarded_runner_times_out_and_reaps_process_group(tmp_path: Path) -> None:
    result = _run_guard(tmp_path, 0.1, [sys.executable, "-c", "import time; time.sleep(10)"])
    child_pid = int((tmp_path / "pid.txt").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))

    assert result.returncode == 124
    assert status["returncode"] == 124
    assert status["timed_out"] is True
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)

