#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def _group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except (PermissionError, ProcessLookupError):
        return False
    return True


def _terminate_group(process_group: int, grace_seconds: float = 5.0) -> None:
    if not _group_exists(process_group):
        return
    os.killpg(process_group, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while _group_exists(process_group) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _group_exists(process_group):
        os.killpg(process_group, signal.SIGKILL)
        deadline = time.monotonic() + grace_seconds
        while _group_exists(process_group) and time.monotonic() < deadline:
            time.sleep(0.05)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one command with a hard timeout.")
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = _parser().parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("A command is required after --.")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive.")

    for path in (args.log, args.pid_file, args.status_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite run evidence: {path}")

    started = datetime.now(UTC)
    timed_out = False
    interrupted = False
    with args.log.open("wb") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        args.pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
        try:
            returncode = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_group(process.pid)
            process.wait()
            returncode = 124
        except KeyboardInterrupt:
            interrupted = True
            _terminate_group(process.pid)
            process.wait()
            returncode = 130
        finally:
            _terminate_group(process.pid)

    completed = datetime.now(UTC)
    status = {
        "status": "complete" if returncode == 0 else "failed",
        "returncode": returncode,
        "timed_out": timed_out,
        "interrupted": interrupted,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "pid": process.pid,
        "command": command,
    }
    args.status_file.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
