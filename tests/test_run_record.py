from __future__ import annotations

from flight_delay_milp.run import create_run_directory


def test_run_directories_are_unique_and_non_overwriting(tmp_path) -> None:
    first = create_run_directory(tmp_path, 42)
    second = create_run_directory(tmp_path, 42)
    assert first != second
    assert first.is_dir()
    assert second.is_dir()
