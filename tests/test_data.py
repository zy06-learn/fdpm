from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from flight_delay_milp.data import (
    FEATURE_COLUMNS,
    discover_data_files,
    feature_target,
    make_synthetic_bts,
    normalize_schema,
    prepare_dataset,
    random_split_and_balance,
)


def data_config() -> dict:
    return {
        "start_year": 2005,
        "start_month": 1,
        "end_year": 2025,
        "end_month": 1,
        "target_threshold": 0.15,
        "target_operator": "gt",
    }


def test_normalize_schema_strips_bom_spaces_and_unnamed_column() -> None:
    frame = pd.DataFrame({"\ufeffYear ": [2024], "Arr Flights": [100], "Unnamed: 21": [0]})
    normalized = normalize_schema(frame)
    assert list(normalized.columns) == ["year", "arr_flights"]


def test_target_is_strict_aggregate_delay_rate() -> None:
    raw = pd.DataFrame(
        {
            "year": [2024, 2024, 2024],
            "month": [1, 1, 1],
            "carrier": ["AA", "AA", "AA"],
            "airport": ["ATL", "ATL", "ATL"],
            "arr_flights": [100, 100, 0],
            "arr_del15": [15, 16, 0],
        }
    )
    prepared, summary = prepare_dataset(raw, data_config())
    assert prepared["target"].tolist() == [0, 1]
    assert prepared["delay_rate"].tolist() == [0.15, 0.16]
    assert summary.raw_rows == 3
    assert summary.valid_rows == 2


def test_feature_boundary_excludes_outcomes() -> None:
    prepared, _ = prepare_dataset(make_synthetic_bts(250, 42), data_config())
    features, target = feature_target(prepared)
    assert list(features.columns) == FEATURE_COLUMNS
    assert "arr_del15" not in features
    assert "delay_rate" not in features
    assert set(np.unique(target)) == {0, 1}


def test_undersampling_touches_training_partition_only() -> None:
    prepared, _ = prepare_dataset(make_synthetic_bts(800, 7), data_config())
    _, y_train, x_test, y_test, _ = random_split_and_balance(prepared, 0.2, 42)
    counts = y_train.value_counts()
    assert counts.loc[0] == counts.loc[1]
    assert len(x_test) == len(y_test) == ceil(len(prepared) * 0.2)
    assert not np.isclose(float(y_test.mean()), 0.5, atol=1e-3)


def test_directory_prefers_extracted_csv_over_preserved_zip(tmp_path) -> None:
    csv_path = tmp_path / "export.csv"
    zip_path = tmp_path / "export.zip"
    csv_path.write_text("year,month\n2024,1\n", encoding="utf-8")
    zip_path.write_bytes(b"not read when a CSV is present")
    assert discover_data_files(tmp_path) == [csv_path]
