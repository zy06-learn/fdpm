from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

CANONICAL_COLUMNS = [
    "year",
    "month",
    "carrier",
    "carrier_name",
    "airport",
    "airport_name",
    "arr_flights",
    "arr_del15",
    "carrier_ct",
    "weather_ct",
    "nas_ct",
    "security_ct",
    "late_aircraft_ct",
    "arr_cancelled",
    "arr_diverted",
    "arr_delay",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
]

FEATURE_COLUMNS = ["year", "month", "carrier", "airport", "arr_flights"]
CATEGORICAL_FEATURES = ["carrier", "airport"]
NUMERIC_FEATURES = ["year", "month", "arr_flights"]
POST_OUTCOME_COLUMNS = sorted(set(CANONICAL_COLUMNS) - set(FEATURE_COLUMNS))


@dataclass(frozen=True)
class DatasetSummary:
    raw_rows: int
    valid_rows: int
    positive_rate: float
    start_period: str
    end_period: str


def normalize_column_name(name: object) -> str:
    normalized = str(name).strip().lower()
    normalized = normalized.replace(" ", "_").replace("-", "_")
    return normalized.lstrip("\ufeff")


def normalize_schema(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [normalize_column_name(column) for column in result.columns]
    unnamed = [column for column in result.columns if column.startswith("unnamed:")]
    if unnamed:
        result = result.drop(columns=unnamed)

    if not set(CANONICAL_COLUMNS).issubset(result.columns) and len(result.columns) >= 21:
        first = list(result.columns[:21])
        if all(str(column).isdigit() or str(column).startswith("x") for column in first):
            result = result.rename(columns=dict(zip(first, CANONICAL_COLUMNS, strict=True)))
    return result


def _read_csv_file(path: Path) -> list[pd.DataFrame]:
    if path.suffix.lower() == ".zip":
        frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                member for member in archive.namelist() if member.lower().endswith(".csv")
            )
            if not members:
                raise ValueError(f"ZIP archive contains no CSV: {path}")
            for member in members:
                with archive.open(member) as handle:
                    frames.append(pd.read_csv(io.BytesIO(handle.read()), low_memory=False))
        return frames
    return [pd.read_csv(path, low_memory=False)]


def discover_data_files(path: str | Path) -> list[Path]:
    source = Path(path)
    if source.is_file():
        if source.suffix.lower() not in {".csv", ".zip"}:
            raise ValueError(f"Expected a CSV or ZIP file: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {source}")
    candidates = [file for file in source.rglob("*") if file.is_file()]
    csv_files = sorted(file for file in candidates if file.suffix.lower() == ".csv")
    zip_files = sorted(file for file in candidates if file.suffix.lower() == ".zip")
    # The official downloader intentionally preserves both the source ZIP and its
    # extracted CSV. Prefer extracted CSVs so one export is never loaded twice.
    files = csv_files or zip_files
    if not files:
        raise FileNotFoundError(f"No CSV or ZIP data files found under: {source}")
    return files


def load_raw_data(path: str | Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file in discover_data_files(path):
        frames.extend(_read_csv_file(file))
    return normalize_schema(pd.concat(frames, ignore_index=True, sort=False))


def filter_period(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    start = int(config["start_year"]) * 12 + int(config["start_month"])
    end = int(config["end_year"]) * 12 + int(config["end_month"])
    period = frame["year"].astype(int) * 12 + frame["month"].astype(int)
    return frame.loc[period.between(start, end)].copy()


def prepare_dataset(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, DatasetSummary]:
    raw_rows = len(frame)
    required = {"year", "month", "carrier", "airport", "arr_flights", "arr_del15"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required BTS columns: {missing}")

    prepared = frame.copy()
    for column in ("year", "month", "arr_flights", "arr_del15"):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=sorted(required))
    prepared = prepared.loc[prepared["arr_flights"] > 0].copy()
    prepared = filter_period(prepared, config)
    if prepared.empty:
        raise ValueError("No rows remain after schema, validity, and date filtering.")

    prepared["year"] = prepared["year"].astype(int)
    prepared["month"] = prepared["month"].astype(int)
    prepared["carrier"] = prepared["carrier"].astype(str)
    prepared["airport"] = prepared["airport"].astype(str)
    prepared["delay_rate"] = prepared["arr_del15"] / prepared["arr_flights"]
    prepared["target"] = (
        prepared["delay_rate"] > float(config["target_threshold"])
    ).astype("int8")

    periods = pd.to_datetime(
        {"year": prepared["year"], "month": prepared["month"], "day": 1}
    )
    summary = DatasetSummary(
        raw_rows=raw_rows,
        valid_rows=len(prepared),
        positive_rate=float(prepared["target"].mean()),
        start_period=periods.min().strftime("%Y-%m"),
        end_period=periods.max().strftime("%Y-%m"),
    )
    return prepared.reset_index(drop=True), summary


def feature_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = sorted(set(FEATURE_COLUMNS + ["target"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Prepared frame is missing columns: {missing}")
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    if set(features.columns) & set(POST_OUTCOME_COLUMNS):
        raise AssertionError("Post-outcome leakage column reached model features.")
    return features, frame["target"].astype(int).copy()


def random_split_and_balance(
    frame: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, np.ndarray]:
    features, target = feature_target(frame)
    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=target,
    )
    x_train = features.iloc[train_idx].reset_index(drop=True)
    y_train = target.iloc[train_idx].reset_index(drop=True)
    x_test = features.iloc[test_idx].reset_index(drop=True)
    y_test = target.iloc[test_idx].reset_index(drop=True)

    sampler = RandomUnderSampler(random_state=seed)
    x_balanced, y_balanced = sampler.fit_resample(x_train, y_train)
    return (
        x_balanced.reset_index(drop=True),
        y_balanced.reset_index(drop=True),
        x_test,
        y_test,
        np.asarray(test_idx),
    )


def temporal_split_and_balance(
    frame: pd.DataFrame, train_end_year: int, test_start_year: int, seed: int
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, np.ndarray]:
    train_frame = frame.loc[frame["year"] <= train_end_year]
    test_frame = frame.loc[frame["year"] >= test_start_year]
    if train_frame.empty or test_frame.empty:
        raise ValueError("Temporal split produced an empty train or test partition.")
    x_train, y_train = feature_target(train_frame)
    x_test, y_test = feature_target(test_frame)
    sampler = RandomUnderSampler(random_state=seed)
    x_balanced, y_balanced = sampler.fit_resample(x_train, y_train)
    return (
        x_balanced.reset_index(drop=True),
        y_balanced.reset_index(drop=True),
        x_test.reset_index(drop=True),
        y_test.reset_index(drop=True),
        test_frame.index.to_numpy(),
    )


def make_synthetic_bts(rows: int, seed: int) -> pd.DataFrame:
    if rows < 200:
        raise ValueError("Synthetic smoke data needs at least 200 rows.")
    rng = np.random.default_rng(seed)
    year = rng.integers(2005, 2026, size=rows)
    month = rng.integers(1, 13, size=rows)
    carrier = rng.choice(["AA", "DL", "UA", "WN", "B6"], size=rows)
    airport = rng.choice(
        ["ATL", "ORD", "DFW", "DEN", "LAX", "JFK", "SFO", "SEA"], size=rows
    )
    arr_flights = rng.integers(30, 1200, size=rows)
    carrier_effect = pd.Series(carrier).map(
        {"AA": 0.01, "DL": -0.02, "UA": 0.02, "WN": -0.01, "B6": 0.04}
    ).to_numpy()
    airport_effect = pd.Series(airport).map(
        {"ATL": 0.01, "ORD": 0.05, "DFW": 0.02, "DEN": 0.01,
         "LAX": 0.00, "JFK": 0.04, "SFO": 0.03, "SEA": -0.01}
    ).to_numpy()
    seasonal = 0.035 * np.sin((month - 1) / 12 * 2 * np.pi)
    trend = 0.002 * (year - 2005)
    congestion = 0.000045 * arr_flights
    noise = rng.normal(0, 0.045, size=rows)
    delay_probability = np.clip(
        0.08 + carrier_effect + airport_effect + seasonal + trend + congestion + noise,
        0.01,
        0.55,
    )
    arr_del15 = rng.binomial(arr_flights, delay_probability)
    frame = pd.DataFrame(
        {
            "year": year,
            "month": month,
            "carrier": carrier,
            "carrier_name": carrier,
            "airport": airport,
            "airport_name": airport,
            "arr_flights": arr_flights,
            "arr_del15": arr_del15,
        }
    )
    for column in CANONICAL_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
    return frame.loc[:, CANONICAL_COLUMNS]


def validation_report(summary: DatasetSummary, config: dict) -> dict[str, object]:
    report: dict[str, object] = {
        "observed": summary.__dict__,
        "expected": {},
        "matches": {},
    }
    for key, observed in (
        ("raw_rows", summary.raw_rows),
        ("valid_rows", summary.valid_rows),
        ("positive_rate", summary.positive_rate),
    ):
        expected_key = f"expected_{key}"
        if expected_key not in config:
            continue
        expected = config[expected_key]
        report["expected"][key] = expected
        tolerance = 1e-4 if key == "positive_rate" else 0
        report["matches"][key] = abs(float(observed) - float(expected)) <= tolerance
    return report


def class_counts(targets: Iterable[int]) -> dict[int, int]:
    values, counts = np.unique(np.asarray(list(targets)), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts, strict=True)}
