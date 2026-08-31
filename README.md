# Flight-delay prediction and MILP scheduling

This repository is an evidence-aware reproduction of the paper *Flight Scheduling
Optimization Using Predicted Delay Probabilities: A Mixed-Integer Linear Programming
Approach*.

The pipeline uses the U.S. Bureau of Transportation Statistics (BTS) monthly
carrier-airport delay-cause table. It benchmarks 11 classifiers using only five
pre-departure features, produces out-of-fold stacking probabilities, and passes those
probabilities to a cost-aware PuLP/CBC scheduling model.

## Reproduction boundary

The paper does not contain the original code, exact hyperparameter search spaces,
scenario grid, or all random draws. This repository therefore separates three kinds of
decisions:

- **Specified by the paper:** January 2005 through January 2025; seed 42; 80/20
  stratified split; training-only random undersampling; five pre-departure features;
  11 candidate models; 5-fold XGBoost/LightGBM/CatBoost stacking with logistic
  regression; ROC-AUC, PR-AUC, F1, accuracy, calibration and Brier score; PuLP/CBC;
  penalty USD 2,000; capacity 60%; and temporal validation on 2005-2023 versus
  2024-2025.
- **Resolved contradiction:** each source row is a carrier-airport-month aggregate,
  so the primary target is `arr_del15 / arr_flights > 0.15`, as stated in the paper's
  dataset description. It is not an individual-flight `ArrDel15` indicator.
- **Explicit reconstruction choices:** model hyperparameters, scenario grid,
  commercial-value seed, mandatory-unit seed, and the baseline implementation are in
  versioned YAML rather than hidden in code.

See [docs/PAPER_CONTRACT.md](docs/PAPER_CONTRACT.md) for the full evidence map and
known limitations.

## Quick start

Python 3.11 is pinned locally. [`uv`](https://docs.astral.sh/uv/) creates an isolated
project environment; no global packages are modified.

```bash
uv sync --extra dev
./scripts/bootstrap_macos.sh  # macOS only; project-local OpenMP for LightGBM
./reproduce.sh test
./reproduce.sh smoke
```

The smoke run uses synthetic carrier-airport-month data, exercises all 11 model paths
and the structural MILP, and writes only under `artifacts/smoke/`. Smoke metrics are
pipeline evidence, not paper results.

## Obtain the BTS data

1. Open the official [BTS Airline On-Time Statistics and Delay Causes](https://www.transtats.bts.gov/OT_Delay/ot_delaycause1.asp?pn=1)
   page.
2. Select all carriers, all airports, January 2005 through January 2025, and choose
   **Download Raw Data**.
3. Copy the generated `https://...transtats.bts.gov/...zip` URL and run:

```bash
uv run flight-delay-milp download-bts \
  --url 'PASTE_THE_OFFICIAL_BTS_ZIP_URL_HERE' \
  --output-dir data/raw
```

The downloader rejects non-BTS hosts, does not overwrite prior files, records the URL,
timestamp, size and SHA-256 hash, and extracts the CSV safely. The BTS site generates
an opaque, selection-specific URL, so a static full-range URL is intentionally not
hard-coded.

Validate the raw export before a formal run:

```bash
./reproduce.sh validate-data data/raw
```

Expected paper counts are 372,765 raw rows, 372,130 rows after removing
`arr_flights <= 0`, and a 43.28% positive rate. Validation reports discrepancies; it
does not silently force the data to match.

## Formal reproduction

The formal command is deliberately explicit:

```bash
./reproduce.sh formal data/raw
```

Every run receives a unique directory under `artifacts/formal/` containing the
resolved configuration, exact command, Git and environment state, complete log,
metrics, predictions, optimization decisions, figures, fitted selected model, and a
`DONE.json` marker. Existing artifacts are never overwritten.

The formal benchmark can take substantially longer than the smoke run because it
fits all 11 models and a nested 5-fold stack on roughly 258k balanced training rows.
Run it only after approving the machine and runtime envelope.

## Repository layout

```text
configs/                 paper and smoke contracts
docs/                    evidence map and limitations
src/flight_delay_milp/   data, models, evaluation, optimization, CLI
tests/                   contract-focused unit tests
data/                    untracked raw and processed data
artifacts/               untracked immutable run directories
```

## Tests

```bash
uv run pytest
uv run ruff check .
```

On macOS, run those commands through `./reproduce.sh test`, or export
`DYLD_LIBRARY_PATH="$PWD/.native/lib"` first. The project-local bootstrap avoids a
global Homebrew dependency.

Tests cover schema normalization, aggregate target construction, leakage exclusions,
training-only balancing, metrics, capacity and structural MILP feasibility, BTS URL
validation, and immutable run directories.
