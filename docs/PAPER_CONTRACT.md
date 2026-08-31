# Paper-to-code contract

This document records what can and cannot be reconstructed from the supplied paper.
Section names refer to the paper rather than to an unpublished source path.

## Data

| Item | Paper evidence | Implementation |
| --- | --- | --- |
| Source | Data and Methodology: BTS Airline On-Time Performance: Delay Cause | Official BTS export only |
| Unit | Unique carrier-airport-month aggregate | One row remains one aggregate unit |
| Date boundary | January 2005-January 2025 | Inclusive filter in `configs/paper.yaml` |
| Raw schema | 21 columns | Canonical lower-case schema is validated |
| Raw count | 372,765 | Reported as a validation target, never forced |
| Valid count | 372,130 after valid positive `arr_flights` | Applied before target construction |
| Main split | Stratified 80/20, seed 42 | `train_test_split` on the aggregate target |
| Balancing | Random undersampling on training only | Test distribution remains untouched |
| Temporal split | 2005-2023 train, 2024-2025 test | Separate formal arm |

## Target contradiction

The dataset description defines a positive aggregate unit when more than 15% of its
arrivals were delayed by at least 15 minutes. A later equation instead defines a
single-flight indicator. The source table contains counts (`arr_del15`, `arr_flights`),
not individual flight outcomes, so the target must be an aggregate delay rate.

The official full-range export gives 372,765 raw rows. Applying the prose threshold
strictly while dropping missing outcome counts gives 371,837 valid rows and a 63.6881%
positive rate, contradicting every reported split and class-count table. In contrast,
retaining the 293 blank `arr_del15` cells as zero-delay counts and using a strict 20%
threshold reproduces all reported counts exactly: 372,130 valid rows; 211,064/161,066
total class counts; 168,851/128,853 training counts; 42,213/32,213 test counts; and
257,706 samples after training-only undersampling. The 293 blank-count rows also have
zero `arr_delay` and zero delay-cause counts in the official export. The report-faithful
formal target is therefore:

```text
arr_del15 = 0 when its count is blank
delay_rate = arr_del15 / arr_flights
target = 1 if delay_rate > 0.20 else 0
```

The strict `>` operator follows the prose phrase “more than”. The 20% threshold is an
evidence-backed reconstruction of the experiment that produced the paper's tables,
not a claim that the paper's 15% prose is correct. The missing-count policy, target
rule and threshold are versioned configuration values.

## Leakage boundary

Only these fields enter a model:

- `year`
- `month`
- `carrier`
- `airport`
- `arr_flights`

The outcome count and all realized cancellation, diversion, cause and delay fields are
excluded. Carrier and airport are ordinal encoded for the paper's tree and
distance/neural paths; the standalone logistic-regression and linear-SVM baselines use
one-hot encoding, matching the Results table.

## Models

The 11 candidates are Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, KNN,
Gaussian Naive Bayes, MLP, one-hot Logistic Regression, calibrated one-hot Linear SVM,
and a stack. The stack obtains out-of-fold probabilities from XGBoost, LightGBM and
CatBoost with shuffled stratified folds and trains a logistic-regression meta-learner.

The paper says hyperparameters were cross-validated but does not provide the search
space or winners. `configs/paper.yaml` contains conservative reconstructed settings.
They are hypotheses, not quotations from the paper.

## Optimization

For unit `f`, the implementation minimizes:

```text
(1 - x_f) * value_f + penalty * probability_f * x_f
```

subject to `sum(x) <= floor(capacity * n)` and optional mandatory-unit and carrier
minimum-quota constraints. The paper fixes values to uniform integers from USD
1,000-5,000 but does not provide their realized vector; this reproduction uses NumPy's
documented generator with seed 42 and saves the vector with every run.

The capacity-only optimum is also computed by net-score ranking as a cross-check. The
structural arm uses a 40% carrier minimum and a seeded 5% mandatory set. PuLP calls CBC
for the constrained solution.

## Reported targets, not acceptance thresholds

The paper reports stacking ROC-AUC 0.8434, PR-AUC 0.8064, F1 0.7331 and Brier 0.1624;
29.52% cost reduction and 15.61% risk reduction in the random-split optimization; and
temporal ROC-AUC 0.7202. Because the original exact code, raw export hash and tuned
hyperparameters are absent, these values are comparison targets. A run is considered
technically successful when it completes the frozen pipeline with valid schemas,
leakage gates, metrics and optimization feasibility—not merely when it numerically
matches a table.

## Known evidence gaps

- Exact BTS export query and raw-file hash.
- Exact hyperparameter search spaces, winners and early-stopping split.
- Exact random generator used for commercial values and mandatory units.
- Exact scenario grid.
- Definition of the paper's unnamed “baseline” in the main MILP table.
- Whether the reported stack's logistic meta-learner was followed by an additional
  calibrator. Logistic stacking is not, by itself, identical to Platt scaling.
- The paper calls the binary-label oracle an upper bound but reports it below the
  probability method. This repository names it `binary_label_policy`, not an oracle.
