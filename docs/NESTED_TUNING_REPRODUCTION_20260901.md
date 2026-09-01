# Nested-CV tuning reproduction report: 2026-09-01

## Outcome

The frozen nested-cross-validation enhancement completed successfully. The tuned
stack exceeds all four paper targets on the held-out random split, remains close to
the paper on the temporal arm, and reproduces the reported optimization effects.
All experiment processes exited after completion.

This result is strong confirmatory reproduction evidence, but it is not a fresh
independent generalization estimate: the same held-out partition was inspected in the
earlier baseline run. No decisions in this tuning run were made from those earlier
test metrics, and this run scored the held-out partition only after its selected
parameters and test-index hash were frozen.

## Frozen execution record

- Code commit: `d15c577a90854a0ca86f1c363db0a50719acdcab`
- Seed: 42
- Hardware: Apple M5 CPU, 10 cores, 24 GB RAM
- Command: `./reproduce.sh tune data/raw`
- Run: `artifacts/tuning/20260901T000710.279112Z-seed42-d49991d2`
- Start: `2026-09-01T00:07:10.331598Z`
- Completion: `2026-09-01T07:25:07.787147Z`
- Elapsed: 26,277.456 seconds (7 hours 17 minutes 57 seconds)
- Launcher: return code 0; neither timed out nor interrupted
- Parameter hash: `d34538223d67af782fdad3765e5ff9374cc28340d16d694e297added50ca1969`
- Test-index hash: `464af854d5884ec196edde0578e70abce0230d7121f100b95f958b708f64f5bb`

The official BTS ZIP SHA-256 remains
`7820e3fb740227102960c7297d70b64762ae118c1374ab2ce975e88b68443180`.

## Protocol

The data, target, leakage, split, optimization and evidence boundaries are inherited
from [PAPER_CONTRACT.md](PAPER_CONTRACT.md). The enhancement adds:

- 3 outer by 3 inner shuffled stratified folds;
- deterministic random search over versioned spaces for all 11 models;
- the one-standard-error ROC-AUC band followed by minimum Brier selection;
- training-only early stopping for XGBoost, LightGBM and CatBoost, with a 2,000-round
  ceiling and patience 50;
- a five-fold out-of-fold stack rebuilt from the tuned boosting bases;
- a final parameter freeze before one held-out evaluation.

The complete official data passed the frozen gate: 372,765 raw rows, 372,130 valid
rows, 43.2822% positives, 257,706 balanced training rows and 74,426 untouched test
rows. The training partition contained exactly 128,853 samples from each class.

## Training-only nested-CV evidence

These values aggregate the three outer validation folds. They are selection-pipeline
evidence that does not use the held-out test set.

| Model | ROC-AUC mean ± SD | PR-AUC | Brier | F1 |
| --- | ---: | ---: | ---: | ---: |
| Stacking | 0.8579 ± 0.0010 | 0.8585 | 0.1547 | 0.7727 |
| LightGBM | 0.8570 ± 0.0011 | 0.8576 | 0.1545 | 0.7712 |
| XGBoost | 0.8551 ± 0.0009 | 0.8557 | 0.1555 | 0.7705 |
| CatBoost | 0.8535 ± 0.0010 | 0.8534 | 0.1566 | 0.7698 |
| Random Forest | 0.8322 ± 0.0004 | 0.8300 | 0.1678 | 0.7490 |
| Extra Trees | 0.8307 ± 0.0019 | 0.8278 | 0.1687 | 0.7509 |
| KNN | 0.7894 ± 0.0012 | 0.7831 | 0.1876 | 0.7189 |
| MLP | 0.7852 ± 0.0022 | 0.7760 | 0.1892 | 0.7124 |
| Logistic Regression | 0.6734 ± 0.0029 | 0.6578 | 0.2270 | 0.6341 |
| Linear SVM | 0.6733 ± 0.0028 | 0.6577 | 0.2270 | 0.6344 |
| Gaussian NB | 0.5844 ± 0.0033 | 0.5809 | 0.2495 | 0.5996 |

The stack ranks first in mean outer-fold ROC-AUC with low fold-to-fold variance. Its
outer-fold PR-AUC is evaluated on balanced validation folds, so it should not be
directly compared numerically with the naturally imbalanced held-out PR-AUC.

## Frozen held-out results

| Model | ROC-AUC | PR-AUC | F1 | Brier |
| --- | ---: | ---: | ---: | ---: |
| Stacking | **0.8635** | **0.8346** | **0.7511** | 0.1518 |
| LightGBM | 0.8627 | 0.8337 | 0.7502 | **0.1515** |
| XGBoost | 0.8612 | 0.8312 | 0.7498 | 0.1523 |
| CatBoost | 0.8586 | 0.8276 | 0.7466 | 0.1541 |
| Random Forest | 0.8373 | 0.8002 | 0.7253 | 0.1653 |
| Extra Trees | 0.8349 | 0.7971 | 0.7236 | 0.1675 |
| KNN | 0.7972 | 0.7480 | 0.6932 | 0.1846 |
| MLP | 0.7849 | 0.7302 | 0.6822 | 0.1897 |
| Linear SVM | 0.6735 | 0.5972 | 0.5984 | 0.2270 |
| Logistic Regression | 0.6735 | 0.5973 | 0.5985 | 0.2270 |
| Gaussian NB | 0.5847 | 0.5165 | 0.5534 | 0.2536 |

The stack is the ROC-AUC, PR-AUC and F1 winner; LightGBM has a slightly lower Brier
score by 0.00028. This is consistent with the frozen model-selection rule because the
stack was selected from training-only nested evidence, not by held-out Brier.

### Comparison with the paper and prior baseline

The preregistered tolerances remain ±0.01 for random-split metrics and ±0.02 for
temporal metrics.

| Random-split metric | Paper | Prior baseline | Tuned | Tuned − paper | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| ROC-AUC | 0.8434 | 0.8267 | 0.8635 | +0.0201 | pass |
| PR-AUC | 0.8064 | 0.7856 | 0.8346 | +0.0282 | pass |
| F1 | 0.7331 | 0.7180 | 0.7511 | +0.0180 | pass |
| Brier | 0.1624 | 0.1709 | 0.1518 | −0.0106 | pass |

The tuned run resolves the prior baseline's failure to reach the paper's random-split
headline values. It should be described as reproducing or exceeding the reported
effect under the declared reconstruction, not as recovering the paper's unknown
original hyperparameters.

## Temporal arm

The frozen random-split parameters were applied to the 2005–2023 balanced training
partition; they were not retuned on 2024–2025. The temporal test contains 24,509 rows.

| Metric | Paper | Prior baseline | Tuned | Tuned − paper | 95% bootstrap CI | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ROC-AUC | 0.7202 | 0.7215 | 0.7261 | +0.0059 | [0.7200, 0.7324] | pass |
| PR-AUC | 0.6825 | 0.6908 | 0.6983 | +0.0158 | [0.6891, 0.7087] | pass |
| F1 | 0.6264 | 0.6150 | 0.6320 | +0.0056 | [0.6241, 0.6393] | pass |
| Brier | 0.2293 | 0.2194 | 0.2275 | −0.0018 | [0.2242, 0.2308] | pass |
| Accuracy | 0.6583 | 0.6559 | 0.6624 | +0.0041 | [0.6558, 0.6683] | pass |

The temporal arm passes every preregistered tolerance. Its Brier score is slightly
worse than the prior baseline but closer to the paper, while discrimination and F1
improve.

## Optimization effects

The comparison is the probability-aware `net_score` policy versus the frozen seeded
random policy. Percentage-point differences are against the paper.

| Effect | Paper | Prior baseline | Tuned | Tuned − paper | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Random-split cost reduction | 29.52% | 29.3941% | 29.8185% | +0.2985 pp | pass |
| Random-split risk reduction | 15.61% | 13.9291% | 16.2314% | +0.6214 pp | pass |
| Temporal cost reduction | 29.46% | 29.0343% | 29.4244% | −0.0356 pp | pass |
| Temporal risk reduction | 15.97% | 12.9044% | 16.2920% | +0.3220 pp | pass |

Both capacity-only and structural CBC arms returned `Optimal` on both splits. The
independent net-score ranking and capacity-only MILP total costs match exactly up to
floating-point output precision. The tuned temporal arm resolves the prior baseline's
out-of-tolerance temporal risk reduction.

## Selected parameters

The full machine-readable record is `best_parameters.json`. Key winners are:

| Model | Selected setting |
| --- | --- |
| Random Forest | 1,200 trees; unlimited depth; min leaf 4; sqrt features |
| Extra Trees | 300 trees; unlimited depth; min leaf 2; sqrt features |
| XGBoost | depth 8; learning rate 0.04; 1,642 rounds |
| LightGBM | 127 leaves; learning rate 0.12; 937 rounds |
| CatBoost | depth 10; learning rate 0.07; 1,999 rounds |
| KNN | 25 neighbors; distance weighting; Manhattan distance |
| MLP | 128–64 hidden units; tanh; alpha 1e-5; learning rate 0.003 |
| Stack meta-learner | logistic regression C = 0.0031623 |

CatBoost's median selected iteration is one below the 2,000-round ceiling, indicating
that this candidate received little practical truncation from early stopping. This is
reported as a protocol observation, not changed after seeing the result.

## Integrity and shutdown verification

- `TUNING_DONE.json` binds the selected-parameter hash to the exact held-out index
  hash before test scoring.
- Complete checkpoints: 33 outer, 11 final and 11 held-out.
- All 11 binary prediction files match their recorded SHA-256 hashes.
- Held-out and temporal metrics recompute from saved predictions to within
  `8.33e-17`; CSV probability round-trip error is at most `1.11e-16`.
- The nested summary recomputes exactly from the 33 outer-fold records.
- No `FAILED` marker exists.
- At final verification, the guarded process group, experiment Python process,
  joblib workers, CBC process and tmux session were all absent.

Key artifact hashes:

- `DONE.json`: `c984c7cfc0c9d6ff38c6cbceb05a78a521f4d5280bc2fdba1bfc6beb288708ec`
- `TUNING_DONE.json`: `7b04e387dc5aabf935c31481260ae28fefa490db4884b6b72b98bb980a3743fe`
- `nested_cv_summary.csv`: `eb340de52942a16ebce4f7e2d8cf13b4585c6993cf185159fbeca73caf4845a3`
- `test_model_metrics.csv`: `fb87905eb745e04414b60e25ee8558ac7a7fc4daffb4551990895947227dd742`
- `temporal_metrics_with_bootstrap_ci.csv`: `340fb34323df3ab79685771d8fc78796ae79a8c8396ee2b6bae7f503b5a99ddb`
- `random_split_optimization_strategies.csv`: `5a2d0086dc128b365572d3aa16c97d52f8d2646693b949cfad9577670684fe74`
- `temporal_split_optimization_strategies.csv`: `3a48fec99e284489b51f12ac9bea65364445ec4f9c6e120205853681df707446`

## Evidence boundary

The exact data counts, split, training-only selection pipeline, random and temporal
predictive effects, and main optimization effects are reproduced under a transparent,
versioned reconstruction. The paper still does not expose its original search spaces,
selected parameters, early-stopping split, raw export hash or all random draws.
Consequently, numerical agreement cannot identify the unpublished original procedure.

The strongest defensible conclusion is: **the paper's reported headline effects are
reproducible or exceeded by a leakage-aware, training-only nested-CV implementation on
the report-faithful BTS target, with temporal and optimization controls preserved.**
