# Controlled image robustness results

Paired 300-image analysis; exact McNemar within model; exact paired model-label-swap tests on differences in accuracy changes between models. Holm correction across all 90 planned tests.

Negative change means accuracy loss vs fresh PNG control; equal-class balanced subset. Single response per condition.

Prompt: **aid-definitions-v1** for both fresh controls and transformed images.

Pointwise 95% paired source bootstrap, stratified by reference class; 5,000 replicates, seed 20261028. Intervals are not multiplicity-adjusted.

| Condition | Luna | Jev | Clef | D1 | D1 Ultra |
|---|---:|---:|---:|---:|---:|
| control | 84.3% | 86.7% | 84.0% | 84.0% | 86.0% |
| blur_1 | 84.0% (-0.3 pp) | 85.3% (-1.3 pp) | 83.3% (-0.7 pp) | 83.0% (-1.0 pp) | 86.0% (+0.0 pp) |
| blur_2 | 80.0% (-4.3 pp) | 82.3% (-4.3 pp) | 79.3% (-4.7 pp) | 82.7% (-1.3 pp) | 84.7% (-1.3 pp) |
| contrast_0.5 | 83.7% (-0.7 pp) | 84.7% (-2.0 pp) | 80.3% (-3.7 pp) | 83.0% (-1.0 pp) | 87.0% (+1.0 pp) |
| contrast_1.5 | 85.3% (+1.0 pp) | 87.7% (+1.0 pp) | 82.3% (-1.7 pp) | 82.3% (-1.7 pp) | 88.0% (+2.0 pp) |
| shift_left_60 | 79.7% (-4.7 pp) | 84.7% (-2.0 pp) | 77.0% (-7.0 pp) | 82.3% (-1.7 pp) | 85.7% (-0.3 pp) |
| shift_right_60 | 83.0% (-1.3 pp) | 85.7% (-1.0 pp) | 80.3% (-3.7 pp) | 85.3% (+1.3 pp) | 86.3% (+0.3 pp) |

## Run audit

| Model / experiment | Valid predictions | Recorded or estimated cost |
|---|---:|---:|
| [luna](https://www.braintrust.dev/app/eval_engineering/p/aid-single-label-comparison/experiments/controlled-definitions-v1-luna__luna) | 2,099/2,100 | $0.2926 |
| [jev](https://www.braintrust.dev/app/eval_engineering/p/aid-single-label-comparison/experiments/controlled-definitions-v1-jev__jev) | 2,100/2,100 | $0.2200 |
| [clef](https://www.braintrust.dev/app/eval_engineering/p/aid-single-label-comparison/experiments/controlled-definitions-v1-clef__clef) | 2,100/2,100 | $0.8986 |
| [d1](https://www.braintrust.dev/app/eval_engineering/p/aid-single-label-comparison/experiments/controlled-definitions-v1-d1__d1) | 2,100/2,100 | $0.0604 |
| [d1-ultra](https://www.braintrust.dev/app/eval_engineering/p/aid-single-label-comparison/experiments/controlled-definitions-v1-d1-ultra__d1-ultra) | 2,100/2,100 | $0.1269 |

Total recorded or estimated model cost: $1.5986. Braintrust charges are excluded.

[Accuracy chart](accuracy-deltas.png) · [Calibration chart](calibration-deltas.png) · [Remote verification](verification.json)

One Luna refusal is retained as incorrect. No historical predictions were replayed.

## Significant comparisons

- clef, shift_left_60: -7.00 pp vs control; Holm p=0.009334.

## Interpretation limits

- All controls and variants are lossless PNG with preserved decoded control pixels; not the earlier original-file-encoding experiment.
- The balanced 300-image sample is not the full test set; rates should not be compared directly to the earlier full-test leaderboard.
- Shifts crop one boundary and reflect another. Effects include cropping/fill artifacts, not solely object position.
- A single call per condition cannot separate stochastic output variation from a small transformation effect.
- Accuracy includes every attempt, counting refusals and errors as wrong. Calibration uses each model's common valid-source intersection across all seven conditions, with 10 equal-width confidence bins.
- No detected significance is not evidence of equivalence; this design is less sensitive to small effects.
- These controlled findings use the same source test pool that motivated the earlier observational analysis; confirm on independent images.
