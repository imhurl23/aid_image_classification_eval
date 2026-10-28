# Controlled image robustness results

Paired 300-image analysis; exact McNemar within model; exact paired model-label-swap tests on differences in accuracy changes between models. Holm correction across all 90 planned tests.

Negative change means accuracy loss vs fresh PNG control; equal-class balanced subset. Single response per condition.

| Condition | Luna | Jev | Clef | D1 | D1 Ultra |
|---|---:|---:|---:|---:|---:|
| control | 75.7% | 79.3% | 76.7% | 75.3% | 78.0% |
| blur_1 | 75.0% (-0.7 pp) | 78.3% (-1.0 pp) | 74.3% (-2.3 pp) | 76.0% (+0.7 pp) | 78.3% (+0.3 pp) |
| blur_2 | 71.0% (-4.7 pp) | 77.3% (-2.0 pp) | 68.3% (-8.3 pp) | 74.7% (-0.7 pp) | 76.3% (-1.7 pp) |
| contrast_0.5 | 74.0% (-1.7 pp) | 77.0% (-2.3 pp) | 73.0% (-3.7 pp) | 75.3% (+0.0 pp) | 77.0% (-1.0 pp) |
| contrast_1.5 | 78.3% (+2.7 pp) | 78.3% (-1.0 pp) | 77.0% (+0.3 pp) | 77.0% (+1.7 pp) | 79.3% (+1.3 pp) |
| shift_left_60 | 75.3% (-0.3 pp) | 76.7% (-2.7 pp) | 72.7% (-4.0 pp) | 74.7% (-0.7 pp) | 78.3% (+0.3 pp) |
| shift_right_60 | 72.7% (-3.0 pp) | 75.7% (-3.7 pp) | 74.7% (-2.0 pp) | 77.3% (+2.0 pp) | 76.7% (-1.3 pp) |

## Significant comparisons

- clef, blur_2: -8.33 pp vs control; Holm p=0.003713.

## Interpretation limits

- All controls and variants are lossless PNG with preserved decoded control pixels; not the earlier original-file-encoding experiment.
- The balanced 300-image sample is not the full test set; rates should not be compared directly to the earlier full-test leaderboard.
- Shifts crop one boundary and reflect another. Effects include cropping/fill artifacts, not solely object position.
- A single call per condition cannot separate stochastic output variation from a small transformation effect.
- No detected significance is not evidence of equivalence; this design is less sensitive to small effects.
- These controlled findings use the same source test pool that motivated the earlier observational analysis; confirm on independent images.
