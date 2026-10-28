# New visual decisions

300 independent source clusters; exact paired sign flips of per-source correct-count differences (two decisions/source); Holm across 20 planned model comparisons. 95% percentile bootstrap CIs resample source clusters, unadjusted. Both A/B orders included.

| Model | Added-blur accuracy | AUROC | Paired sharpness accuracy | Pair swap consistency |
|---|---:|---:|---:|---:|
| luna | 62.3% | 0.926 | 99.3% | 98.7% |
| jev | 94.8% | 0.994 | 99.7% | 99.3% |
| clef | 90.0% | 0.970 | 100.0% | 100.0% |
| d1 | 93.7% | 0.992 | 99.7% | 99.3% |
| d1-ultra | 93.7% | 0.984 | 99.7% | 99.3% |

## Significant model differences

- added-blur: luna minus jev -32.50 pp; Holm p=7.965e-58.
- added-blur: luna minus clef -27.67 pp; Holm p=1.78e-38.
- added-blur: luna minus d1 -31.33 pp; Holm p=1.122e-52.
- added-blur: luna minus d1-ultra -31.33 pp; Holm p=1.122e-52.
- added-blur: jev minus clef +4.83 pp; Holm p=0.003107.

## Interpretation

- Added-blur labels encode the known treatment, not a human judgment of whether an image is blurry. Controls can have natural softness.
- Paired sharpness treats the original as the sharper reference. Both orders are tested; swap consistency measures whether the model picks the same physical image after reversal (even if wrong).
- Each task has 600 rows, but only 300 source images. Statistical tests and intervals preserve that clustering.
- Percentile bootstrap intervals can collapse to 100–100% when no errors are observed (Clef paired sharpness); this is a bootstrap boundary limitation, not certainty about population accuracy.
- Same exploratory source subset as the robustness experiment; no independent confirmation or generalization to other distortions.
- API failures count as incorrect. AUROC and Brier use valid outputs only. No prompt tuning or model-specific prompt variants.
- Pilot predictions are reused in full experiments, explicitly marked as replays. Cost counts each prediction once.
- Estimated API cost: $0.468399 (provider usage.cost for Liquid, configured token rates otherwise; excludes Braintrust storage).

## Braintrust experiments

- [added-blur: luna](https://www.braintrust.dev/app/eval_engineering/p/aid-single-label-comparison/experiments/visual-decisions-v1-added-blur__luna)
- [added-blur: jev](https://www.braintrust.dev/app/eval_engineering/p/aid-single-label-comparison/experiments/visual-decisions-v1-added-blur__jev)
- [added-blur: clef](https://www.braintrust.dev/app/eval_engineering/p/aid-single-label-comparison/experiments/visual-decisions-v1-added-blur__clef)
- [added-blur: d1](https://www.braintrust.dev/app/eval_engineering/p/aid-single-label-comparison/experiments/visual-decisions-v1-added-blur__d1)
- [added-blur: d1-ultra](https://www.braintrust.dev/app/eval_engineering/p/aid-single-label-comparison/experiments/visual-decisions-v1-added-blur__d1-ultra)
- [paired-sharpness: luna](https://www.braintrust.dev/app/eval_engineering/p/aid-single-label-comparison/experiments/visual-decisions-v1-paired-sharpness__luna)
- [paired-sharpness: jev](https://www.braintrust.dev/app/eval_engineering/p/aid-single-label-comparison/experiments/visual-decisions-v1-paired-sharpness__jev)
- [paired-sharpness: clef](https://www.braintrust.dev/app/eval_engineering/p/aid-single-label-comparison/experiments/visual-decisions-v1-paired-sharpness__clef)
- [paired-sharpness: d1](https://www.braintrust.dev/app/eval_engineering/p/aid-single-label-comparison/experiments/visual-decisions-v1-paired-sharpness__d1)
- [paired-sharpness: d1-ultra](https://www.braintrust.dev/app/eval_engineering/p/aid-single-label-comparison/experiments/visual-decisions-v1-paired-sharpness__d1-ultra)
