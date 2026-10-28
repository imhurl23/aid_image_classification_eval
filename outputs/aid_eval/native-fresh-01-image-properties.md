# Image-property analysis

Linear probability regressions with 30 scene-class fixed effects and all eight standardized image properties; HC3 robust SE. Paired model differences regress correctness(A)-correctness(B) on the same design. Holm correction across all 120 coefficient tests.

Each effect is an accuracy change in percentage points per one dataset-wide standard deviation, holding class and the other measured properties constant.

| Target | Property | Effect (pp/SD) | Adjusted p |
|---|---|---:|---:|
| clef_minus_d1-ultra | sharpness | +6.85 | 0.006618 |

## Feature definitions

- Brightness: mean Rec.709 luminance, 0–1.
- Contrast: standard deviation of luminance.
- Sharpness: log(1 + variance of luminance Laplacian × 255²).
- Entropy: 256-bin luminance Shannon entropy in bits.
- Edge density: fraction with luminance gradient magnitude above 0.05.
- Detail offset: distance of edge-weighted centroid from image center, normalized by corner distance.
- Central detail: fraction of total gradient magnitude in the central 50% × 50% of the frame.
- Clipped fraction: luminance pixels at or below 0.02 or at or above 0.98.

## Limitations

- Exploratory associations, not causal effects or interventions.
- Detail centroid and central edge share are proxies, not detected object positions.
- Sharpness/entropy can reflect scene texture, not only acquisition quality.
- Only the same 2,000 test images; no independent validation.
- HC3 assumes independence between images; unknown geographic correlation is not modeled.
- Confidence intervals shown are unadjusted; significance uses Holm-adjusted p values.

## Post-hoc sensitivity checks

These checks were selected after the sharpness finding; they are descriptive, not independent confirmation.

- log_laplacian: +6.85 percentage points per SD; unadjusted interval [3.5189844197831452, 10.174177492416037].
- rank_laplacian: +6.31 percentage points per SD; unadjusted interval [3.0767173616349757, 9.534541510997316].
- raw_laplacian: +2.10 percentage points per SD; unadjusted interval [-0.4882449001573419, 4.684077851192986].
- class_adjusted_only: +3.88 percentage points per SD; unadjusted interval [1.7888378747976836, 5.965193747331792].
