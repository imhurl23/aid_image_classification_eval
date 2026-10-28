# AID subgroup analysis

Fresh corrected native-fresh-01 runs only; 2,000 paired test images

Two-sided exact McNemar tests; Holm family-wise correction across all 320 subgroup/model-pair tests; alpha=0.05

| Class / encoding | N | Luna | Jev | Clef | D1 | D1 Ultra |
|---|---:|---:|---:|---:|---:|---:|
| Airport | 72 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| BareLand | 62 | 83.87% | 82.26% | 62.90% | 82.26% | 67.74% |
| BaseballField | 44 | 100.00% | 97.73% | 100.00% | 100.00% | 100.00% |
| Beach | 80 | 97.50% | 98.75% | 98.75% | 100.00% | 100.00% |
| Bridge | 72 | 81.94% | 93.06% | 95.83% | 91.67% | 95.83% |
| Center | 52 | 23.08% | 32.69% | 0.00% | 9.62% | 21.15% |
| Church | 48 | 25.00% | 18.75% | 68.75% | 68.75% | 68.75% |
| Commercial | 70 | 11.43% | 18.57% | 41.43% | 61.43% | 27.14% |
| DenseResidential | 82 | 53.66% | 95.12% | 87.80% | 37.80% | 54.88% |
| Desert | 60 | 93.33% | 96.67% | 90.00% | 65.00% | 95.00% |
| Farmland | 74 | 97.30% | 97.30% | 100.00% | 100.00% | 98.65% |
| Forest | 50 | 100.00% | 96.00% | 98.00% | 100.00% | 100.00% |
| Industrial | 78 | 91.03% | 85.90% | 93.59% | 94.87% | 94.87% |
| Meadow | 56 | 39.29% | 48.21% | 33.93% | 35.71% | 69.64% |
| MediumResidential | 58 | 44.83% | 58.62% | 46.55% | 65.52% | 63.79% |
| Mountain | 68 | 92.65% | 98.53% | 92.65% | 91.18% | 98.53% |
| Park | 70 | 42.86% | 91.43% | 81.43% | 88.57% | 68.57% |
| Parking | 78 | 98.72% | 98.72% | 100.00% | 98.72% | 98.72% |
| Playground | 74 | 14.86% | 0.00% | 2.70% | 0.00% | 5.41% |
| Pond | 84 | 98.81% | 92.86% | 94.05% | 90.48% | 95.24% |
| Port | 76 | 100.00% | 100.00% | 100.00% | 100.00% | 98.68% |
| RailwayStation | 52 | 94.23% | 100.00% | 100.00% | 98.08% | 98.08% |
| Resort | 58 | 60.34% | 65.52% | 68.97% | 82.76% | 84.48% |
| River | 82 | 85.37% | 89.02% | 97.56% | 87.80% | 96.34% |
| School | 60 | 76.67% | 70.00% | 75.00% | 83.33% | 63.33% |
| SparseResidential | 60 | 65.00% | 80.00% | 60.00% | 73.33% | 48.33% |
| Square | 66 | 54.55% | 50.00% | 33.33% | 13.64% | 50.00% |
| Stadium | 58 | 98.28% | 100.00% | 98.28% | 100.00% | 98.28% |
| StorageTanks | 72 | 77.78% | 90.28% | 88.89% | 88.89% | 98.61% |
| Viaduct | 84 | 95.24% | 86.90% | 91.67% | 90.48% | 90.48% |
| JPEG | 1905 | 74.96% | 78.90% | 78.53% | 77.64% | 80.16% |
| PNG | 95 | 61.05% | 80.00% | 68.42% | 69.47% | 54.74% |

## Significant pairwise differences after Holm correction

| Group | Model A | Model B | A minus B (points) | Discordant A/B wins | Adjusted p |
|---|---|---|---:|---:|---:|
| DenseResidential | jev | d1 | +57.32 | 47/0 | 4.547e-12 |
| DenseResidential | clef | d1 | +50.00 | 41/0 | 2.901e-10 |
| DenseResidential | luna | jev | -41.46 | 0/34 | 3.702e-08 |
| Park | luna | jev | -48.57 | 0/34 | 3.702e-08 |
| DenseResidential | jev | d1-ultra | +40.24 | 33/0 | 7.357e-08 |
| Park | luna | d1 | -45.71 | 0/32 | 1.467e-07 |
| Commercial | luna | d1 | -50.00 | 1/36 | 1.736e-07 |
| DenseResidential | luna | clef | -34.15 | 0/28 | 2.332e-06 |
| DenseResidential | clef | d1-ultra | +32.93 | 27/0 | 4.649e-06 |
| Park | luna | clef | -38.57 | 0/27 | 4.649e-06 |
| JPEG | luna | d1-ultra | -5.20 | 103/202 | 4.691e-06 |
| Commercial | jev | d1 | -42.86 | 1/31 | 4.748e-06 |
| Church | jev | clef | -50.00 | 0/24 | 3.672e-05 |
| Church | jev | d1 | -50.00 | 0/24 | 3.672e-05 |
| Church | jev | d1-ultra | -50.00 | 0/24 | 3.672e-05 |
| Square | jev | d1 | +36.36 | 24/0 | 3.672e-05 |
| Square | luna | d1 | +40.91 | 29/2 | 0.0001407 |
| Square | d1 | d1-ultra | -36.36 | 1/25 | 0.0002438 |
| Commercial | d1 | d1-ultra | +34.29 | 26/2 | 0.0009158 |
| Desert | jev | d1 | +31.67 | 19/0 | 0.001148 |
| SparseResidential | jev | d1-ultra | +31.67 | 19/0 | 0.001148 |
| Church | luna | clef | -43.75 | 1/22 | 0.001711 |
| Desert | d1 | d1-ultra | -30.00 | 0/18 | 0.002274 |
| Park | luna | d1-ultra | -25.71 | 0/18 | 0.002274 |
| Meadow | clef | d1-ultra | -35.71 | 1/21 | 0.003246 |
| JPEG | luna | jev | -3.94 | 110/185 | 0.004392 |
| Center | jev | clef | +32.69 | 17/0 | 0.004486 |
| Desert | luna | d1 | +28.33 | 17/0 | 0.004486 |
| Meadow | d1 | d1-ultra | -33.93 | 1/20 | 0.006126 |
| Park | jev | d1-ultra | +22.86 | 16/0 | 0.008881 |
| PNG | jev | d1-ultra | +25.26 | 29/5 | 0.01118 |
| Church | luna | d1 | -43.75 | 3/24 | 0.01423 |
| Church | luna | d1-ultra | -43.75 | 3/24 | 0.01423 |
| Desert | clef | d1 | +25.00 | 15/0 | 0.01752 |
| SparseResidential | d1 | d1-ultra | +25.00 | 15/0 | 0.01752 |
| StorageTanks | luna | d1-ultra | -20.83 | 0/15 | 0.01752 |
| PNG | luna | jev | -18.95 | 2/20 | 0.0344 |
| Park | d1 | d1-ultra | +20.00 | 14/0 | 0.03455 |
| JPEG | luna | clef | -3.57 | 125/193 | 0.04598 |

## Limits

- Exploratory subgroup analysis, not prespecified confirmatory tests.
- No full matched five-model runs for train or validation.
- Encoding is a derived technical subgroup, not a semantic dataset tag; class mix may confound encoding comparisons.
- A significant within-group comparison does not establish a model-by-group interaction.
- No correction can address geographic correlation without location metadata.

## Metadata audit

The pinned Braintrust dataset has no tags. Metadata fields are source_id, sha256, split, and dimensions. All 2,000 rows are test images at 600×600. Class labels are the expected outputs. No location, season, or resolution strata are available in these records.

All 95 PNG images are residential: 19 DenseResidential, 20 MediumResidential, and 56 SparseResidential. Therefore the PNG/JPEG accuracy difference must not be interpreted as a causal encoding effect. A matched re-encoding experiment would be required to isolate encoding.

There are 39 Holm-significant pairwise comparisons: 34 within 10 scene classes and 5 within encoding subsets. Other classes failing to reach significance are not evidence of equivalence. Train (6,000) and validation (2,000) splits exist but lack complete comparable five-model predictions; this report does not test performance differences across those splits.
