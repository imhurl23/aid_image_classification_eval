# AID evaluation guide

Start with the [repository setup](../../README.md#quick-start). Commands below run from the repository root. `config.json` fixes the 30 labels and source revisions; `split-manifest.json` fixes the community train/validation/test partition.

## Native scene classification

```sh
python outputs/aid_eval/native_eval.py \
  --models luna jev clef d1 d1-ultra \
  --run-name my-five-model-01 \
  --env-file .env
```

The runner uses `braintrust.Eval` with the remote dataset, a model task, and `accuracy` / `valid_output` scorers. Each model gets a separate experiment and a local JSONL prediction export plus summary in the run directory.

| Option | Meaning |
| --- | --- |
| `--dataset` | Defaults to `aid-test-2000`; use `aid-validation-pilot-30` for integration checks. |
| `--dataset-version` | Reuse an exact dataset version; otherwise pin the current version at startup. |
| `--cases-file` | Frozen local cases for a custom dataset. |
| `--protocol-file` | Record the custom transformation protocol in the run manifest. |
| `--trials` | Fresh calls per image, default 1. Analyze repeated trials at image level. |
| `--max-concurrency` | Concurrent tasks, default 6. |
| `--base-experiment` | UI comparison baseline, default `test-2000-rerun-01__jev`. Choose a matching split. |
| `--resume` | Replay saved predictions into a recovery experiment; call only missing cases. |

Use a new run name for fresh predictions. Resume requires the same manifest, including dataset version, model list, and trial count. Replayed rows are marked and do not create fresh model-call spans. API requests and attachment downloads retry up to three times for supported transient failures.

The native runner reads original bytes from Braintrust attachments, checks SHA256 hashes against the manifest, and detects MIME type without resizing. Child model spans contain image references, instructions, responses, usage, latency, errors, and cost. Request base64 and authorization headers are not logged.

### Scoring and comparisons

Accuracy includes failed requests and invalid outputs as incorrect. `valid_output` measures coverage. The parser validates all 30 probabilities, permits total mass within 0.02 of one, then normalizes. Local summaries report valid-only macro F1, log loss, and confusion matrices.

The five `native-fresh-01-*` runs use dataset version `1000198073523310044`, corrected MIME types, and the same frozen question and retry policy. Their [comparison](native-fresh-01-comparison.json), [subgroup analysis](native-fresh-01-subgroups.md), and [image-property analysis](native-fresh-01-image-properties.md) are saved separately from earlier runs. To reuse that version, add `--dataset-version 1000198073523310044` to a new run.

An audit found 95 test files and one pilot file with `.jpg` names but PNG bytes. Earlier runs labeled those bytes as JPEG. `repair_image_types.py` published correctly typed attachments without changing bytes, labels, or row IDs. Treat this as a transport difference when comparing old and new experiments.

## Controlled blur, contrast, and shifts

`prepare_robustness.py` selects 10 test images per class by a frozen SHA256 seed, independent of predictions. Seven conditions produce 2,100 rows: control, Gaussian blur radii 1/2, contrast factors 0.5/1.5, and horizontal shifts of 60 pixels in either direction with reflected fill. Controls and variants are 600×600 lossless RGB PNGs.

Preparation requires the original images under `outputs/aid_eval/test-2000-rerun-01/images/`. Those files are not in Git. Restore the matching source bytes before preparing variants; hashes are checked against the cases manifest.

```sh
python outputs/aid_eval/prepare_robustness.py --env-file .env
python outputs/aid_eval/native_eval.py \
  --models luna jev clef d1 d1-ultra \
  --run-name my-controlled-01 \
  --dataset aid-controlled-300-v1 \
  --cases-file outputs/aid_eval/controlled-300-v1/cases.json \
  --protocol-file outputs/aid_eval/controlled-300-v1/protocol.json \
  --env-file .env
```

[Saved results](controlled-300-v1/results.md) use paired accuracy changes from the fresh control. The 90-test family includes 30 within-model tests and 60 between-model differences of changes, with Holm correction. Overall experiment accuracy mixes all seven conditions; filter by condition. Shift effects include cropping and reflected padding, and the balanced control is not directly comparable to full-test accuracy.

Two interrupted runs each required one additional image call, for Luna and D1 Ultra. Their complete `__resumed` experiments preserve the other 2,099 predictions. `analyze_robustness.py` reads the fixed `controlled-300-v1-{model}` exports; a differently named new run requires updating its input paths before analysis.

## Added-blur and paired-sharpness decisions

These tasks reuse the 300 controlled-experiment sources and radius-2 blur. Added-blur detection has original/blur rows; paired sharpness has both A/B orders. Each task has 600 rows. Requests include only images and fixed questions/options, without condition metadata or filenames.

```sh
export AID_ENV_FILE="$PWD/.env"
python outputs/aid_eval/visual_decisions.py --prepare
python outputs/aid_eval/visual_decisions.py --model luna --pilot
python outputs/aid_eval/visual_decisions.py --model luna
python outputs/aid_eval/analyze_visual_decisions.py
python outputs/aid_eval/verify_visual_decisions.py
```

Run the model commands for `luna`, `jev`, `clef`, `d1`, and `d1-ultra`. Preparation and inference require the local controlled images. Dataset versions are frozen in `visual-decisions-v1/datasets.json`. Repeated execution replays cached predictions, including pilot rows; use a separately versioned protocol and directory for a fresh experiment. The verification script checks remote Braintrust data.

[Analysis](visual-decisions-v1/results.md) clusters both decisions by source image, uses exact paired sign-flip tests with Holm correction across 20 comparisons, and bootstraps source clusters for unadjusted intervals. Added-blur truth is treatment assignment, not human perceptual judgment. The original is the operational sharper reference; original images may already be soft. A-choice bias and swap consistency are descriptive diagnostics. These are exploratory results on a reused subset.

## Web images and human review

The `web-aid-v1` batch contains 1,000 Wikimedia Commons candidates, retrieved using aerial/drone/satellite searches targeting the 30 classes. Retrieval targets are **not ground truth**. Source pages, URLs, authors, licenses, and original dimensions are retained. Exact hashes, page IDs, and dHash distance ≤2 filter duplicates; they do not establish geographic independence or absence from model training.

All providers receive the same RGB PNG bytes, preserving aspect ratio with a longest side of at most 600 pixels. Jev requires at least 280 pixels per side: 22 initial images needed neutral-gray padding and were rerun for all models. The original cases and predictions remain in `cases-before-padding.json` and `pre-padding-predictions.jsonl`; `padding-repair.json` records the change.

```sh
export AID_ENV_FILE="$PWD/.env"
python outputs/aid_eval/collect_web_aid.py
python outputs/aid_eval/web_aid_eval.py --limit 3
python outputs/aid_eval/web_aid_eval.py
```

Collection downloads images; inference requires those local images and all five provider keys. The runner caches predictions by image/model and checks input hashes before reuse. It also publishes image review records and, after a full run, per-model presentation rows.

### Label once, then synchronize and score

The current review source of truth is the image-level `human_review` record. In the **Web images — label once** queue, open the image and set `web_scene_category` to one of the 30 AID classes or `Ambiguous`, `Other`, `NotAerial`, or `Unusable`. The category is stored in `expected`. The category guide is in [labeling_guide.py](labeling_guide.py).

After entering or correcting labels:

```sh
python outputs/aid_eval/sync_image_labels.py
python outputs/aid_eval/web_aid_eval.py --score-reviewed
```

Synchronization propagates each authoritative label to the five model rows and inference root, updating `human_category_accuracy`. It also clears that score when a label is removed or excluded. Scoring publishes a versioned `web-aid-v1-human-reviewed` dataset and evaluates cached predictions without new model calls. Unreviewed and excluded images are not counted as wrong. These commands run manually, not on a schedule.

The older **Web images — human category review** view and model review queue show predictions and are not blind review. `publish_model_review_rows.py` maintains those cached presentations. `setup_label_once.py` migrates the historical queue, imports unanimous prior labels, and flags disagreements for reassessment. Queue setup scripts contain the original project/reviewer IDs and are migration utilities, not general setup commands for another account.

Each scored dataset version is reconciled against the current eligible labels. Removing a label or changing it to an exclusion removes that image from the next dataset version; prior versions and experiment results remain historical.

## Cost accounting and repairs

`cost_accounting.py` uses a finite, nonnegative Liquid `usage.cost` when available. `recorded_cost_usd` and the compatibility field `estimated_api_cost_usd` contain the chosen value; `token_estimated_cost_usd` retains the independent configured-rate calculation. `cost_source` identifies provider cost or token estimate. The configured input rates for D1 and D1 Ultra are $0.04/M and $0.042/M, respectively, with no output-token charge in this protocol.

The original D1 token estimate doubled provider-reported cost. The historical correction is recorded in `liquid-cost-repair-applied.json` and `liquid-cost-verification.json`. `repair_liquid_costs.py` defaults to a local audit; `--apply` changes saved artifacts and `--remote` with `--apply` also updates Braintrust. Repairs make no new inference calls.

For cost audits, filter model spans with `metrics.provider_cost_available = 1` and compare `metrics.recorded_to_provider_cost_ratio` and `metrics.cost_discrepancy`. The expected ratio is one; both-zero costs also have ratio one. Missing provider costs have no ratio. Tolerances are relative `1e-6` and absolute USD `1e-12`. Changed historical amounts are retained in `output.cost_before_reconciliation_usd`. Replayed records must not be counted as fresh API spend.

Other `repair_*`, `cleanup_*`, `setup_*`, and publishing scripts can mutate historical artifacts or remote records. Their names describe maintenance operations, not verification checks; inspect their fixed targets before running them.

## Historical runs and specialist baseline

- `eval.py`: original balanced 30-image validation pilot; requires `--env-file` and local `images/`. Reuses saved outputs and publishes stable row IDs.
- `full_test.py`: historical Luna/Jev 2,000-image runner. Downloads images, resumes saved outputs, and publishes results. Use a fresh `--run-name` for fresh calls.
- `test-2000/`: original full run, with 74.30% Luna and 79.40% Jev accuracy and historical combined estimated cost of $0.2885. Named reruns remain separate.
- `specialist/aitlas/`: verified ResNet-50 baseline on the **382-image shared held-out intersection**. Of our 2,000 test images, 1,217 occur in AiTLAS training and 401 in its validation split; the full set is unsuitable for this checkpoint. See [findings and provenance](specialist/aitlas/findings.txt). Do not compare the 382-image specialist score to full-test API scores.
- `team_travel_eval.py`: illustrative photos without reference labels, excluded from benchmark scores. Requires `--source-dir` and `--instructions-file`; the frozen prompt hash is checked. Use `AID_ENV_FILE` for credentials.

The original student checkpoint investigation in `specialist/findings.txt` remains historical; AiTLAS supplied the usable split manifests. Model weights are not committed. Specialist inference additionally requires `torch` and `torchvision`.

## Checks

```sh
python -m unittest discover -s outputs/aid_eval -p 'test_*.py'
ruff check .
ruff format --check .
```

Local tests do not need credentials. Remote verification scripts require Braintrust access. The [LADI evaluation](../ladi_eval/README.md) is a separate task and project.
