# LADI v2a decision comparison

This is a separate multilabel disaster-image task, preserved alongside the [AID scene-classification work](../aid_eval/README.md). The saved `pilot-native/` run contains five validation images evaluated by Luna and Jev. The official test split has not been evaluated; five pilot images cannot rank models. `config.json` retains the original preparation-time status, while the pilot manifest and exports record the completed work.

## Setup and data

Run commands from the repository root after the [base setup](../../README.md#quick-start). LADI additionally needs `datasets` and `pyarrow` to read the source. Local reference classifiers need `transformers` and `torch`.

```sh
pip install datasets pyarrow huggingface_hub
# Only for local reference classifier arms:
pip install transformers torch
```

The pinned source is `MITLL/LADI-v2-dataset` at revision `5f2dbfe8c466d32edafd1bab847ec5252309acdb`. Download the validation shards into a local directory:

```sh
hf download MITLL/LADI-v2-dataset \
  --repo-type dataset \
  --revision 5f2dbfe8c466d32edafd1bab847ec5252309acdb \
  --include 'data/validation/*' \
  --local-dir work/ladi-source
```

Source images, shards, and model weights are not committed. Partial source downloads must contain contiguous shards starting at zero so that global row IDs remain valid.

## Run a validation pilot

```sh
python outputs/ladi_eval/run.py \
  --source work/ladi-source \
  --split validation \
  --limit 5 \
  --image-condition native \
  --out outputs/ladi_eval/my-pilot-native \
  --arms decisions_luna jev_multimodal \
  --env-file .env
```

API arms use `OPENAI_API_KEY` and `TYPESAFE_API_KEY` from the environment or the explicit file. Add `mitll_small_reference mitll_large_reference` to `--arms` to include the reference checkpoints. Deployment checkpoints are excluded from the primary comparison because they trained on all splits.

Output directories must be empty; the runner refuses to overwrite saved results. Each prediction is flushed immediately. Compare arms only when source revision, split, row IDs, rubric, and image policy match.

## Image conditions

| Condition | Supplied images |
| --- | --- |
| `native` (primary) | Source resolution for Luna and MITLL; Jev preserves aspect ratio within 1,568 pixels, with minimum-dimension handling. |
| `matched` (secondary) | All arms receive the same prepared pixels under Jev's size constraints. |

Both conditions rotate EXIF orientation into pixels, remove metadata, and encode lossless PNG. Input hashes and dimensions are recorded. Model-specific image processors still apply, so matched inputs do not imply identical internal crops or tokenization. Repeat with `--image-condition matched` and a fresh output directory for that comparison.

## Scoring and publication

The runner writes truth, predictions, a manifest, and per-arm scores. To recalculate one saved score locally:

```sh
python outputs/ladi_eval/score.py \
  outputs/ladi_eval/pilot-native/truth.jsonl \
  outputs/ladi_eval/pilot-native/decisions_luna.jsonl
```

Scoring uses a fixed 0.5 threshold and reports macro/micro F1, per-label precision/recall/F1/support, Brier score, exact match, coverage, and latency p50/p95. Undefined label metrics are zero with support reported. Failures are retained and do not become negative labels. Quality metrics condition on complete valid images; when coverage differs, compare the common valid-ID intersection.

Latency includes API network/retries or local preprocessing/inference, excluding model loading. It is not a hardware-normalized speed comparison. Usage is retained, but cost calculation, paired bootstrap intervals, and statistical ranking are not implemented.

To publish a five-image validation pilot and verify its image attachments:

```sh
python outputs/ladi_eval/publish.py \
  outputs/ladi_eval/my-pilot-native \
  --env-file .env
```

Publishing needs `BRAINTRUST_API_KEY`, local images in the run directory, and access to `ladi-v2a-decision-comparison`. This publisher uses the fixed `ladi-validation-pilot` dataset and pilot descriptions; adapt those targets before publishing a different split or full benchmark.

## Protocol limitations

The damage rubric is provisional. Review definitions and errors against annotation guidance using validation data, then freeze the rubric, image policy, checkpoints, and thresholds before test inference. API arms use independent predicate/Noul questions, one batch per image. Local reference classifiers use sigmoid probabilities over all twelve labels. See `config.json`, `run.py`, and the saved pilot manifest for the exact definitions.
