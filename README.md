# AID image classification evaluations

Compare Luna Decisions, Jev 1.14, Cloudflare Clef, Liquid D1, and D1 Ultra on 30-class aerial scene classification. Braintrust stores the datasets, image attachments, traces, scores, and experiment comparisons. This repository contains the runners, frozen protocols, prediction exports, and analyses.

## Quick start

Run commands from the repository root with Python 3.12 or later. You need access to the existing Braintrust project, `aid-single-label-comparison`, and its datasets.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your Braintrust key and the provider keys you need.

# Start with the 30-image validation integration pilot.
python outputs/aid_eval/native_eval.py \
  --models clef \
  --dataset aid-validation-pilot-30 \
  --run-name my-clef-pilot-01 \
  --env-file .env
```

For the 2,000-image test split, omit `--dataset`. To run all providers, use `--models luna jev clef d1 d1-ultra`. These commands make model API calls and publish experiments. Pilot results are integration checks, not held-out benchmark estimates.

Use a fresh `--run-name` for each new evaluation. `--resume` recovers an interrupted run by replaying saved predictions and calling models only for missing cases. `--trials 3` makes three fresh calls per image; those trials are not independent test images.

## Credentials

| Service | Variables |
| --- | --- |
| Braintrust, required for evaluation | `BRAINTRUST_API_KEY` |
| Luna | `OPENAI_API_KEY` |
| Jev | `TYPESAFE_API_KEY` |
| Clef | `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_AUTH_TOKEN` |
| D1 and D1 Ultra | `LIQUID_AI_API_KEY` |

Clef also accepts `CLOUDFLARE_API_TOKEN` or `CLOUDFLARE_API_KEY`. Liquid also accepts `LIQUID_AI_KEY` or `LIQUID_API_KEY`.

The native, visual-decision, and web-image runners accept `--env-file` and `--liquid-key-file`. Files contain literal assignments, optionally prefixed with `export`; they are never sourced. Explicit files override environment values. Without a configured file, credentials come from the environment.

For maintenance scripts, set `AID_ENV_FILE=/absolute/path/to/credentials.env` and, optionally, `AID_LIQUID_KEY_FILE=/absolute/path/to/key-file`. There is no implicit dependency on a sibling repository or shell startup file. Historical AID and LADI runners document their own credential options.

## Workflows and results

| Workflow | Guide or saved results |
| --- | --- |
| Native scene classification, retries, scoring, and recovery | [AID guide](outputs/aid_eval/README.md) |
| Fresh five-model comparison | [Comparison export](outputs/aid_eval/native-fresh-01-comparison.json) |
| Controlled blur, contrast, and shifts | [Protocol and results](outputs/aid_eval/controlled-300-v1/results.md) |
| Added-blur detection and paired sharpness | [Results](outputs/aid_eval/visual-decisions-v1/results.md) |
| Web-image collection and human category review | [Review workflow](outputs/aid_eval/README.md#web-images-and-human-review) |
| Specialist baseline on 382 shared held-out images | [AiTLAS findings](outputs/aid_eval/specialist/aitlas/findings.txt) |
| Earlier multilabel disaster-image evaluation | [LADI guide](outputs/ladi_eval/README.md) |

The AID split is a fixed community partition: 6,000 train / 2,000 validation / 2,000 test. It is not the original paper's benchmark protocol. Native runs pin a Braintrust dataset version, check it against a local manifest, and hash-check image bytes before inference. Accuracy counts failed or invalid outputs as incorrect; valid-only metrics are labeled separately.

Costs are recorded per prediction. Liquid uses valid provider-reported `usage.cost`, falling back to configured token rates when absent. Other providers use configured estimates. Rates in the code describe these experiments and should be checked before budgeting a new run. Replayed rows are historical predictions; summing all experiments does not measure unique API spend.

## Repository layout

- `outputs/aid_eval/`: AID scripts, tests, manifests, and named experiment exports. Scripts remain alongside results to preserve existing paths.
- `outputs/ladi_eval/`: separate LADI runner, scoring, publishing, and validation pilot.
- `work/aid/`: original dataset preparation and report scripts.

Image binaries, checkpoints, credentials, local environments, and image-directory symlinks are ignored by Git. A clone includes audit exports, not the source images. The native runner reads remote Braintrust attachments; transformation and historical runners also require the local images described in the AID guide. Do not edit frozen prompts, labels, or inputs and reuse an existing prediction cache.

## Development

```sh
pip install -r requirements-dev.txt
python -m unittest discover -s outputs/aid_eval -p 'test_*.py'
ruff check .
ruff format --check .
```

Tests run locally with mocked provider calls and Braintrust logging disabled for the native Eval test. They cover payloads, MIME types, scoring, trial hooks, cost reconciliation, credential loading, human-label propagation, and paired statistical tests. They do not run paid evaluations or publish data.
