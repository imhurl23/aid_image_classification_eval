# AID image classification evaluations

AID stands for **Aerial Image Dataset**, a [30-class aerial scene dataset](https://captain-whu.github.io/AID/).

Compare Luna Decisions, Jev 1.14, Cloudflare Clef, Liquid D1, and D1 Ultra on 30-class aerial scene classification. Braintrust stores the datasets, image attachments, traces, scores, and experiment comparisons. This repository contains the runners, frozen protocols, prediction exports, and analyses.

## Quick start

Run commands from the repository root with Python 3.12 or later. Saved results and local tests need no credentials. To run a new evaluation, use your own Braintrust project and keys for the selected providers. Model availability depends on your provider account; model IDs record the versions used in these experiments.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in .env with your Braintrust key and the provider keys you need.

# Download and hash-check the frozen pilot images, then upload to your project.
# Omit --publish to download only (no Braintrust credentials required).
python outputs/aid_eval/prepare_dataset.py \
  --split pilot --images-dir work/aid-pilot-images \
  --project my-aid-evaluation --publish --env-file .env

# Start with the 30-image validation integration pilot.
python outputs/aid_eval/native_eval.py \
  --models clef \
  --project my-aid-evaluation \
  --instructions-file outputs/aid_eval/controlled-definitions-v1/prompt.txt \
  --dataset aid-validation-pilot-30 \
  --run-name my-clef-pilot-01 \
  --env-file .env
```

For the 2,000-image test split, prepare with `--split test` and a separate image directory, then omit `--dataset` in the native runner. Dataset preparation refuses to modify a nonempty dataset; use a new `--dataset` name when needed. If using a custom test dataset name, also pass `--cases-file outputs/aid_eval/test-2000-rerun-01/cases.json` to the runner. To run all providers, use `--models luna jev clef d1 d1-ultra`. These commands make model API calls and publish experiments. Pilot results are integration checks, not held-out benchmark estimates.

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
| Controlled blur, contrast, and shifts | [Definitions-matched results](outputs/aid_eval/controlled-definitions-v1/results.md) |
| Added-blur detection and paired sharpness | [Results](outputs/aid_eval/visual-decisions-v1/results.md) |
| Web-image collection and human category review | [Review workflow](outputs/aid_eval/README.md#web-images-and-human-review) |
| Specialist baseline on 382 shared held-out images | [AiTLAS findings](outputs/aid_eval/specialist/aitlas/findings.txt) |
| Earlier multilabel disaster-image evaluation | [LADI guide](outputs/ladi_eval/README.md) |

The AID split is a fixed community partition: 6,000 train / 2,000 validation / 2,000 test. It is not the original paper's benchmark protocol. Native runs pin a Braintrust dataset version, check it against a local manifest, and hash-check image bytes before inference. Accuracy counts failed or invalid outputs as incorrect; valid-only metrics are labeled separately.

The current robustness study uses the same frozen class definitions as the newer main results, with fresh controls on the same 300 sources as all six transformations. [Earlier no-definitions results](outputs/aid_eval/controlled-300-v1/results.md) remain historical and are not its baseline.

Costs are recorded per prediction. Liquid uses valid provider-reported `usage.cost`, falling back to configured token rates when absent. Other providers use configured estimates. Rates in the code describe these experiments and should be checked before budgeting a new run. Replayed rows are historical predictions; summing all experiments does not measure unique API spend.

## Repository layout

- `outputs/aid_eval/`: AID scripts, tests, manifests, and named experiment exports. Scripts remain alongside results to preserve existing paths.
- `outputs/ladi_eval/`: separate LADI runner, scoring, publishing, and validation pilot.
- `work/aid/`: original dataset preparation and report scripts.

Image binaries, checkpoints, credentials, local environments, and image-directory symlinks are ignored by Git. A clone includes audit exports, not the source images. The native runner reads remote Braintrust attachments; transformation and historical runners also require the local images described in the AID guide. Do not edit frozen prompts, labels, or inputs and reuse an existing prediction cache.

## Public evaluation results

[BraintrustDataDev/aid-evaluation-results](https://huggingface.co/datasets/BraintrustDataDev/aid-evaluation-results) contains 30,500 predictions from the native baseline, class-definition follow-up, and definitions-matched robustness studies, with source references, hashes, prompts, and protocols. AID image redistribution permission could not be established; images and image derivatives are excluded. The published [rights review](https://huggingface.co/datasets/BraintrustDataDev/aid-evaluation-results/blob/main/RIGHTS.md) records the evidence. `outputs/aid_eval/export_hf_results.py` builds the allowlisted release with `--definitions-source` pointing to the completed class-definition study and `--out` to a staging directory; it requires `pyarrow`. `outputs/aid_eval/hf-publication.json` records the verified published revision.

## Development

```sh
pip install -r requirements-dev.txt
python -m unittest discover -s outputs/aid_eval -p 'test_*.py'
ruff check .
ruff format --check .
python scripts/release.py
```

Tests run locally with mocked provider calls and Braintrust logging disabled for the native Eval test. They cover payloads, MIME types, scoring, trial hooks, cost reconciliation, credential loading, human-label propagation, and paired statistical tests. They do not run paid evaluations or publish data.

## License and release

Original code and documentation are [MIT-licensed](LICENSE). Dataset images, provider outputs, models, and other third-party material have separate terms; see [third-party notices](THIRD_PARTY_NOTICES.md). Source images and paper PDFs are not distributed here.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and [RELEASING.md](RELEASING.md) for building a public archive without private research history.
