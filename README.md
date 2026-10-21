# AID image classification eval

Compare OpenAI Decisions Luna and TypeSafe Jev on AID single-label aerial scenes, with Braintrust image review and specialist-baseline analysis.

- [AID runner and protocol](outputs/aid_eval/README.txt)
- [Pilot runner](outputs/aid_eval/eval.py)
- [Full test runner](outputs/aid_eval/full_test.py)
- [Full test results](outputs/aid_eval/test-2000/results.txt)
- [Specialist analysis](outputs/aid_eval/specialist/)
- [Earlier LADI work](outputs/ladi_eval/README.txt)
- [Dataset preparation](work/aid/prepare.py)

Run scripts from this repository root. Credentials are external and are not committed; the pilot and full test runners accept --env-file. Specialist publication scripts still reference the sibling classifier_eval credentials file.

All original images, checkpoints, downloaded source repositories, and local environments have been moved here but are ignored by Git. Existing links at the original Codex workspace preserve old paths and virtual-environment entry points. The GitHub repository is private.

## Run a fresh full evaluation

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python outputs/aid_eval/full_test.py --run-name test-2000-new-run --env-file /path/to/credentials.env
```

The credentials file needs OPENAI_API_KEY, TYPESAFE_API_KEY, and BRAINTRUST_API_KEY.
Use a new run name for fresh model calls; repeating a name resumes cached predictions.
The runner downloads the pinned 2,000 test images and publishes them to Braintrust.
Existing test-2000 results are already complete. The command above starts a new run.
