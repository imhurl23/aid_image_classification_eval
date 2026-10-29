# Contributing

Use Python 3.12 or later and install `requirements-dev.txt` in a virtual environment.
Run these checks from the repository root:

```sh
python -m unittest discover -s outputs/aid_eval -p 'test_*.py'
ruff check .
ruff format --check .
python scripts/release.py
```

Tests use mocked providers and local Braintrust evaluation with logging disabled.
Do not run model evaluations or remote maintenance scripts as part of the test suite.

Keep frozen prompts, splits, labels, manifests, and historical prediction exports
unchanged when modifying code. Use a new run name for a fresh experiment. Explain
any intentional correction to an export, including its effect on reported metrics.

Keep credentials, downloaded images, personal galleries, review-session snapshots,
model weights, and local paths out of commits. Add regressions for changes to
scoring, input integrity, cache behavior, and publishing destinations. Include the
checks you ran in a pull request.

The [release guide](RELEASING.md) explains how to create a public snapshot without
including private research history.
