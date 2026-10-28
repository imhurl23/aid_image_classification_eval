"""Compare fresh single-trial runs using paired image-level bootstraps."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="native-fresh-01")
    args = parser.parse_args()
    if "/" in args.prefix or ".." in args.prefix:
        parser.error("Prefix must be a directory name prefix")
    base = Path(__file__).resolve().parent
    cases = json.loads((base / "test-2000-rerun-01/cases.json").read_text())
    models = ["luna", "jev", "clef", "d1", "d1-ultra"]
    correct = {}
    report = {
        "models": {},
        "paired_comparisons": {},
        "bootstrap_replicates": 10000,
        "seed": 20261021,
    }
    first_manifest = None
    for arm in models:
        root = base / (args.prefix + "-" + arm)
        manifest = json.loads((root / "native-manifest.json").read_text())
        protocol = {
            k: manifest[k]
            for k in (
                "dataset",
                "dataset_version",
                "instructions",
                "labels",
                "trials",
                "retry_policy",
                "image_policy",
            )
        }
        if first_manifest is None:
            first_manifest = protocol
        assert protocol == first_manifest and protocol["trials"] == 1
        records = [json.loads(x) for x in (root / (arm + ".jsonl")).read_text().splitlines()]
        byid = {r["id"]: r for r in records}
        assert len(records) == len(byid) == len(cases)
        for c in cases:
            assert byid[c["id"]]["input_sha256"] == c["sha256"]
        correct[arm] = np.array(
            [
                int(byid[c["id"]]["status"] == "ok" and byid[c["id"]]["label"] == c["label"])
                for c in cases
            ]
        )
        summary = json.loads((root / (arm + "-summary.json")).read_text())
        s = summary["trials"][0]
        report["models"][arm] = {
            k: s[k] for k in ("accuracy_all_attempts", "valid", "estimated_api_cost_usd")
        }
        report["models"][arm]["experiment_url"] = summary["experiment_url"]
    for a, b in itertools.combinations(models, 2):
        diff = correct[a] - correct[b]
        rng = np.random.default_rng(report["seed"])
        boot = [
            diff[rng.integers(0, len(cases), len(cases))].mean()
            for _ in range(report["bootstrap_replicates"])
        ]
        report["paired_comparisons"][a + "_minus_" + b] = {
            "accuracy_difference": float(diff.mean()),
            "paired_bootstrap_95pct_ci": np.percentile(boot, [2.5, 97.5]).tolist(),
            "note": "Pairwise intervals, not adjusted for multiple comparisons",
        }
    report["protocol"] = first_manifest
    (base / (args.prefix + "-comparison.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
