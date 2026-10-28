"""Exploratory paired subgroup accuracy tests for the corrected five-model run."""

import itertools
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parent
MODELS = ["luna", "jev", "clef", "d1", "d1-ultra"]


def exact_mcnemar(wins, losses):
    n = wins + losses
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n)


def holm(tests):
    ordered = sorted(tests, key=lambda r: r["p"])
    prior = 0.0
    for i, row in enumerate(ordered):
        prior = max(prior, min(1.0, row["p"] * (len(ordered) - i)))
        row["p_holm"] = prior
        row["significant"] = prior < 0.05


def main():
    cases = json.loads((BASE / "test-2000-rerun-01/cases.json").read_text())
    models = {}
    manifests = []
    for model in MODELS:
        root = BASE / ("native-fresh-01-" + model)
        records = [
            json.loads(line) for line in (root / (model + ".jsonl")).read_text().splitlines()
        ]
        models[model] = {r["id"]: r for r in records}
        assert len(records) == len(models[model]) == len(cases)
        manifests.append(json.loads((root / "native-manifest.json").read_text()))
    for field in ("dataset_version", "instructions", "labels", "retry_policy", "image_policy"):
        assert all(m[field] == manifests[0][field] for m in manifests)
    for c in cases:
        for model in MODELS:
            assert models[model][c["id"]]["input_sha256"] == c["sha256"]
        with Image.open(BASE / "test-2000-rerun-01/images" / c["image"]) as im:
            c["encoding"] = im.format
    slices = [
        ("class", label, [c for c in cases if c["label"] == label])
        for label in sorted({c["label"] for c in cases})
    ]
    slices += [
        ("encoding", fmt, [c for c in cases if c["encoding"] == fmt]) for fmt in ("JPEG", "PNG")
    ]
    tests, accuracies, confusions = [], [], {}
    for kind, name, group in slices:
        correct = {
            m: [
                int(
                    models[m][c["id"]]["status"] == "ok"
                    and models[m][c["id"]]["label"] == c["label"]
                )
                for c in group
            ]
            for m in MODELS
        }
        accuracies.append(
            {
                "kind": kind,
                "group": name,
                "n": len(group),
                "accuracy": {m: sum(v) / len(group) for m, v in correct.items()},
            }
        )
        for a, b in itertools.combinations(MODELS, 2):
            wins = sum(x > y for x, y in zip(correct[a], correct[b]))
            losses = sum(x < y for x, y in zip(correct[a], correct[b]))
            tests.append(
                {
                    "kind": kind,
                    "group": name,
                    "n": len(group),
                    "a": a,
                    "b": b,
                    "a_accuracy": sum(correct[a]) / len(group),
                    "b_accuracy": sum(correct[b]) / len(group),
                    "difference_pp": 100 * (wins - losses) / len(group),
                    "a_only_correct": wins,
                    "b_only_correct": losses,
                    "p": exact_mcnemar(wins, losses),
                }
            )
    holm(tests)  # One family: 30 classes + 2 encodings, 10 pairs each = 320 tests.
    for m in MODELS:
        counts = Counter(
            (c["label"], models[m][c["id"]].get("label", "ERROR"))
            for c in cases
            if models[m][c["id"]].get("label") != c["label"]
        )
        confusions[m] = [
            {"expected": a, "predicted": b, "n": n} for (a, b), n in counts.most_common(10)
        ]
    split_counts = {
        k: len(v) for k, v in json.loads((BASE / "split-manifest.json").read_text()).items()
    }
    result = {
        "scope": "Fresh corrected native-fresh-01 runs only; 2,000 paired test images",
        "method": "Two-sided exact McNemar tests; Holm family-wise correction across all 320 subgroup/model-pair tests; alpha=0.05",
        "caveats": [
            "Exploratory subgroup analysis, not prespecified confirmatory tests.",
            "No full matched five-model runs for train or validation.",
            "Encoding is a derived technical subgroup, not a semantic dataset tag; class mix may confound encoding comparisons.",
            "A significant within-group comparison does not establish a model-by-group interaction.",
            "No correction can address geographic correlation without location metadata.",
        ],
        "available_splits": split_counts,
        "accuracy_by_group": accuracies,
        "tests": tests,
        "significant_tests": [t for t in tests if t["significant"]],
        "top_confusions": confusions,
        "encoding_class_counts": {
            fmt: dict(Counter(c["label"] for c in cases if c["encoding"] == fmt))
            for fmt in ("JPEG", "PNG")
        },
    }
    (BASE / "native-fresh-01-subgroups.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# AID subgroup analysis",
        "",
        result["scope"],
        "",
        result["method"],
        "",
        "| Class / encoding | N | Luna | Jev | Clef | D1 | D1 Ultra |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in accuracies:
        lines.append(
            "| "
            + r["group"]
            + " | "
            + str(r["n"])
            + " | "
            + " | ".join(f"{r['accuracy'][m] * 100:.2f}%" for m in MODELS)
            + " |"
        )
    lines += [
        "",
        "## Significant pairwise differences after Holm correction",
        "",
        "| Group | Model A | Model B | A minus B (points) | Discordant A/B wins | Adjusted p |",
        "|---|---|---|---:|---:|---:|",
    ]
    for r in sorted(result["significant_tests"], key=lambda r: r["p_holm"]):
        lines.append(
            f"| {r['group']} | {r['a']} | {r['b']} | {r['difference_pp']:+.2f} | {r['a_only_correct']}/{r['b_only_correct']} | {r['p_holm']:.4g} |"
        )
    lines += ["", "## Limits", ""] + ["- " + c for c in result["caveats"]]
    (BASE / "native-fresh-01-subgroups.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "number_of_tests": len(tests),
                "significant_tests": result["significant_tests"],
                "encoding_accuracy": [r for r in accuracies if r["kind"] == "encoding"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
