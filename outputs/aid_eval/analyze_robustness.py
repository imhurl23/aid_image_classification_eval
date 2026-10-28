"""Paired robustness effects vs fresh controls, with family-wise multiplicity correction."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from analyze_subgroups import exact_mcnemar, holm
from scipy.stats import binom

BASE = Path(__file__).resolve().parent
MODELS = ["luna", "jev", "clef", "d1", "d1-ultra"]
CONDITIONS = [
    "control",
    "blur_1",
    "blur_2",
    "contrast_0.5",
    "contrast_1.5",
    "shift_left_60",
    "shift_right_60",
]


def sign_flip_p(d):
    # Exact model-label swaps per image; differences of accuracy changes are -2..2.
    d = np.asarray(d, dtype=int)
    target = abs(d.sum())
    n1 = int((abs(d) == 1).sum())
    n2 = int((abs(d) == 2).sum())
    a = np.arange(n1 + 1)
    b = np.arange(n2 + 1)
    sums = (2 * a - n1)[:, None] + (4 * b - 2 * n2)[None, :]
    probability = binom.pmf(a, n1, 0.5)[:, None] * binom.pmf(b, n2, 0.5)[None, :]
    return min(1.0, float(probability[abs(sums) >= target].sum()))


def validate_manifest(manifest, expected):
    for field in (
        "instructions",
        "labels",
        "dataset",
        "dataset_version",
        "trials",
        "transform_protocol",
    ):
        if manifest.get(field) != expected.get(field):
            raise ValueError("Cannot combine runs with different " + field)
    if manifest.get("trials") != 1:
        raise ValueError("This analysis requires exactly one trial per condition")
    prompt_hash = manifest.get("transform_protocol", {}).get("prompt_sha256")
    if prompt_hash and hashlib.sha256(manifest["instructions"].encode()).hexdigest() != prompt_hash:
        raise ValueError("Run prompt differs from frozen definitions protocol")


def bootstrap_indices(ids, labels, reps=5000):
    rng = np.random.default_rng(20261028)
    groups = [
        np.array([i for i, source in enumerate(ids) if labels[source] == label])
        for label in sorted(set(labels.values()))
    ]
    return np.concatenate(
        [rng.choice(group, (reps, len(group)), replace=True) for group in groups], axis=1
    )


def ece(confidence, correct, valid, indices=None):
    if indices is not None:
        confidence, correct, valid = confidence[indices], correct[indices], valid[indices]
    else:
        confidence, correct, valid = confidence[None, :], correct[None, :], valid[None, :]
    count = valid.sum(axis=1)
    total = np.zeros(len(count))
    bins = np.minimum((confidence * 10).astype(int), 9)
    for bucket in range(10):
        mask = valid & (bins == bucket)
        total += np.abs(((confidence - correct) * mask).sum(axis=1))
    return np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)


def interval(values):
    return [float(x) for x in np.nanquantile(values, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", default="controlled-300-v1")
    args = parser.parse_args()
    root = BASE / args.run_prefix
    cases = json.loads((root / "cases.json").read_text())
    labels = {c["original_id"]: c["label"] for c in cases}
    ids = sorted(labels)
    assert len(ids) == 300
    resamples = bootstrap_indices(ids, labels)
    expected_manifest = None
    result = {
        "method": "Paired 300-image analysis; exact McNemar within model; exact paired model-label-swap tests on differences in accuracy changes between models. Holm correction across all 90 planned tests.",
        "note": "Negative change means accuracy loss vs fresh PNG control; equal-class balanced subset. Single response per condition.",
        "run_prefix": args.run_prefix,
        "interval_method": "Pointwise 95% paired source bootstrap, stratified by reference class; 5,000 replicates, seed 20261028. Intervals are not multiplicity-adjusted.",
        "models": {},
        "tests": [],
    }
    correctness = {}
    changes = {}
    for m in MODELS:
        folder = BASE / f"{args.run_prefix}-{m}"
        manifest = json.loads((folder / "native-manifest.json").read_text())
        expected_manifest = manifest if expected_manifest is None else expected_manifest
        validate_manifest(manifest, expected_manifest)
        if args.run_prefix == "controlled-definitions-v1":
            expected_protocol = json.loads((root / "protocol.json").read_text())
            if manifest["transform_protocol"] != expected_protocol:
                raise ValueError("Run differs from the requested robustness protocol")
        result["prompt_sha256"] = hashlib.sha256(manifest["instructions"].encode()).hexdigest()
        result["prompt_condition"] = manifest.get("transform_protocol", {}).get(
            "prompt_condition", "original prompt without definitions"
        )
        result["dataset_version"] = manifest["dataset_version"]
        rows = [json.loads(x) for x in (folder / f"{m}.jsonl").read_text().splitlines()]
        bykey = {(r["original_id"], r["condition"]): r for r in rows}
        assert len(rows) == len(bykey) == 2100
        if {r["id"] for r in rows} != {c["id"] for c in cases} or any(
            r["trial_index"] != 0 for r in rows
        ):
            raise ValueError("Prediction IDs or trial indices differ from the frozen cases")
        for c in cases:
            assert bykey[c["original_id"], c["condition"]]["input_sha256"] == c["sha256"]
        correctness[m] = {
            c: np.array(
                [
                    int(bykey[i, c]["status"] == "ok" and bykey[i, c]["label"] == labels[i])
                    for i in ids
                ]
            )
            for c in CONDITIONS
        }
        changes[m] = {c: correctness[m][c] - correctness[m]["control"] for c in CONDITIONS[1:]}
        result["models"][m] = {
            "experiment_url": json.loads((folder / f"{m}-summary.json").read_text())[
                "experiment_url"
            ],
            "cost": sum(r["estimated_api_cost_usd"] for r in rows),
            "predictions": len(rows),
            "valid": sum(r["status"] == "ok" for r in rows),
            "prediction_export_sha256": hashlib.sha256(
                (folder / f"{m}.jsonl").read_bytes()
            ).hexdigest(),
            "conditions": {},
        }
        calibration = {}
        common_valid = np.array(
            [all(bykey[i, c]["status"] == "ok" for c in CONDITIONS) for i in ids]
        )
        result["models"][m]["calibration_common_valid_sources"] = int(common_valid.sum())
        for c in CONDITIONS:
            valid = common_valid
            confidence = np.array(
                [
                    bykey[i, c].get("probabilities", {}).get(bykey[i, c].get("label"), 0.0)
                    for i in ids
                ]
            )
            calibration[c] = (
                ece(confidence, correctness[m][c], valid)[0],
                ece(confidence, correctness[m][c], valid, resamples),
            )
        for c in CONDITIONS:
            values = correctness[m][c]
            delta = values - correctness[m]["control"]
            result["models"][m]["conditions"][c] = {
                "accuracy": float(values.mean()),
                "accuracy_bootstrap_95": interval(values[resamples].mean(axis=1)),
                "change_pp_bootstrap_95": interval(100 * delta[resamples].mean(axis=1)),
                "ece_10": float(calibration[c][0]),
                "ece_change_pp": float(100 * (calibration[c][0] - calibration["control"][0])),
                "ece_change_pp_bootstrap_95": interval(
                    100 * (calibration[c][1] - calibration["control"][1])
                ),
                "change_pp": float(100 * delta.mean()),
                "valid": sum(bykey[i, c]["status"] == "ok" for i in ids),
                "prediction_flips": sum(
                    bykey[i, c].get("label") != bykey[i, "control"].get("label") for i in ids
                ),
                "lost_correct": int((delta == -1).sum()),
                "gained_correct": int((delta == 1).sum()),
            }
            if c != "control":
                result["tests"].append(
                    {
                        "kind": "within_model",
                        "model": m,
                        "condition": c,
                        "change_pp": float(100 * delta.mean()),
                        "p": exact_mcnemar(int((delta == 1).sum()), int((delta == -1).sum())),
                    }
                )
    for c in CONDITIONS[1:]:
        for a, b in itertools.combinations(MODELS, 2):
            d = changes[a][c] - changes[b][c]
            result["tests"].append(
                {
                    "kind": "between_models",
                    "a": a,
                    "b": b,
                    "condition": c,
                    "difference_in_change_pp": float(d.mean() * 100),
                    "p": sign_flip_p(d),
                }
            )
    assert len(result["tests"]) == 90
    holm(result["tests"])
    result["significant_tests"] = [x for x in result["tests"] if x["significant"]]
    result["total_cost"] = sum(v["cost"] for v in result["models"].values())
    (root / "analysis.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Controlled image robustness results",
        "",
        result["method"],
        "",
        result["note"],
        "",
        "Prompt: **"
        + result["prompt_condition"]
        + "** for both fresh controls and transformed images.",
        "",
        result["interval_method"],
        "",
        "| Condition | Luna | Jev | Clef | D1 | D1 Ultra |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in CONDITIONS:
        cells = []
        for m in MODELS:
            r = result["models"][m]["conditions"][c]
            cells.append(
                f"{r['accuracy'] * 100:.1f}% ({r['change_pp']:+.1f} pp)"
                if c != "control"
                else f"{r['accuracy'] * 100:.1f}%"
            )
        lines.append("| " + c + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Run audit",
        "",
        "| Model / experiment | Valid predictions | Recorded or estimated cost |",
        "|---|---:|---:|",
    ]
    for model in MODELS:
        audit = result["models"][model]
        lines.append(
            f"| [{model}]({audit['experiment_url']}) | {audit['valid']:,}/{audit['predictions']:,} | ${audit['cost']:.4f} |"
        )
    lines += [
        "",
        f"Total recorded or estimated model cost: ${result['total_cost']:.4f}. Braintrust charges are excluded.",
    ]
    if args.run_prefix == "controlled-definitions-v1":
        lines += [
            "",
            "[Accuracy chart](accuracy-deltas.png) · [Calibration chart](calibration-deltas.png) · [Remote verification](verification.json)",
            "",
            "One Luna refusal is retained as incorrect. No historical predictions were replayed.",
        ]
    lines += ["", "## Significant comparisons", ""]
    if not result["significant_tests"]:
        lines.append("No comparisons passed the Holm-adjusted 0.05 threshold.")
    for t in result["significant_tests"]:
        if t["kind"] == "within_model":
            lines.append(
                f"- {t['model']}, {t['condition']}: {t['change_pp']:+.2f} pp vs control; Holm p={t['p_holm']:.4g}."
            )
        else:
            lines.append(
                f"- {t['condition']}: {t['a']} minus {t['b']} change = {t['difference_in_change_pp']:+.2f} pp; Holm p={t['p_holm']:.4g}."
            )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- All controls and variants are lossless PNG with preserved decoded control pixels; not the earlier original-file-encoding experiment.",
        "- The balanced 300-image sample is not the full test set; rates should not be compared directly to the earlier full-test leaderboard.",
        "- Shifts crop one boundary and reflect another. Effects include cropping/fill artifacts, not solely object position.",
        "- A single call per condition cannot separate stochastic output variation from a small transformation effect.",
        "- Accuracy includes every attempt, counting refusals and errors as wrong. Calibration uses each model's common valid-source intersection across all seven conditions, with 10 equal-width confidence bins.",
        "- No detected significance is not evidence of equivalence; this design is less sensitive to small effects.",
        "- These controlled findings use the same source test pool that motivated the earlier observational analysis; confirm on independent images.",
    ]
    (root / "results.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "significant_tests": result["significant_tests"],
                "total_cost": result["total_cost"],
                "accuracies": {
                    m: {c: result["models"][m]["conditions"][c]["accuracy"] for c in CONDITIONS}
                    for m in MODELS
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
