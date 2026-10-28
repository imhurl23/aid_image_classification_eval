"""Paired robustness effects vs fresh controls, with family-wise multiplicity correction."""

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


def main():
    root = BASE / "controlled-300-v1"
    cases = json.loads((root / "cases.json").read_text())
    labels = {c["original_id"]: c["label"] for c in cases}
    ids = sorted(labels)
    assert len(ids) == 300
    result = {
        "method": "Paired 300-image analysis; exact McNemar within model; exact paired model-label-swap tests on differences in accuracy changes between models. Holm correction across all 90 planned tests.",
        "note": "Negative change means accuracy loss vs fresh PNG control; equal-class balanced subset. Single response per condition.",
        "models": {},
        "tests": [],
    }
    correctness = {}
    changes = {}
    for m in MODELS:
        folder = BASE / f"controlled-300-v1-{m}"
        rows = [json.loads(x) for x in (folder / f"{m}.jsonl").read_text().splitlines()]
        bykey = {(r["original_id"], r["condition"]): r for r in rows}
        assert len(rows) == len(bykey) == 2100
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
            "conditions": {},
        }
        for c in CONDITIONS:
            values = correctness[m][c]
            delta = values - correctness[m]["control"]
            result["models"][m]["conditions"][c] = {
                "accuracy": float(values.mean()),
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
    lines += ["", "## Significant comparisons", ""]
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
