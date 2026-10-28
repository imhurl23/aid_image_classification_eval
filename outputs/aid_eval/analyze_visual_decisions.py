"""Analyze repeated decisions using the source image, not each variant, as the unit."""

import itertools
import json

import numpy as np
from analyze_robustness import sign_flip_p
from analyze_subgroups import holm
from scipy.stats import rankdata
from visual_decisions import MODELS, OUT, SPECS


def main():
    manifests = json.loads((OUT / "datasets.json").read_text())
    rng = np.random.default_rng(20261023)
    bootstrap = rng.integers(0, 300, size=(10000, 300))
    result = {
        "method": "300 independent source clusters; exact paired sign flips of per-source correct-count differences (two decisions/source); Holm across 20 planned model comparisons. 95% percentile bootstrap CIs resample source clusters, unadjusted. Both A/B orders included.",
        "tasks": {},
        "tests": [],
    }
    for task in SPECS:
        cases = manifests[task]["rows"]
        sources = sorted({c["metadata"]["original_id"] for c in cases})
        conditions = (
            ["original", "added_blur"]
            if task == "added-blur"
            else ["original_first", "original_second"]
        )
        truth = {c["id"]: c for c in cases}
        counts = {}
        summaries = {}
        for model in MODELS:
            folder = OUT / model
            rows = list(map(json.loads, (folder / (task + ".jsonl")).read_text().splitlines()))
            assert len(rows) == len({r["id"] for r in rows}) == 600
            for r in rows:
                assert r["input_sha256"] == truth[r["id"]]["metadata"]["sha256"]
                assert r["expected"] == truth[r["id"]]["expected"]
            bykey = {(r["original_id"], r["condition"]): r for r in rows}
            matrix = np.array(
                [
                    [
                        int(
                            bykey[s, c]["status"] == "ok"
                            and bykey[s, c].get("label") == bykey[s, c]["expected"]
                        )
                        for c in conditions
                    ]
                    for s in sources
                ]
            )
            counts[model] = matrix.sum(axis=1)
            summary = json.loads((folder / (task + "-summary.json")).read_text())
            summary["accuracy_ci95"] = np.quantile(
                counts[model][bootstrap].mean(axis=1) / 2, [0.025, 0.975]
            ).tolist()
            summary["by_condition"] = {
                c: float(matrix[:, i].mean()) for i, c in enumerate(conditions)
            }
            summary["both_correct"] = float((counts[model] == 2).mean())
            if task == "added-blur":
                summary["false_positive_rate"] = (
                    sum(
                        r.get("label") == "added_blur" for r in rows if r["condition"] == "original"
                    )
                    / 300
                )
                summary["brier"] = float(
                    np.mean(
                        [
                            (r["probabilities"]["added_blur"] - int(r["expected"] == "added_blur"))
                            ** 2
                            for r in rows
                            if r["status"] == "ok"
                        ]
                    )
                )
                valid = [r for r in rows if r["status"] == "ok"]
                y = np.array([r["expected"] == "added_blur" for r in valid])
                p = np.array([r["probabilities"]["added_blur"] for r in valid])
                n1 = y.sum()
                n0 = len(y) - n1
                summary["auroc"] = float((rankdata(p)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
            else:
                summary["choose_A_rate"] = sum(r.get("label") == "A" for r in rows) / 600
                summary["swap_consistency"] = (
                    sum(
                        bykey[s, conditions[0]]["status"] == "ok"
                        and bykey[s, conditions[1]]["status"] == "ok"
                        and bykey[s, conditions[0]].get("label")
                        != bykey[s, conditions[1]].get("label")
                        for s in sources
                    )
                    / 300
                )
            summaries[model] = summary
        for a, b in itertools.combinations(MODELS, 2):
            d = counts[a] - counts[b]
            result["tests"].append(
                {
                    "task": task,
                    "a": a,
                    "b": b,
                    "difference_pp": float(d.mean() * 50),
                    "ci95_pp": np.quantile(d[bootstrap].mean(axis=1) * 50, [0.025, 0.975]).tolist(),
                    "p": sign_flip_p(d),
                }
            )
        result["tasks"][task] = summaries
    holm(result["tests"])
    result["significant_tests"] = [t for t in result["tests"] if t["significant"]]
    result["total_cost"] = sum(
        r["cost"] for task in result["tasks"].values() for r in task.values()
    )
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# New visual decisions",
        "",
        result["method"],
        "",
        "| Model | Added-blur accuracy | AUROC | Paired sharpness accuracy | Pair swap consistency |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in MODELS:
        a = result["tasks"]["added-blur"][m]
        b = result["tasks"]["paired-sharpness"][m]
        lines.append(
            f"| {m} | {a['accuracy']:.1%} | {a['auroc']:.3f} | {b['accuracy']:.1%} | {b['swap_consistency']:.1%} |"
        )
    lines += ["", "## Significant model differences", ""]
    for t in result["significant_tests"]:
        lines.append(
            f"- {t['task']}: {t['a']} minus {t['b']} {t['difference_pp']:+.2f} pp; Holm p={t['p_holm']:.4g}."
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Added-blur labels encode the known treatment, not a human judgment of whether an image is blurry. Controls can have natural softness.",
        "- Paired sharpness treats the original as the sharper reference. Both orders are tested; swap consistency measures whether the model picks the same physical image after reversal (even if wrong).",
        "- Each task has 600 rows, but only 300 source images. Statistical tests and intervals preserve that clustering.",
        "- Percentile bootstrap intervals can collapse to 100–100% when no errors are observed (Clef paired sharpness); this is a bootstrap boundary limitation, not certainty about population accuracy.",
        "- Same exploratory source subset as the robustness experiment; no independent confirmation or generalization to other distortions.",
        "- API failures count as incorrect. AUROC and Brier use valid outputs only. No prompt tuning or model-specific prompt variants.",
        "- Pilot predictions are reused in full experiments, explicitly marked as replays. Cost counts each prediction once.",
        f"- Estimated API cost: ${result['total_cost']:.6f} (provider usage.cost for Liquid, configured token rates otherwise; excludes Braintrust storage).",
        "",
        "## Braintrust experiments",
        "",
    ]
    for task, summaries in result["tasks"].items():
        for m, r in summaries.items():
            lines.append(f"- [{task}: {m}]({r['url']})")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
