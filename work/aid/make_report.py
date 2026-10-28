import json
from pathlib import Path

p = Path("outputs/aid_eval/test-2000")
s = {a: json.loads((p / (a + "-summary.json")).read_text()) for a in ["luna", "jev"]}
c = json.loads((p / "comparison.json").read_text())
links = json.loads((p / "braintrust-links.json").read_text())
lines = [
    "AID full test results",
    "",
    "2,000 frozen test images; 30 classes; identical original 600x600 JPEG inputs.",
    "",
]
for a, title in [("luna", "Luna Decisions"), ("jev", "Jev 1.14")]:
    r = s[a]
    lo, hi = r["accuracy_95pct_wilson"]
    lines += [
        title,
        f"Accuracy: {r['accuracy_all_attempts']:.2%} (95% Wilson interval {lo:.2%}–{hi:.2%})",
        f"Valid outputs: {r['valid']}/{r['images']}",
        f"Macro F1 on valid outputs: {r['macro_f1_valid_only']:.4f}",
        f"Median / p95 request latency: {r['latency_ms_p50']:.0f} / {r['latency_ms_p95']:.0f} ms",
        f"Input tokens: {r['input_tokens']:,}; estimated API cost: ${r['estimated_api_cost_usd']:.4f}",
        links["experiments"][a],
        "",
    ]
lo, hi = c["paired_bootstrap_95pct_ci"]
lines += [
    f"Jev minus Luna accuracy: {c['jev_minus_luna_accuracy'] * 100:.2f} percentage points",
    f"Paired bootstrap 95% interval: {lo * 100:.2f} to {hi * 100:.2f} percentage points",
    f"Jev-only correct: {c['jev_only_correct']}; Luna-only correct: {c['luna_only_correct']}",
    f"Combined estimated API cost: ${sum(r['estimated_api_cost_usd'] for r in s.values()):.4f}",
    "",
    links["project"],
    links["dataset"],
    "",
    "Limits: community test split rather than official AID split. No specialist baseline yet; its training-ID verification remains pending. Geographic overlap, near-duplicates and foundation-model pretraining exposure have not been independently excluded. Costs use returned token usage and published standard API rates, excluding Braintrust charges.",
]
(p / "results.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
