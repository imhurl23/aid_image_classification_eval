"""Score aligned predictions. Usage: python score.py truth.jsonl predictions.jsonl

Truth: {"id":"test:0","labels":{"bridges_any":false,...}}
Prediction: {"id":"test:0","status":"ok","probabilities":{...},"latency_ms":42}
Failures must have status != ok and omit probabilities. One file per model.
Requires numpy. Does not call providers or infer missing outputs.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())
LABELS = CONFIG["labels"]


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate row IDs")
    return indexed


def score(truth, predictions):
    if not truth or set(truth) != set(predictions):
        raise ValueError("Require nonempty, identical ID sets; record every failed request")
    valid = []
    latencies = []
    for key, row in truth.items():
        labels = row["labels"]
        if set(labels) != set(LABELS) or any(type(v) is not bool for v in labels.values()):
            raise ValueError(f"Invalid ground truth: {key}")
        prediction = predictions[key]
        status = prediction["status"]
        if not isinstance(status, str) or not status:
            raise ValueError(f"Invalid status: {key}")
        latency = prediction.get("latency_ms")
        if latency is not None:
            if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
                raise ValueError(f"Invalid latency: {key}")
            latencies.append(latency)
        if status != "ok":
            continue
        probs = prediction["probabilities"]
        if set(probs) != set(LABELS) or any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probs.values()
        ):
            raise ValueError(f"Invalid probabilities: {key}")
        valid.append(key)
    report = {
        "total_images": len(truth),
        "valid_images": len(valid),
        "coverage": len(valid) / len(truth),
        "quality_denominator": "complete valid images only; compare coverage alongside quality",
        "latency_observations": len(latencies),
    }
    if latencies:
        report.update({f"latency_ms_p{p}": float(np.percentile(latencies, p)) for p in (50, 95)})
    if not valid:
        return report
    y = np.array([[truth[k]["labels"][label] for label in LABELS] for k in valid])
    p = np.array([[predictions[k]["probabilities"][label] for label in LABELS] for k in valid])
    pred = p >= CONFIG["threshold"]
    tp = (y & pred).sum(axis=0)
    fp = (~y & pred).sum(axis=0)
    fn = (y & ~pred).sum(axis=0)
    support = y.sum(axis=0)

    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    f1 = divide(2 * tp, 2 * tp + fp + fn)
    report.update(
        macro_f1=float(f1.mean()),
        micro_f1=float(divide(2 * tp.sum(), (2 * tp + fp + fn).sum())),
        brier_score=float(np.mean((p - y.astype(float)) ** 2)),
        exact_match=float(np.all(y == pred, axis=1).mean()),
    )
    report["per_label"] = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "positive_support": int(support[i]),
        }
        for i, label in enumerate(LABELS)
    }
    report["zero_division_policy"] = (
        "precision, recall, F1 set to zero when undefined; inspect support"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("truth")
    parser.add_argument("predictions")
    args = parser.parse_args()
    print(
        json.dumps(
            score(read_rows(args.truth), read_rows(args.predictions)), indent=2, allow_nan=False
        )
    )
