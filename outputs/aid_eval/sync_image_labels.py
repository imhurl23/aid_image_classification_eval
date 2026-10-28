"""Propagate authoritative human image labels to corresponding model logs. No model calls."""

import hashlib
import json
import uuid

from native_eval import LABELS
from publish_model_review_rows import P
from web_aid_eval import OUT, REVIEW_FIELD, api, keys_load, review_rows

ALLOWED = set(LABELS + ["Ambiguous", "Other", "NotAerial", "Unusable"])


def label_of(row):
    value = row.get("expected")
    return value.get(REVIEW_FIELD) if isinstance(value, dict) else value


def model_row_id(case, model):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "web-aid-v1/model-review/" + case + "/" + model))


def updates_for(master, predictions, label):
    if label is not None and label not in ALLOWED:
        raise ValueError("Unknown human label: " + str(label))
    events = []
    for r in predictions:
        assert r["id"] == master["metadata"]["id"]
        metadata = {
            "human_label_source_id": master["id"],
            "human_label_value": label,
            "human_label_sync_version": 1,
            "human_label_eligible": label in LABELS,
        }
        score = (
            int(r.get("status") == "ok" and r.get("label") == label) if label in LABELS else None
        )
        events.append(
            {
                "id": model_row_id(r["id"], r["model"]),
                "_is_merge": True,
                "expected": {REVIEW_FIELD: label},
                "metadata": metadata,
                "scores": {"human_category_accuracy": score},
            }
        )
    if predictions:
        events.append(
            {
                "id": predictions[0]["inference_log_id"],
                "_is_merge": True,
                "expected": {REVIEW_FIELD: label},
                "metadata": {"human_label_source_id": master["id"], "human_label_value": label},
            }
        )
    return events


def main():
    keys = keys_load()
    masters = review_rows(keys, P)
    by = {}
    for r in map(json.loads, (OUT / "predictions.jsonl").read_text().splitlines()):
        by.setdefault(r["id"], []).append(r)
    statefile = OUT / "label-sync-state.json"
    state = json.loads(statefile.read_text()) if statefile.exists() else {}
    events = []
    pending = {}
    for r in masters:
        case = r["metadata"]["id"]
        label = label_of(r)
        if label is None and case not in state:
            continue
        if case in state and state[case] == label:
            continue
        predictions = by.get(case, [])
        assert len(predictions) == 5
        events.extend(updates_for(r, predictions, label))
        pending[case] = label
    if events:
        for i in range(0, len(events), 200):
            api(keys, "POST", f"/v1/project_logs/{P}/insert", json={"events": events[i : i + 200]})
        # Persist only after all writes succeed; deterministic merge updates are retry-safe.
        state.update(pending)
        tmp = statefile.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(statefile)
    report = {
        "changed_images": len(pending),
        "updated_model_rows": len(pending) * 5,
        "updated_inference_roots": len(pending),
        "labeled_images": sum(v is not None for v in state.values()),
        "new_model_calls": 0,
    }
    (OUT / "label-sync-status.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
