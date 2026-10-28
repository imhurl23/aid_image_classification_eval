"""Read back published experiments and validate scores, origins and image references."""

import collections
import concurrent.futures
import hashlib
import json

import braintrust as bt
from visual_decisions import MODELS, OUT, PROJECT, SPECS, keys_load


def verify(item):
    model, task = item
    exp = bt.init(
        project=PROJECT,
        experiment=f"visual-decisions-v1-{task}__{model}",
        open=True,
        set_current=False,
    )
    rows = list(exp.fetch())
    roots = [r for r in rows if not r.get("span_parents")]
    assert len(roots) == 600, (model, task, len(roots))
    assert len({r["metadata"]["source_id"] for r in roots}) == 600
    assert all(r.get("origin") for r in roots)
    scores = {}
    for row in rows:
        scores.setdefault(row["root_span_id"], {}).update(row.get("scores") or {})
    image_count = 1 if task == "added-blur" else 2
    for r in roots:
        assert len(r["input"]["images"]) == image_count
        assert all(
            (a.reference if isinstance(a, bt.ReadonlyAttachment) else a)["content_type"]
            == "image/png"
            for a in r["input"]["images"]
        )
        assert scores[r["root_span_id"]]["valid_output"] == 1
        assert scores[r["root_span_id"]]["accuracy"] == int(r["output"]["label"] == r["expected"])
    summary = json.loads((OUT / model / (task + "-summary.json")).read_text())
    assert (
        abs(sum(scores[r["root_span_id"]]["accuracy"] for r in roots) / 600 - summary["accuracy"])
        < 1e-10
    )
    # Actually download both/control variants for one row per experiment; metadata hashes prove bytes.
    r = roots[0]
    for ref, sha in zip(r["input"]["images"], r["metadata"]["sha256"]):
        attachment = ref if isinstance(ref, bt.ReadonlyAttachment) else bt.ReadonlyAttachment(ref)
        assert hashlib.sha256(attachment.data).hexdigest() == sha
    return model + "/" + task, {
        "rows": 600,
        "images_per_row": image_count,
        "origin_verified": True,
        "sample_attachment_bytes_verified": True,
        "replayed": sum(r["metadata"].get("replayed_prediction", False) for r in roots),
        "conditions": dict(collections.Counter(r["metadata"]["condition"] for r in roots)),
    }


if __name__ == "__main__":
    bt.login(api_key=keys_load()["BRAINTRUST_API_KEY"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        result = dict(pool.map(verify, [(m, t) for m in MODELS for t in SPECS]))
    (OUT / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
