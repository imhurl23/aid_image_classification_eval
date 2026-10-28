from native_review import review_image

"""Verify image coverage and logged predictions after collection and shared padding repair."""
import collections
import hashlib
import json

import braintrust as bt
import numpy as np
import web_aid_eval as w
from PIL import Image


def main():
    keys = w.keys_load()
    bt.login(api_key=keys["BRAINTRUST_API_KEY"])
    info = json.loads((w.OUT / "braintrust.json").read_text())
    cases = json.loads((w.OUT / "cases.json").read_text())
    byid = {c["id"]: c for c in cases}
    assert len(byid) == 1000
    for c in cases:
        assert min(c["dimensions"]) >= 280 and max(c["dimensions"]) <= 600
        assert c["source_page"] and c["license"] and c["source_url"]
        assert (
            hashlib.sha256((w.OUT / "images" / c["image"]).read_bytes()).hexdigest() == c["sha256"]
        )
        if c.get("pre_padding_image"):
            original = Image.open(w.OUT / "images" / c["pre_padding_image"]).convert("RGB")
            padded = Image.open(w.OUT / "images" / c["image"]).convert("RGB")
            x, y = c["padding"]["offset"]
            assert np.array_equal(
                np.asarray(original),
                np.asarray(padded.crop((x, y, x + original.width, y + original.height))),
            )
    rows = w.review_rows(keys, info["project_id"])
    assert len(rows) == 1000
    assert {r["metadata"]["id"] for r in rows} == set(byid)
    for r in rows:
        assert review_image(r["input"])["content_type"] == "image/png"
        assert r["metadata"]["sha256"] == byid[r["metadata"]["id"]]["sha256"]
    sample = sorted(rows, key=lambda r: r["id"])[::100][:9] + [
        next(r for r in rows if r["metadata"].get("padding"))
    ]
    for r in sample:
        assert (
            hashlib.sha256(bt.ReadonlyAttachment(review_image(r["input"])).data).hexdigest()
            == r["metadata"]["sha256"]
        )
    ds = bt.init_dataset(project=w.PROJECT, name=w.BATCH)
    dataset = list(ds.fetch())
    assert len(dataset) == 1000
    assert all(r["metadata"]["sha256"] == byid[r["id"]]["sha256"] for r in dataset)
    preds = list(map(json.loads, (w.OUT / "predictions.jsonl").read_text().splitlines()))
    assert len(preds) == 5000
    lookup = {(r["id"], r["model"]): r for r in preds}
    assert len(lookup) == 5000
    for r in preds:
        assert r["input_sha256"] == byid[r["id"]]["sha256"]
    # Verify stored root predictions match the local audit, without transferring the full raw provider response.
    query = f"select id, metadata.source_id as source_id, output from project_logs('{info['project_id']}') where metadata.batch = 'web-aid-v1' and metadata.record_type = 'inference' limit 1000"
    inferred = w.api(keys, "POST", "/btql", json={"query": query})["data"]
    assert len(inferred) == 1000
    for r in inferred:
        assert set(r["output"]) == set(w.native.MODELS)
        for m, o in r["output"].items():
            saved = lookup[r["source_id"], m]
            assert o["status"] == saved["status"] and o.get("label") == saved.get("label")
            assert o["input_sha256"] == byid[r["source_id"]]["sha256"]
    result = {
        "review_logs": len(rows),
        "inference_logs": len(inferred),
        "model_predictions": 5000,
        "dataset_rows": 1000,
        "sample_attachment_hash_checks": 10,
        "valid_by_model": {
            m: sum(r["model"] == m and r["status"] == "ok" for r in preds) for m in w.native.MODELS
        },
        "final_estimated_api_cost_usd": sum(r["estimated_api_cost_usd"] for r in preds),
        "superseded_padding_attempt_cost_usd": sum(
            r.get("estimated_api_cost_usd", 0)
            for r in map(
                json.loads, (w.OUT / "pre-padding-predictions.jsonl").read_text().splitlines()
            )
        ),
    }
    result["total_estimated_api_cost_usd"] = (
        result["final_estimated_api_cost_usd"] + result["superseded_padding_attempt_cost_usd"]
    )
    (w.OUT / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
