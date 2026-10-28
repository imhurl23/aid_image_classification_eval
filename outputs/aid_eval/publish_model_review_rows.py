"""One native Braintrust root row per cached image/model decision; no inference calls."""

import json
import time
import uuid
from urllib.parse import quote

import native_eval as native
from native_review import NAMES, review_image
from web_aid_eval import OUT, PROJECT, REVIEW_FIELD, api, keys_load, review_rows

P = "aa1a18b9-6a22-48a4-b10d-b7b1f79c473e"


def model_review_rows(keys):
    rows = []
    for model in native.MODELS:
        q = f"select id,input,output,expected,metadata from project_logs('{P}') where metadata.batch = 'web-aid-v1' and metadata.record_type = 'model_review' and metadata.model = '{model}' limit 1000"
        for attempt in range(5):
            try:
                rows.extend(api(keys, "POST", "/btql", json={"query": q})["data"])
                break
            except RuntimeError as exc:
                if "429" not in str(exc) or attempt == 4:
                    raise
                time.sleep(35)
        time.sleep(3.1)
    return rows


def publish(keys=None):
    keys = keys if keys is not None else keys_load()
    masters = review_rows(keys, P)
    bycase = {r["metadata"]["id"]: r for r in masters}
    saved = list(map(json.loads, (OUT / "predictions.jsonl").read_text().splitlines()))
    existing = {r["id"]: r for r in model_review_rows(keys)}
    events = []
    wanted = {}
    for prediction in saved:
        c = prediction["id"]
        m = prediction["model"]
        master = bycase[c]
        rid = str(uuid.uuid5(uuid.NAMESPACE_URL, "web-aid-v1/model-review/" + c + "/" + m))
        content = (
            prediction["label"]
            if prediction["status"] == "ok"
            else "[" + prediction["status"] + ": no category returned]"
        )
        inp = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": native.protocol.INSTRUCTIONS},
                    {"type": "image_url", "image_url": {"url": review_image(master["input"])}},
                ],
            }
        ]
        output = [{"role": "assistant", "content": content}]
        e = {
            "id": rid,
            "_is_merge": True,
            "_merge_paths": [["input"], ["output"]],
            "input": inp,
            "output": output,
            "span_attributes": {"name": NAMES[m] + " — scene classification", "type": "task"},
            "metadata": {
                "batch": "web-aid-v1",
                "record_type": "model_review",
                "id": c,
                "model": m,
                "served_model": prediction.get("served_model"),
                "sha256": prediction["input_sha256"],
                "source_review_log_id": master["id"],
                "source_inference_log_id": prediction["inference_log_id"],
                "prediction": prediction,
                "review_presentation_version": 5,
                "decision_schema": {
                    "name": "scene",
                    "type": "choice",
                    **(
                        {
                            "choices": [
                                {"value": label, "description": label} for label in native.LABELS
                            ]
                        }
                        if m == "luna"
                        else {"criteria": {label: label for label in native.LABELS}}
                    ),
                },
                "input_display_note": "Review rendering of the image and question instructions; provider request places instructions and allowed choices in the structured question, not a user text message.",
                "cached_prediction": True,
                "review_blinding": "Model prediction visible.",
            },
            "tags": ["web-aid-v1", "web-model-review", m],
            "metrics": {
                "cached": 1,
                "original_latency_ms": prediction.get("latency_ms", 0),
                "recorded_cost_usd": prediction.get("estimated_api_cost_usd", 0),
            },
        }
        if rid not in existing:
            e.update(
                span_id=uuid.UUID(rid).hex,
                root_span_id=uuid.UUID(rid).hex,
                span_parents=[],
                expected=master.get("expected") or {},
            )
            e["metadata"]["label_from_image_review"] = True
        events.append(e)
        wanted[rid] = e
    for i in range(0, len(events), 200):
        api(keys, "POST", f"/v1/project_logs/{P}/insert", json={"events": events[i : i + 200]})
    views = api(keys, "GET", "/v1/view", params={"object_type": "project", "object_id": P})[
        "objects"
    ]
    view = next(v for v in views if v["name"] == "Web images — human category review")
    filt = "metadata.batch = 'web-aid-v1' AND metadata.record_type = 'model_review'"
    data = view["view_data"]
    data["search"]["filter"] = [{"text": filt, "label": filt, "originType": "btql"}]
    api(
        keys,
        "PATCH",
        "/v1/view/" + view["id"],
        json={
            "object_type": "project",
            "object_id": P,
            "view_data": data,
            "description": "One image and one model answer per row. Native message rendering in Thread view; human category goes in Expected. Historical API calls are replayed for review without new inference.",
        },
    )
    for attempt in range(6):
        rows = model_review_rows(keys)
        if len(rows) == len(events) and all(
            r["input"] == wanted[r["id"]]["input"] and r["output"] == wanted[r["id"]]["output"]
            for r in rows
        ):
            break
        time.sleep(5)
    assert len(rows) == len(events) == 5000
    for r in rows:
        e = wanted[r["id"]]
        assert r["input"] == e["input"] and r["output"] == e["output"]
        expected = (
            existing[r["id"]].get("expected")
            if r["id"] in existing
            else bycase[r["metadata"]["id"]].get("expected") or {}
        )
        assert r.get("expected") == expected
    report = {
        "rows": len(rows),
        "images": len(bycase),
        "rows_per_model": {
            m: sum(r["metadata"]["model"] == m for r in rows) for m in native.MODELS
        },
        "new_inference_calls": 0,
        "native_messages": True,
        "existing_human_labels_preserved": True,
        "url": "https://www.braintrust.dev/app/eval_engineering/p/"
        + PROJECT
        + "/logs?v="
        + quote(view["name"]),
    }
    (OUT / "model-review-verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    publish()
