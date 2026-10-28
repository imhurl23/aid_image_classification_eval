"""Five-model web-image inference, native review logs, and scoring after human labels."""

import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

import braintrust as bt
import native_eval as native
import requests
from credentials import add_credential_arguments, keys_load, validate_keys
from native_review import review_input

BASE = Path(__file__).resolve().parent
OUT = BASE / "web-aid-v1"
BATCH = "web-aid-v1"
PROJECT = native.CONFIG["project"]
REVIEW_INSTRUCTIONS = "Classify the overall scene shown in the image. Open the Review tab and select web_scene_category. Choose one best matching AID category, or Ambiguous, Other, NotAerial, or Unusable. Label the image before consulting model predictions."
REVIEW_FIELD = "web_scene_category"
EXCLUSIONS = ["Ambiguous", "Other", "NotAerial", "Unusable"]


def api(keys, method, path, **kwargs):
    for attempt in range(4):
        r = requests.request(
            method,
            "https://api.braintrust.dev" + path,
            headers={"Authorization": "Bearer " + keys["BRAINTRUST_API_KEY"]},
            timeout=90,
            **kwargs,
        )
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            time.sleep(2**attempt)
            continue
        if not r.ok:
            raise RuntimeError(f"Braintrust HTTP {r.status_code}: {r.text[:300]}")
        return r.json()


def review_setup(keys, logger):
    labels = native.LABELS + EXCLUSIONS
    existing = api(keys, "GET", "/v1/project_score", params={"project_id": logger.id})
    scores = existing.get("objects", existing if isinstance(existing, list) else [])
    found = next((s for s in scores if s["name"] == REVIEW_FIELD), None)
    if found is None:
        found = api(
            keys,
            "POST",
            "/v1/project_score",
            json={
                "project_id": logger.id,
                "name": REVIEW_FIELD,
                "score_type": "categorical",
                "description": "Label the overall aerial scene using the image only. Choose one AID scene class. Use Ambiguous when no unique class is defensible, Other for a scene outside the taxonomy, NotAerial for a ground-level/non-aerial image, and Unusable for a map, diagram, unreadable or corrupted image. Search terms are not ground truth. Review the web-human-review rows before inspecting model predictions. This writes the category to expected; numeric category values are identifiers, not accuracy.",
                "categories": [
                    {"name": label, "value": i / (len(labels) - 1)}
                    for i, label in enumerate(labels)
                ],
                "config": {
                    "destination": "expected",
                    "multi_select": False,
                    "object_types": ["project_logs", "dataset"],
                    "condition": {
                        "when": {
                            "clauses": [
                                "metadata.batch = 'web-aid-v1' AND metadata.record_type = 'human_review'"
                            ]
                        }
                    },
                },
            },
        )
    assert found["config"]["destination"] == "expected"
    info = {
        "project_id": logger.id,
        "review_field": REVIEW_FIELD,
        "review_score_id": found["id"],
        "logs_url": "https://www.braintrust.dev/app/eval_engineering/p/" + PROJECT + "/logs",
        "review_filter": "metadata.batch = 'web-aid-v1' AND metadata.record_type = 'human_review'",
    }
    (OUT / "braintrust.json").write_text(json.dumps(info, indent=2))
    return info


def run(limit=None, keys=None):
    keys = keys if keys is not None else keys_load()
    validate_keys(keys, native.MODELS)
    bt.login(api_key=keys["BRAINTRUST_API_KEY"])
    logger = bt.init_logger(project=PROJECT)
    info = review_setup(keys, logger)
    cases = json.loads((OUT / "cases.json").read_text())
    if limit:
        cases = cases[:limit]
    ds = bt.init_dataset(
        project=PROJECT,
        name=BATCH,
        description="Web aerial-image candidates; expected blank until human review; search categories are retrieval targets only.",
    )
    path = OUT / "predictions.jsonl"
    saved = list(map(json.loads, path.read_text().splitlines())) if path.exists() else []
    cache = {(r["id"], r["model"]): r for r in saved}
    assert len(cache) == len(saved)
    refpath = OUT / "attachments.jsonl"
    refs = (
        {r["id"]: r["reference"] for r in map(json.loads, refpath.read_text().splitlines())}
        if refpath.exists()
        else {}
    )
    donepath = OUT / "published.jsonl"
    done = (
        {r["id"] for r in map(json.loads, donepath.read_text().splitlines())}
        if donepath.exists()
        else set()
    )
    lock = threading.Lock()
    completed = 0

    def work(c):
        nonlocal completed
        if c["id"] in done:
            return
        raw = (OUT / "images" / c["image"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == c["sha256"]
        if c["id"] not in refs:
            a = bt.Attachment(
                data=str(OUT / "images" / c["image"]),
                filename=c["id"][:16] + ".png",
                content_type="image/png",
            )
            for attempt in range(4):
                try:
                    assert a.upload()["upload_status"] == "done"
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
            with lock:
                refs[c["id"]] = a.reference
                with refpath.open("a") as f:
                    f.write(json.dumps({"id": c["id"], "reference": a.reference}) + "\n")
        image = refs[c["id"]]
        rid = str(uuid.uuid5(uuid.NAMESPACE_URL, BATCH + "/review/" + c["id"]))
        iid = str(uuid.uuid5(uuid.NAMESPACE_URL, BATCH + "/inference/" + c["id"]))
        # Preserve any human edits on resume: never re-log expected for an existing review row.
        review_cache = OUT / "review-published.jsonl"
        with lock:
            reviewed = (
                {r["id"] for r in map(json.loads, review_cache.read_text().splitlines())}
                if review_cache.exists()
                else set()
            )
            if c["id"] not in reviewed:
                meta = {
                    **c,
                    "batch": BATCH,
                    "record_type": "human_review",
                    "review_log_id": rid,
                    "inference_log_id": iid,
                }
                logger.log(
                    id=rid,
                    input=review_input(image),
                    expected={},
                    span_attributes={"name": "Scene classification — five-model comparison"},
                    metadata=meta,
                    tags=[BATCH, "web-human-review", "needs-review"],
                    allow_concurrent_with_spans=True,
                )
                ds.insert(
                    id=c["id"],
                    input={"image": image},
                    expected={},
                    metadata=meta,
                    tags=[BATCH, "needs-review"],
                )
                with review_cache.open("a") as f:
                    f.write(json.dumps({"id": c["id"], "review_log_id": rid}) + "\n")
        outputs = {}
        with logger.start_span(
            name="web-image-five-model-inference",
            id=iid,
            input={"image": image},
            metadata={
                "batch": BATCH,
                "record_type": "inference",
                "source_id": c["id"],
                "review_log_id": rid,
                "sha256": c["sha256"],
            },
            tags=[BATCH, "web-model-predictions"],
        ) as span:
            for model in native.MODELS:
                result = cache.get((c["id"], model))
                if result is not None:
                    assert result["input_sha256"] == c["sha256"], "Cached input bytes differ"
                if result is None:
                    result = native.predict(model, raw, keys, image)
                    result.update(id=c["id"], model=model, review_log_id=rid, inference_log_id=iid)
                    with lock:
                        with path.open("a") as f:
                            f.write(json.dumps(result) + "\n")
                        cache[c["id"], model] = result
                outputs[model] = {
                    k: v
                    for k, v in result.items()
                    if k not in ("response", "id", "review_log_id", "inference_log_id")
                }
            choices = [r.get("label") for r in outputs.values() if r["status"] == "ok"]
            span.log(
                output=outputs,
                scores={"valid_output": sum(r["status"] == "ok" for r in outputs.values()) / 5},
                metadata={"distinct_predictions": len(set(choices))},
            )
        with lock:
            with donepath.open("a") as f:
                f.write(json.dumps({"id": c["id"]}) + "\n")
            completed += 1
            if completed % 25 == 0:
                print(
                    json.dumps({"completed_images": completed, "models_per_image": 5}), flush=True
                )

    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(work, cases):
            pass
    ds.flush()
    bt.flush()
    result = {
        "images": len(cases),
        "predictions": sum((c["id"], m) in cache for c in cases for m in native.MODELS),
        "valid": {
            m: sum(cache[c["id"], m]["status"] == "ok" for c in cases) for m in native.MODELS
        },
        "estimated_api_cost_usd": sum(
            cache[c["id"], m].get("estimated_api_cost_usd", 0) for c in cases for m in native.MODELS
        ),
        "dataset_url": ds.summarize().dataset_url,
        "dataset_version": ds.version,
        "accuracy_status": "Awaiting human category labels; no accuracy scores have been fabricated.",
        **info,
    }
    (OUT / ("pilot-summary.json" if limit else "summary.json")).write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result), flush=True)


def review_rows(keys, project_id):
    query = f"select id, input, expected, metadata from project_logs('{project_id}') where metadata.batch = 'web-aid-v1' and metadata.record_type = 'human_review' limit 1000"
    d = api(keys, "POST", "/btql", json={"query": query})
    return d["data"]


def sync_reviewed_dataset(ds, accepted):
    """Make the next dataset version match the current eligible human labels."""
    accepted_ids = {row["metadata"]["id"] for row, _ in accepted}
    # Materialize the snapshot before queuing edits to it.
    for row in list(ds.fetch()):
        if row["id"] not in accepted_ids:
            ds.delete(row["id"])
    for r, label in accepted:
        ds.insert(
            id=r["metadata"]["id"],
            input=r["input"],
            expected=label,
            metadata={
                **r["metadata"],
                "human_review_log_id": r["id"],
                "review_status": "human_labeled",
            },
        )
    ds.flush()


def score_reviewed(keys=None):
    keys = keys if keys is not None else keys_load()
    validate_keys(keys)
    bt.login(api_key=keys["BRAINTRUST_API_KEY"])
    info = json.loads((OUT / "braintrust.json").read_text())
    rows = review_rows(keys, info["project_id"])
    accepted = []
    excluded = collections.Counter()
    for r in rows:
        expected = r.get("expected")
        label = expected.get(REVIEW_FIELD) if isinstance(expected, dict) else expected
        if label in native.LABELS:
            accepted.append((r, label))
        elif label in EXCLUSIONS:
            excluded[label] += 1
        elif label is not None:
            raise ValueError("Unknown reviewed label; do not infer a replacement")
    status = {
        "review_rows": len(rows),
        "reviewed_in_scope": len(accepted),
        "excluded": dict(excluded),
        "unreviewed": len(rows) - len(accepted) - sum(excluded.values()),
    }
    cache = (
        {
            (r["id"], r["model"]): r
            for r in map(json.loads, (OUT / "predictions.jsonl").read_text().splitlines())
        }
        if accepted
        else {}
    )
    for row, _ in accepted:
        for model in native.MODELS:
            if (row["metadata"]["id"], model) not in cache:
                raise ValueError("Missing cached prediction for a reviewed image")
    ds = bt.init_dataset(project=PROJECT, name=BATCH + "-human-reviewed")
    sync_reviewed_dataset(ds, accepted)
    if not accepted:
        bt.flush()
        (OUT / "review-progress.json").write_text(json.dumps(status, indent=2))
        print(json.dumps(status))
        return
    version = ds.version
    for model in native.MODELS:

        def task(input, hooks):
            saved = cache[hooks.metadata["id"], model]
            return {k: v for k, v in saved.items() if k != "response"}

        result = bt.Eval(
            PROJECT,
            data=bt.init_dataset(project=PROJECT, name=BATCH + "-human-reviewed", version=version),
            task=task,
            scores=[native.accuracy, native.valid_output],
            experiment_name=f"{BATCH}-human-scored-{version}__{model}",
            metadata={
                "cached_predictions": True,
                "no_new_model_calls": True,
                "human_reviewed_dataset_version": version,
            },
        )
        status[model] = {
            "url": result.summary.experiment_url,
            "accuracy": sum(
                cache[r["metadata"]["id"], model]["status"] == "ok"
                and cache[r["metadata"]["id"], model].get("label") == label
                for r, label in accepted
            )
            / len(accepted),
        }
    bt.flush()
    (OUT / "review-progress.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    p.add_argument("--score-reviewed", action="store_true")
    add_credential_arguments(p)
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    if args.score_reviewed and args.limit is not None:
        p.error("--limit cannot be combined with --score-reviewed")
    keys = keys_load(args.env_file, args.liquid_key_file)
    if args.score_reviewed:
        score_reviewed(keys=keys)
    else:
        run(args.limit, keys=keys)
        if not args.limit:
            from publish_model_review_rows import publish

            publish(keys=keys)
