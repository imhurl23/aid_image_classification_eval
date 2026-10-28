"""Run and publish a balanced AID validation pilot. Never overwrite completed results."""

import argparse
import base64
import concurrent.futures
import hashlib
import json
import math
import time
import uuid
from pathlib import Path

import braintrust as bt
import requests

ROOT = Path(__file__).resolve().parent
INSTRUCTIONS = "Classify the overall aerial scene in the image into exactly one of the supplied AID scene classes. Select the class that best describes the scene as a whole, rather than every object visible. Use only the image."


def payload(arm, raw, labels):
    b64 = base64.b64encode(raw).decode()
    if arm == "luna":
        return (
            "https://api.openai.com/v1/decisions",
            "OPENAI_API_KEY",
            {},
            {
                "model": "gpt-6-luna",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": "data:image/jpeg;base64," + b64}
                        ],
                    }
                ],
                "questions": [
                    {
                        "name": "scene",
                        "type": "choice",
                        "instructions": INSTRUCTIONS,
                        "choices": [{"value": x, "description": x} for x in labels],
                    }
                ],
            },
        )
    ref = str(uuid.uuid4())
    return (
        "https://api.typesafe.ai/v1/systemone",
        "TYPESAFE_API_KEY",
        {"TypeSafe-Danger-Zone-Images": "true"},
        {
            "model": "jev-1.14.0",
            "state": {"image": ref},
            "assets": {ref: {"type": "image", "format": "jpg", "data_base64": b64}},
            "questions": {
                "scene": {
                    "type": "choice",
                    "instructions": INSTRUCTIONS,
                    "criteria": {x: x for x in labels},
                }
            },
        },
    )


def parse(arm, p, labels):
    if arm == "luna":
        if len(p["answers"]) != 1 or p["answers"][0]["name"] != "scene":
            raise ValueError("Wrong answers")
        a = p["answers"][0]
        if len(a["probabilities"]) != len(labels):
            raise ValueError("Wrong probability count")
        probs = {x["value"]: x["probability"] for x in a["probabilities"]}
    else:
        a = p["answers"]["scene"]
        probs = a["probabilities"]
    if a["type"] != "choice" or a["choice"] not in labels or set(probs) != set(labels):
        raise ValueError("Invalid choice")
    if (
        any(
            type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1
            for x in probs.values()
        )
        or abs(sum(probs.values()) - 1) > 0.02
    ):
        raise ValueError("Invalid probabilities")
    total = sum(probs.values())
    return a["choice"], {k: v / total for k, v in probs.items()}


def predict(arm, case, keys, labels):
    raw = (ROOT / "images" / case["image"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == case["sha256"]
    url, k, headers, body = payload(arm, raw, labels)
    headers["Authorization"] = "Bearer " + keys[k]
    start = time.perf_counter()
    result = {
        "id": case["id"],
        "status": "api_error",
        "input_sha256": case["sha256"],
        "requested_model": body["model"],
    }
    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=90)
            result["http_status"] = r.status_code
            if r.status_code in (429, 500, 502, 503, 504, 529) and attempt < 2:
                time.sleep(2**attempt)
                continue
            p = r.json()
            result["response"] = p
            if r.ok:
                result["status"] = "invalid_output"
                choice, probs = parse(arm, p, labels)
                raw_probs = (
                    p["answers"][0]["probabilities"]
                    if arm == "luna"
                    else p["answers"]["scene"]["probabilities"]
                )
                total = (
                    sum(x["probability"] for x in raw_probs)
                    if arm == "luna"
                    else sum(raw_probs.values())
                )
                result.update(
                    status="ok",
                    label=choice,
                    probabilities=probs,
                    raw_probability_sum=total,
                    probabilities_normalized=abs(total - 1) > 1e-9,
                    served_model=p.get("model"),
                    usage=p.get("usage", {}),
                )
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            result["error_type"] = type(e).__name__
        break
    result.update(latency_ms=(time.perf_counter() - start) * 1000, attempts=attempt + 1)
    return result


def summarize(cases, results, labels):
    byid = {x["id"]: x for x in results}
    cm = {x: {y: 0 for y in labels} for x in labels}
    correct = 0
    valid = 0
    nll = []
    for c in cases:
        r = byid[c["id"]]
        if r["status"] != "ok":
            continue
        valid += 1
        correct += r["label"] == c["label"]
        cm[c["label"]][r["label"]] += 1
        nll.append(-math.log(max(r["probabilities"][c["label"]], 1e-15)))
    f1 = []
    for x in labels:
        tp = cm[x][x]
        fp = sum(cm[y][x] for y in labels if y != x)
        fn = sum(cm[x][y] for y in labels if y != x)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return {
        "images": len(cases),
        "valid": valid,
        "coverage": valid / len(cases),
        "accuracy_all_attempts": correct / len(cases),
        "accuracy_valid_only": correct / valid if valid else None,
        "macro_f1_valid_only": sum(f1) / len(f1) if valid else None,
        "log_loss_valid_only": sum(nll) / len(nll) if nll else None,
        "confusion_matrix": cm,
        "note": "One validation image per class; integration pilot, not final benchmark.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-file", required=True, type=Path)
    args = p.parse_args()
    keys = {}
    for line in args.env_file.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            keys[k.strip()] = v.strip().strip(chr(34)).strip(chr(39))
    for k in ["OPENAI_API_KEY", "TYPESAFE_API_KEY", "BRAINTRUST_API_KEY"]:
        assert keys.get(k), k + " missing"
    config = json.loads((ROOT / "config.json").read_text())
    labels = config["labels"]
    cases = json.loads((ROOT / "cases.json").read_text())
    for arm in ["luna", "jev"]:
        path = ROOT / (arm + ".jsonl")
        existing = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        done = {x["id"] for x in existing}
        assert len(done) == len(existing)
        pending = [c for c in cases if c["id"] not in done]
        with path.open("a") as out, concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(predict, arm, c, keys, labels) for c in pending]
            for f in concurrent.futures.as_completed(futures):
                r = f.result()
                out.write(json.dumps(r) + "\n")
                out.flush()
                existing.append(r)
                print(arm, r["id"], r["status"], flush=True)
        (ROOT / (arm + "-summary.json")).write_text(
            json.dumps(summarize(cases, existing, labels), indent=2)
        )
    bt.login(api_key=keys["BRAINTRUST_API_KEY"])
    ds = bt.init_dataset(
        project=config["project"],
        name="aid-validation-pilot-30",
        description="One image per class; single-label 30-way Choice integration pilot.",
        metadata=config,
    )
    attachments = {}
    ids = {}
    for c in cases:
        a = bt.Attachment(
            data=str(ROOT / "images" / c["image"]), filename=c["image"], content_type="image/jpeg"
        )
        assert a.upload()["upload_status"] == "done"
        attachments[c["id"]] = a
        ids[c["id"]] = ds.insert(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, config["revision"] + c["id"])),
            input={"image": a},
            expected=c["label"],
            metadata={
                "source_id": c["id"],
                "sha256": c["sha256"],
                "split": "validation",
                "dimensions": [600, 600],
            },
        )
    ds.flush()
    s = ds.summarize()
    links = {"project": s.project_url, "dataset": s.dataset_url, "experiments": {}}
    for arm in ["luna", "jev"]:
        results = {
            r["id"]: r for r in map(json.loads, (ROOT / (arm + ".jsonl")).read_text().splitlines())
        }
        ex = bt.init(
            project=config["project"],
            experiment="pilot-30__" + arm,
            dataset=ds,
            update=True,
            description="Balanced validation pilot, 30 images. Not held-out benchmark results.",
            metadata={
                "instructions": INSTRUCTIONS,
                "label_order": labels,
                "image_policy": config["image_policy"],
                "aggregate_metrics": json.loads((ROOT / (arm + "-summary.json")).read_text()),
            },
        )
        for c in cases:
            r = results[c["id"]]
            ok = r["status"] == "ok"
            ex.log(
                id=ids[c["id"]],
                dataset_record_id=ids[c["id"]],
                input={"image": attachments[c["id"]]},
                expected=c["label"],
                output={
                    "label": r.get("label"),
                    "probabilities": r.get("probabilities"),
                    "status": r["status"],
                },
                scores={"accuracy": int(ok and r["label"] == c["label"]), "valid_output": int(ok)},
                metrics={"latency_ms": r["latency_ms"]},
                metadata={k: v for k, v in r.items() if k not in ("response", "probabilities")},
            )
        ex.flush()
        links["experiments"][arm] = ex.summarize(summarize_scores=False).experiment_url
    # Verify every stored source attachment downloads byte-for-byte.
    fresh = bt.init_dataset(project=config["project"], name="aid-validation-pilot-30")
    count = 0
    for row in fresh.fetch():
        a = row["input"]["image"]
        a = bt.ReadonlyAttachment(a) if isinstance(a, dict) else a
        assert hashlib.sha256(a.data).hexdigest() == row["metadata"]["sha256"]
        count += 1
    assert count == 30
    links["verified_images"] = count
    (ROOT / "braintrust-links.json").write_text(json.dumps(links, indent=2))
    print(json.dumps(links), flush=True)


if __name__ == "__main__":
    main()
