"""Native Braintrust decisions about added blur and paired sharpness."""

import argparse
import base64
import hashlib
import json
import threading
import time
import uuid
from pathlib import Path

import braintrust as bt
import eval as protocol
import native_eval as native
import requests
from cost_accounting import reconcile
from credentials import add_credential_arguments, keys_load, validate_keys

BASE = Path(__file__).resolve().parent
OUT = BASE / "visual-decisions-v1"
SOURCE = BASE / "controlled-300-v1"
MODELS = native.MODELS
PRICES = native.PRICES
PROJECT = native.CONFIG["project"]
SPECS = {
    "added-blur": {
        "instructions": "Determine whether additional Gaussian blur has been applied to this aerial image. Choose added_blur if it appears to have been smoothed by an additional Gaussian blur of radius 2 pixels at its 600 by 600 resolution; otherwise choose original. The source photographs can already contain natural softness. Use visual evidence only.",
        "criteria": {
            "original": "Original aerial photograph without the additional blur treatment",
            "added_blur": "Aerial photograph with additional Gaussian blur, radius 2 pixels",
        },
    },
    "paired-sharpness": {
        "instructions": "The first image is A and the second image is B. They show the same aerial scene. Which image is sharper, with better-preserved fine detail and less blur? Choose A or B. Judge sharpness only, not scene content, brightness, or contrast.",
        "criteria": {"A": "The first image is sharper", "B": "The second image is sharper"},
    },
}


def request_payload(arm, raws, keys, spec):
    url, headers, body = native.request_payload(arm, raws[0], keys)
    encoded = [base64.b64encode(raw).decode() for raw in raws]
    question = {
        "type": "choice",
        "instructions": spec["instructions"],
        "criteria": spec["criteria"],
    }
    if arm == "luna":
        body["input"] = [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64," + b}
                    for b in encoded
                ],
            }
        ]
        body["questions"] = [
            {
                "name": "scene",
                "type": "choice",
                "instructions": spec["instructions"],
                "choices": [{"value": k, "description": v} for k, v in spec["criteria"].items()],
            }
        ]
    else:
        body["questions"] = {"scene": question}
        if arm == "jev":
            ids = [str(uuid.uuid4()) for _ in raws]
            body["state"] = {("A" if i == 0 else "B"): ref for i, ref in enumerate(ids)}
            body["assets"] = {
                ref: {"type": "image", "format": "png", "data_base64": b}
                for ref, b in zip(ids, encoded)
            }
        else:
            body["state"] = "The first image is A and the second image, if present, is B."
            body["images"] = ["data:image/png;base64," + b for b in encoded]
    return url, headers, body


def predict(arm, raws, keys, image, spec):
    url, headers, body = request_payload(arm, raws, keys, spec)
    result = {
        "status": "api_error",
        "requested_model": MODELS[arm],
        "input_sha256": [hashlib.sha256(raw).hexdigest() for raw in raws],
    }
    start = time.perf_counter()
    # Log attachment references, never base64 request bodies or authorization headers.
    with bt.start_span(
        name=arm,
        type="llm",
        input={
            "image": image,
            "instructions": spec["instructions"],
            "labels": list(spec["criteria"]),
        },
        metadata={"model": MODELS[arm], "provider": "cloudflare" if arm == "clef" else arm},
    ) as span:
        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, json=body, timeout=90)
                result["http_status"] = response.status_code
                if response.status_code in (429, 500, 502, 503, 504, 520, 529) and attempt < 2:
                    time.sleep(2**attempt)
                    continue
                envelope = response.json()
                p = envelope.get("result", envelope) if arm == "clef" else envelope
                result["response"] = p
                if not response.ok:
                    result["api_errors"] = envelope.get("errors", [])
                if response.ok and p and envelope.get("success", True):
                    result["status"] = "invalid_output"
                    label, probabilities = protocol.parse(arm, p, list(spec["criteria"]))
                    result.update(
                        status="ok",
                        label=label,
                        probabilities=probabilities,
                        served_model=p.get("model"),
                        usage=p.get("usage", {}),
                    )
                break
            except requests.RequestException as exc:
                result["error_type"] = type(exc).__name__
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
            except (ValueError, KeyError, TypeError) as exc:
                result["error_type"] = type(exc).__name__
                break
        result.update(attempts=attempt + 1, latency_ms=(time.perf_counter() - start) * 1000)
        usage = result.get("usage", {})
        metrics = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        }
        metrics["tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
        metrics.update(reconcile(result, arm, PRICES))
        span.log(
            output=result,
            metrics=metrics,
            metadata={"http_status": result.get("http_status"), "attempts": result["attempts"]},
        )
        if result["status"] != "ok":
            span.log(
                error=f"{result['status']}: HTTP {result.get('http_status')}, {result.get('error_type', '')}"
            )
    return result


def prepare():
    OUT.mkdir(exist_ok=True)
    cases = json.loads((SOURCE / "cases.json").read_text())
    refs = {
        r["id"]: r["reference"]
        for r in map(json.loads, (SOURCE / "attachments.jsonl").read_text().splitlines())
    }
    byid = {}
    for c in cases:
        if c["condition"] in ("control", "blur_2"):
            byid.setdefault(c["original_id"], {})[c["condition"]] = c
    plan = {
        "sources": 300,
        "rows_per_task": 600,
        "specs": SPECS,
        "image_policy": "Existing 600x600 lossless PNG controls and GaussianBlur radius 2 variants; no resizing. No filenames, labels, or condition metadata sent to models.",
        "analysis": "Source-cluster paired exact sign flips across 20 model comparisons, Holm family-wise correction. Source-cluster bootstrap 95% intervals. Both pair orders; descriptive position and consistency diagnostics.",
        "limitations": "Added-blur ground truth is treatment assignment, not human perceptual blur labels. Same exploratory 300 sources as robustness experiment; original is the operational sharper reference, not human adjudication.",
    }
    planpath = OUT / "protocol.json"
    if planpath.exists():
        assert json.loads(planpath.read_text()) == plan
    planpath.write_text(json.dumps(plan, indent=2))
    manifests = {}
    for task in SPECS:
        rows = []
        for source, variants in sorted(byid.items()):
            control, blur = variants["control"], variants["blur_2"]
            for condition in (
                ("original", "added_blur")
                if task == "added-blur"
                else ("original_first", "original_second")
            ):
                selected = (
                    ([control] if condition == "original" else [blur])
                    if task == "added-blur"
                    else ([control, blur] if condition == "original_first" else [blur, control])
                )
                expected = (
                    condition
                    if task == "added-blur"
                    else ("A" if condition == "original_first" else "B")
                )
                rid = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        "visual-decisions-v1/" + task + "/" + source + "/" + condition,
                    )
                )
                rows.append(
                    {
                        "id": rid,
                        "input": {"images": [refs[c["id"]] for c in selected]},
                        "expected": expected,
                        "metadata": {
                            "source_id": rid,
                            "original_id": source,
                            "scene_class": control["label"],
                            "condition": condition,
                            "task": task,
                            "files": [c["image"] for c in selected],
                            "sha256": [c["sha256"] for c in selected],
                        },
                        "tags": [task, condition, control["label"]],
                    }
                )
        rows.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        assert len(rows) == 600
        ds = bt.init_dataset(project=PROJECT, name="aid-" + task + "-v1", metadata=plan)
        for row in rows:
            ds.insert(**row)
        ds.flush()
        ds = bt.init_dataset(project=PROJECT, name="aid-" + task + "-v1")
        remote = list(ds.fetch())
        assert len(remote) == 600
        manifests[task] = {"version": ds.version, "url": ds.summarize().dataset_url, "rows": rows}
    (OUT / "datasets.json").write_text(json.dumps(manifests, indent=2))
    print(
        json.dumps({k: {"version": v["version"], "url": v["url"]} for k, v in manifests.items()}),
        flush=True,
    )


def run(arm, pilot=False, keys=None):
    manifests = json.loads((OUT / "datasets.json").read_text())
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    keys = keys if keys is not None else keys_load()
    validate_keys(keys, [arm])
    for task, spec in SPECS.items():
        savedpath = folder / (task + ".jsonl")
        cache = (
            {r["id"]: r for r in map(json.loads, savedpath.read_text().splitlines())}
            if savedpath.exists()
            else {}
        )
        rows = manifests[task]["rows"]
        byid = {r["id"]: r for r in rows}
        ds = bt.init_dataset(
            project=PROJECT, name="aid-" + task + "-v1", version=manifests[task]["version"]
        )
        if pilot:
            source = sorted({r["metadata"]["original_id"] for r in rows})[0]
            data = [r for r in ds.fetch() if r["metadata"]["original_id"] == source]
        else:
            data = ds
        lock = threading.Lock()
        results = []

        def evaluate(input, hooks):
            meta = hooks.metadata
            rid = meta["source_id"]
            assert byid[rid]["metadata"] == {k: meta[k] for k in byid[rid]["metadata"]}
            if rid in cache:
                out = cache[rid]
                assert out["input_sha256"] == meta["sha256"]
                hooks.metadata["replayed_prediction"] = True
            else:
                raws = [(SOURCE / "images" / f).read_bytes() for f in meta["files"]]
                assert [hashlib.sha256(r).hexdigest() for r in raws] == meta["sha256"]
                # Use verified local bytes matching dataset attachments to avoid redundant signed-URL downloads.
                out = predict(arm, raws, keys, input["images"], spec)
                out.update(
                    id=rid,
                    original_id=meta["original_id"],
                    condition=meta["condition"],
                    expected=byid[rid]["expected"],
                    task=task,
                )
                with lock:
                    with savedpath.open("a") as f:
                        f.write(json.dumps(out) + "\n")
            with lock:
                results.append(out)
                if len(results) % 100 == 0:
                    print(
                        json.dumps({"model": arm, "task": task, "completed": len(results)}),
                        flush=True,
                    )
            return {k: v for k, v in out.items() if k != "response"}

        result = bt.Eval(
            PROJECT,
            data=data,
            task=evaluate,
            scores=[native.accuracy, native.valid_output],
            experiment_name="visual-decisions-v1-"
            + task
            + "__"
            + arm
            + ("__pilot" if pilot else ""),
            max_concurrency=4,
            metadata={
                "protocol": json.loads((OUT / "protocol.json").read_text()),
                "dataset_version": manifests[task]["version"],
            },
            description="Added-blur detection or paired sharpness; preserved 600px PNGs; source-cluster analysis.",
        )
        bt.flush()
        summary = {
            "url": result.summary.experiment_url,
            "n": len(results),
            "valid": sum(r["status"] == "ok" for r in results),
            "accuracy": sum(
                r["status"] == "ok" and r.get("label") == r["expected"] for r in results
            )
            / len(results),
            "cost": sum(r.get("estimated_api_cost_usd", 0) for r in results),
        }
        (folder / (task + ("-pilot" if pilot else "") + "-summary.json")).write_text(
            json.dumps(summary, indent=2)
        )
        print(json.dumps({"model": arm, "task": task, **summary}), flush=True)
        assert summary["n"] == (2 if pilot else 600)
        if pilot:
            assert summary["valid"] == 2, (
                "Pilot schema failure: inspect saved results before full run"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--model", choices=list(MODELS))
    p.add_argument("--pilot", action="store_true")
    add_credential_arguments(p)
    args = p.parse_args()
    if not args.prepare and not args.model:
        p.error("Choose --prepare or --model")
    if args.pilot and not args.model:
        p.error("--pilot requires --model")
    keys = keys_load(args.env_file, args.liquid_key_file)
    validate_keys(keys, [args.model] if args.model else [])
    bt.login(api_key=keys["BRAINTRUST_API_KEY"])
    if args.prepare:
        prepare()
    if args.model:
        run(args.model, args.pilot, keys=keys)
