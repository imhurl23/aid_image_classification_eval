"""Braintrust-native AID evaluation. Run with --help for configuration."""

import argparse
import base64
import hashlib
import io
import json
import re
import threading
import time
from pathlib import Path

import braintrust as bt
import eval as protocol
import requests
from cost_accounting import reconcile
from credentials import add_credential_arguments, keys_load, validate_keys
from PIL import Image

BASE = Path(__file__).resolve().parent
CONFIG = json.loads((BASE / "config.json").read_text())
LABELS = CONFIG["labels"]
MODELS = {
    "luna": "gpt-6-luna",
    "jev": "jev-1.14.0",
    "clef": "clef",
    "d1": "d1",
    "d1-ultra": "d1-ultra",
}
PRICES = {"luna": 0.1, "jev": 0.042, "clef": 0.24, "d1": 0.04, "d1-ultra": 0.042}


def accuracy(output, expected):
    return int(output["status"] == "ok" and output["label"] == expected)


def valid_output(output):
    return int(output["status"] == "ok")


def request_payload(arm, raw, keys, instructions=None):
    instructions = protocol.INSTRUCTIONS if instructions is None else instructions
    with Image.open(io.BytesIO(raw)) as im:
        image_format = im.format
    mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[image_format]
    if arm in ("d1", "d1-ultra"):
        return (
            "https://api.liquid.ai/decisions/v1/systemone",
            {"Authorization": "Bearer " + keys["LIQUID_AI_API_KEY"]},
            {
                "model": MODELS[arm],
                "state": "",
                "images": ["data:" + mime + ";base64," + base64.b64encode(raw).decode()],
                "questions": {
                    "scene": {
                        "type": "choice",
                        "instructions": instructions,
                        "criteria": {label: label for label in LABELS},
                    }
                },
            },
        )
    if arm != "clef":
        url, key, headers, body = protocol.payload(arm, raw, LABELS)
        if arm == "luna":
            body["questions"][0]["instructions"] = instructions
            body["input"][0]["content"][0]["image_url"] = (
                "data:" + mime + ";base64," + base64.b64encode(raw).decode()
            )
        else:
            body["questions"]["scene"]["instructions"] = instructions
            for asset in body["assets"].values():
                asset["format"] = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}[image_format]
        return url, {**headers, "Authorization": "Bearer " + keys[key]}, body
    token = (
        keys.get("CLOUDFLARE_AUTH_TOKEN")
        or keys.get("CLOUDFLARE_API_TOKEN")
        or keys["CLOUDFLARE_API_KEY"]
    )
    url = f"https://api.cloudflare.com/client/v4/accounts/{keys['CLOUDFLARE_ACCOUNT_ID']}/ai/run/@cf/cloudflare/clef"
    return (
        url,
        {"Authorization": "Bearer " + token},
        {
            "model": "clef",
            "state": "",
            "images": ["data:" + mime + ";base64," + base64.b64encode(raw).decode()],
            "questions": {
                "scene": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": {label: label for label in LABELS},
                }
            },
        },
    )


def predict(arm, raw, keys, image, instructions=None):
    instructions = protocol.INSTRUCTIONS if instructions is None else instructions
    url, headers, body = request_payload(arm, raw, keys, instructions=instructions)
    result = {
        "status": "api_error",
        "requested_model": MODELS[arm],
        "input_sha256": hashlib.sha256(raw).hexdigest(),
    }
    start = time.perf_counter()
    # Log attachment references, never base64 request bodies or authorization headers.
    with bt.start_span(
        name=arm,
        type="llm",
        input={"image": image, "instructions": instructions, "labels": LABELS},
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
                    result.update(served_model=p.get("model"), usage=p.get("usage", {}))
                    if arm == "luna" and any(
                        a.get("type") == "refusal" for a in p.get("answers", [])
                    ):
                        result["status"] = "refusal"
                        break
                    result["status"] = "invalid_output"
                    label, probabilities = protocol.parse(arm, p, LABELS)
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Replay saved predictions and request only missing cases into a recovery experiment",
    )
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=["clef"])
    p.add_argument("--run-name", required=True)
    add_credential_arguments(p)
    p.add_argument("--project", default=CONFIG["project"], help="Your Braintrust project name")
    p.add_argument("--dataset", default="aid-test-2000")
    p.add_argument("--dataset-version")
    p.add_argument(
        "--cases-file", type=Path, help="Frozen manifest for an explicitly prepared dataset"
    )
    p.add_argument("--protocol-file", type=Path, help="Experiment transformation protocol")
    p.add_argument(
        "--instructions-file", type=Path, help="Frozen prompt used identically for every image"
    )
    p.add_argument(
        "--images-dir", type=Path, help="Optional local image cache; every file is hash-checked"
    )
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--max-concurrency", type=int, default=6)
    p.add_argument("--base-experiment", help="Optional comparison experiment in your project")
    args = p.parse_args()
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name)
        or args.trials < 1
        or args.max_concurrency < 1
    ):
        p.error("Invalid run name, trials, or concurrency")
    instructions = (
        args.instructions_file.read_text() if args.instructions_file else protocol.INSTRUCTIONS
    )
    if not instructions.strip():
        p.error("Instructions must not be empty")
    keys = keys_load(args.env_file, args.liquid_key_file)
    validate_keys(keys, args.models)
    bt.login(api_key=keys["BRAINTRUST_API_KEY"])
    dataset = bt.init_dataset(project=args.project, name=args.dataset, version=args.dataset_version)
    version = dataset.version
    dataset = bt.init_dataset(project=args.project, name=args.dataset, version=version)
    rows = list(dataset.fetch())
    # Check the remote frozen split against the original manifest before any model calls.
    source = BASE / ("test-2000-rerun-01" if args.dataset == "aid-test-2000" else "")
    cases = json.loads((args.cases_file or (source / "cases.json")).read_text())
    byid = {c["id"]: c for c in cases}
    assert len(rows) == len(cases)
    assert {r["metadata"]["source_id"] for r in rows} == set(byid)
    for row in rows:
        c = byid[row["metadata"]["source_id"]]
        assert row["expected"] == c["label"] and row["metadata"]["sha256"] == c["sha256"]
    out = BASE / args.run_name
    out.mkdir(exist_ok=True)
    manifest = {
        "dataset": args.dataset,
        "dataset_version": version,
        "models": args.models,
        "trials": args.trials,
        "instructions": instructions,
        "labels": LABELS,
        "image_policy": "Original image bytes, detected MIME type; no client resizing; provider preprocessing applies",
        "prices_per_million_input_tokens": PRICES,
        "retry_policy": "3 attempts; 429/5xx and transport errors",
        "scoring": "Failed requests count as incorrect; same 30-way choice validation as original eval",
    }
    if args.instructions_file:
        manifest["instructions_sha256"] = hashlib.sha256(instructions.encode()).hexdigest()
    if args.protocol_file:
        manifest["transform_protocol"] = json.loads(args.protocol_file.read_text())
        manifest["image_policy"] = manifest["transform_protocol"]["image_policy"]
        expected_prompt = manifest["transform_protocol"].get("prompt_sha256")
        if expected_prompt and hashlib.sha256(instructions.encode()).hexdigest() != expected_prompt:
            raise ValueError("Instructions do not match the frozen protocol")
    manifest_path = out / "native-manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise RuntimeError("Use a fresh run name or explicit --resume")
        assert json.loads(manifest_path.read_text()) == manifest
    elif args.resume:
        raise RuntimeError("No manifest to resume")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for arm in args.models:
        lock = threading.Lock()
        completed = 0
        records = []
        saved_path = out / f"{arm}.jsonl"
        saved = (
            [json.loads(line) for line in saved_path.read_text().splitlines()]
            if args.resume and saved_path.exists()
            else []
        )
        cache = {(r["id"], r["trial_index"]): r for r in saved}
        assert len(cache) == len(saved)
        for r in saved:
            assert r["input_sha256"] == byid[r["id"]]["sha256"]

        def task(input, hooks):
            nonlocal completed
            metadata = hooks.metadata
            image = input["image"]
            image = bt.ReadonlyAttachment(image) if isinstance(image, dict) else image
            cached = cache.get((metadata["source_id"], hooks.trial_index))
            if cached is not None:
                result = dict(cached)
                metadata["replayed_prediction"] = True
                metadata["original_experiment"] = args.run_name + "__" + arm
            else:
                if args.images_dir:
                    raw = (args.images_dir / byid[metadata["source_id"]]["image"]).read_bytes()
                else:
                    for download_attempt in range(3):
                        try:
                            raw = image.data
                            break
                        except RuntimeError:
                            if download_attempt == 2:
                                raise RuntimeError(
                                    "Attachment download failed after three attempts"
                                ) from None
                            time.sleep(2**download_attempt)
                            image = bt.ReadonlyAttachment(image.reference)
                assert hashlib.sha256(raw).hexdigest() == metadata["sha256"]
                result = predict(arm, raw, keys, image, instructions=instructions)
            result.update(id=metadata["source_id"], trial_index=hooks.trial_index)
            for field in ("original_id", "condition", "scene_class"):
                if field in metadata:
                    result[field] = metadata[field]
            with lock:
                records.append(result)
                if cached is None:
                    with (out / f"{arm}.jsonl").open("a") as f:
                        f.write(json.dumps(result) + "\n")
                completed += 1
                if completed % 100 == 0:
                    print(
                        json.dumps(
                            {"model": arm, "completed": completed, "total": len(rows) * args.trials}
                        ),
                        flush=True,
                    )
            return {k: v for k, v in result.items() if k != "response"}

        result = bt.Eval(
            args.project,
            data=dataset,
            task=task,
            scores=[accuracy, valid_output],
            experiment_name=args.run_name + "__" + arm + ("__resumed" if args.resume else ""),
            trial_count=args.trials,
            max_concurrency=args.max_concurrency,
            base_experiment_name=args.base_experiment or None,
            metadata=manifest,
            description="Native Eval: pinned AID images, frozen instructions and labels.",
        )
        bt.flush()
        summaries = []
        for trial in range(args.trials):
            trial_rows = [r for r in records if r["trial_index"] == trial]
            assert len(trial_rows) == len(cases)
            summary = protocol.summarize(cases, trial_rows, LABELS)
            summary["note"] = "Native Braintrust Eval; pinned dataset and original image bytes."
            summary["estimated_api_cost_usd"] = sum(r["estimated_api_cost_usd"] for r in trial_rows)
            summaries.append(summary)
        payload = {
            "experiment_url": result.summary.experiment_url,
            "dataset_version": version,
            "trials": summaries,
        }
        (out / f"{arm}-summary.json").write_text(json.dumps(payload, indent=2))
        print(
            json.dumps(
                {
                    "stage": "complete",
                    "model": arm,
                    "url": result.summary.experiment_url,
                    "accuracy": [s["accuracy_all_attempts"] for s in summaries],
                    "valid": [s["valid"] for s in summaries],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
