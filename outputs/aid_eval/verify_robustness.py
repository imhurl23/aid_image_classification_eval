"""Read back the definitions-matched experiments and verify prompts, scores, and provenance."""

import argparse
import concurrent.futures
import hashlib
import json
from collections import Counter
from pathlib import Path

import braintrust as bt
import native_eval as native
from credentials import add_credential_arguments, keys_load, validate_keys

BASE = Path(__file__).resolve().parent
ROOT = BASE / "controlled-definitions-v1"


def verify(model):
    folder = BASE / f"controlled-definitions-v1-{model}"
    manifest = json.loads((folder / "native-manifest.json").read_text())
    prompt = (ROOT / "prompt.txt").read_text()
    if manifest["instructions"] != prompt:
        raise ValueError("Wrong local run prompt")
    experiment = bt.init(
        project=native.CONFIG["project"],
        experiment=folder.name + "__" + model,
        open=True,
        set_current=False,
    )
    rows = list(experiment.fetch())
    roots = [r for r in rows if not r.get("span_parents")]
    calls = [r for r in rows if r.get("span_attributes", {}).get("type") == "llm"]
    cases = {c["id"]: c for c in json.loads((ROOT / "cases.json").read_text())}
    saved = {
        r["id"]: r for r in map(json.loads, (folder / f"{model}.jsonl").read_text().splitlines())
    }
    if len(roots) != 2100 or len(calls) != 2100:
        raise ValueError(
            f"{model}: expected 2100 roots and fresh model calls; found {len(roots)}, {len(calls)}"
        )
    if {r["metadata"]["source_id"] for r in roots} != set(cases):
        raise ValueError("Remote case IDs differ from frozen manifest")
    scores = {}
    for row in rows:
        scores.setdefault(row["root_span_id"], {}).update(row.get("scores") or {})
    for row in roots:
        case = cases[row["metadata"]["source_id"]]
        if not row.get("origin") or row["metadata"].get("replayed_prediction"):
            raise ValueError("Missing dataset origin or unexpected cached prediction")
        output = row["output"]
        if row["expected"] != case["label"] or output["input_sha256"] != case["sha256"]:
            raise ValueError("Remote labels or image hashes differ")
        if output["status"] != saved[case["id"]]["status"] or output.get("label") != saved[
            case["id"]
        ].get("label"):
            raise ValueError("Remote prediction differs from local export")
        if scores[row["root_span_id"]]["accuracy"] != native.accuracy(output, case["label"]):
            raise ValueError("Remote accuracy differs from expected scoring")
    for row in calls:
        if row["input"]["instructions"] != prompt:
            raise ValueError("Class definitions missing from a remote model span")
    sample = roots[0]
    attachment = sample["input"]["image"]
    if isinstance(attachment, dict):
        attachment = bt.ReadonlyAttachment(attachment)
    if hashlib.sha256(attachment.data).hexdigest() != sample["metadata"]["sha256"]:
        raise ValueError("Remote attachment bytes differ")
    return model, {
        "root_rows": len(roots),
        "fresh_model_calls": len(calls),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "conditions": dict(Counter(r["metadata"]["condition"] for r in roots)),
        "statuses": dict(Counter(r["output"]["status"] for r in roots)),
        "dataset_origins_verified": True,
        "scores_verified": True,
        "all_model_prompts_verified": True,
        "sample_remote_attachment_verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_credential_arguments(parser)
    args = parser.parse_args()
    keys = keys_load(args.env_file, args.liquid_key_file)
    validate_keys(keys)
    bt.login(api_key=keys["BRAINTRUST_API_KEY"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = dict(pool.map(verify, native.MODELS))
    (ROOT / "verification.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
