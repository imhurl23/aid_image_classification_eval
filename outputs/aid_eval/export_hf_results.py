"""Build an image-free, allowlisted Hugging Face evaluation-results release."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def write_parquet(path, rows, labels):
    import pyarrow as pa
    import pyarrow.parquet as pq

    types = {key: pa.string() for key in rows[0]}
    for key in ["valid", "correct"]:
        types[key] = pa.bool_()
    for key in ["trial_index", "attempts", "input_tokens", "output_tokens"]:
        types[key] = pa.int64()
    for key in [
        "latency_ms",
        "provider_reported_cost_usd",
        "recorded_cost_usd",
        "token_estimated_cost_usd",
        "legacy_estimated_api_cost_usd",
    ]:
        types[key] = pa.float64()
    types["probabilities"] = pa.struct([(label, pa.float64()) for label in labels])
    table = pa.Table.from_pylist(rows, schema=pa.schema(list(types.items())))
    pq.write_table(table, path)
    assert pq.read_table(path).to_pylist() == rows


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str) and value.startswith("/Users/"):
        return Path(value).name
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definitions-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    config = read(base / "config.json")
    labels = config["labels"]
    source = args.definitions_source
    original = {c["id"]: c for c in read(source / "cases.json")}
    assert len(original) == 2000
    assert set(original) == set(read(base / "split-manifest.json")["test"])
    controlled = {c["id"]: c for c in read(base / "controlled-definitions-v1/cases.json")}
    assert len(controlled) == 2100
    models = ["luna", "jev", "clef", "d1", "d1-ultra"]
    studies = [
        ("native_baseline", original, base, "native-fresh-01", 2000),
        ("class_definitions", original, source / "runs", "aid-definitions-v1", 2000),
        ("controlled_definitions", controlled, base, "controlled-definitions-v1", 2100),
    ]
    audit, metrics, counts = [], [], {}
    for study, cases, root, prefix, count in studies:
        rows = []
        for model in models:
            run = root / f"{prefix}-{model}"
            path = run / f"{model}.jsonl"
            predictions = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(predictions) == count
            assert {p["id"] for p in predictions} == set(cases)
            manifest = read(run / "native-manifest.json")
            prompt = manifest["instructions"]
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            prompt_path = out / "prompts" / f"{study}.txt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            if prompt_path.exists():
                assert prompt_path.read_text() == prompt
            prompt_path.write_text(prompt)
            correct = valid = 0
            conditions = {}
            for p in sorted(predictions, key=lambda p: p["id"]):
                case = cases[p["id"]]
                original_id = case.get("original_id", case["id"])
                original_case = original[original_id]
                assert case["label"] == original_case["label"]
                assert p["input_sha256"] == case["sha256"]
                assert p.get("trial_index", 0) == 0
                is_valid = p["status"] == "ok" and p.get("label") in labels
                is_correct = is_valid and p["label"] == case["label"]
                valid += is_valid
                correct += is_correct
                probabilities = p.get("probabilities") if is_valid else None
                if probabilities is not None:
                    assert set(probabilities) == set(labels)
                    assert all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities.values())
                    assert abs(sum(probabilities.values()) - 1) < 0.000001
                condition = case.get("condition", "original")
                stats = conditions.setdefault(condition, Counter())
                stats.update(rows=1, valid=int(is_valid), correct=int(is_correct))
                usage = p.get("usage") or {}
                provider_cost = usage.get("cost")
                if not isinstance(provider_cost, (int, float)) or isinstance(provider_cost, bool):
                    provider_cost = None
                row = {
                    "study": study,
                    "run_name": f"{prefix}-{model}",
                    "model": model,
                    "requested_model": p.get("requested_model"),
                    "served_model": p.get("served_model"),
                    "case_id": p["id"],
                    "original_id": original_id,
                    "condition": condition,
                    "split": "test",
                    "reference_label": case["label"],
                    "predicted_label": p.get("label"),
                    "status": p["status"],
                    "valid": is_valid,
                    "correct": is_correct,
                    "probabilities": probabilities,
                    "trial_index": 0,
                    "input_sha256": p["input_sha256"],
                    "original_sha256": original_case["sha256"],
                    "source_repo": config["dataset"],
                    "source_revision": config["revision"],
                    "source_path": f"data/{case['label']}/{original_id}",
                    "prompt_sha256": prompt_sha,
                    "dataset_version": manifest["dataset_version"],
                    "latency_ms": p.get("latency_ms"),
                    "attempts": p.get("attempts"),
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "provider_reported_cost_usd": provider_cost,
                    "recorded_cost_usd": p.get("recorded_cost_usd"),
                    "token_estimated_cost_usd": p.get("token_estimated_cost_usd"),
                    "legacy_estimated_api_cost_usd": p.get("estimated_api_cost_usd"),
                    "cost_source": p.get("cost_source"),
                }
                rows.append(row)
            summary = read(run / f"{model}-summary.json")["trials"][0]
            assert summary["images"] == count and summary["valid"] == valid
            assert abs(summary["accuracy_all_attempts"] - correct / count) < 1e-12
            for condition, stats in conditions.items():
                metrics.append(
                    {
                        "study": study,
                        "model": model,
                        "condition": condition,
                        **stats,
                        "accuracy": stats["correct"] / stats["rows"],
                    }
                )
            audit.append(
                {
                    "study": study,
                    "model": model,
                    "source_file": f"{prefix}-{model}/{path.name}",
                    "source_sha256": sha(path),
                    "manifest_sha256": sha(run / "native-manifest.json"),
                    "rows": count,
                    "valid": valid,
                    "correct": correct,
                    "accuracy_matches_saved_summary": True,
                    "input_hashes_match_manifest": True,
                }
            )
        counts[study] = len(rows)
        write_rows(out / "data" / f"{study}.jsonl", rows)
        write_parquet(out / "data" / f"{study}.parquet", rows, labels)
    references = []
    for case in sorted(original.values(), key=lambda c: c["id"]):
        references.append(
            {
                "original_id": case["id"],
                "reference_label": case["label"],
                "sha256": case["sha256"],
                "width": case["dimensions"][0],
                "height": case["dimensions"][1],
                "source_repo": config["dataset"],
                "source_revision": config["revision"],
                "source_path": f"data/{case['label']}/{case['id']}",
                "split_source": config["split_source"],
                "split_revision": config["split_revision"],
            }
        )
    write_rows(out / "references/test_images.jsonl", references)
    write_json(out / "protocols/labels.json", labels)
    for name, path in [
        ("class_definitions", source / "protocol.json"),
        ("controlled_definitions", base / "controlled-definitions-v1/protocol.json"),
    ]:
        write_json(out / "protocols" / f"{name}.json", portable(read(path)))
    write_json(out / "metrics.json", metrics)
    write_json(
        out / "release_summary.json",
        {
            "release_date": "2026-10-29",
            "counts": counts,
            "total_predictions": sum(counts.values()),
            "unique_original_images": len(original),
            "image_files_included": False,
            "source_license": "unknown",
            "verified_runs": audit,
        },
    )
    for path in out.rglob("*"):
        if path.is_file():
            assert path.suffix in {".json", ".jsonl", ".txt", ".md", ".parquet"}
            if path.suffix == ".parquet":
                continue  # Exact round-trip against the scanned JSONL is checked above.
            text = path.read_text()
            assert not re.search(
                r"/Users/|data:image/|Bearer |sk-[A-Za-z0-9]{15}|hf_[A-Za-z0-9]{15}|X-Amz-Signature|attachment_id",
                text,
            ), path
    print(json.dumps({"counts": counts, "verified_runs": len(audit), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
