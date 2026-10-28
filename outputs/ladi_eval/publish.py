"""Publish saved LADI results and verifiable image attachments to Braintrust."""

import argparse
import hashlib
import io
import json
import os
import uuid
from pathlib import Path

import braintrust as bt
from PIL import Image
from run import prepare


def publish(folder, key, project="ladi-v2a-decision-comparison"):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    rows = [json.loads(x) for x in (folder / "truth.jsonl").read_text().splitlines()]
    bt.login(api_key=key)
    dataset = bt.init_dataset(
        project=project,
        name="ladi-validation-pilot",
        description="Five validation images for integration review. Not a representative benchmark.",
        metadata={
            "dataset_revision": manifest["dataset_revision"],
            "source": "MITLL/LADI-v2-dataset",
        },
    )
    attachments = {}
    ids = {}
    for row in rows:
        path = folder / "images" / Path(row["image_path"]).name
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["image_sha256"]
        attachment = bt.Attachment(data=raw, filename=path.name, content_type="image/png")
        assert attachment.upload()["upload_status"] == "done", "Image upload failed"
        attachments[row["id"]] = attachment
        rid = str(uuid.uuid5(uuid.NAMESPACE_URL, manifest["dataset_revision"] + ":" + row["id"]))
        ids[row["id"]] = dataset.insert(
            id=rid,
            input={"image": attachment},
            expected=row["labels"],
            metadata={
                "source_id": row["id"],
                "image_sha256": row["image_sha256"],
                "source": "https://huggingface.co/datasets/MITLL/LADI-v2-dataset",
                "split": manifest["split"],
            },
        )
    dataset.flush()
    summary = dataset.summarize()
    links = {
        "project_url": summary.project_url,
        "dataset_url": summary.dataset_url,
        "experiments": {},
    }
    for arm in manifest["arms"]:
        path = folder / (arm + ".jsonl")
        if not path.exists():
            continue
        results = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
        exp = bt.init(
            project=project,
            experiment=folder.name + "__" + arm,
            dataset=dataset,
            description="Validation integration pilot; five images, not final benchmark results.",
            metadata={
                "condition": manifest["image_condition"],
                "rubric": manifest["rubric"],
                "rubric_hash": manifest["rubric_hash"],
                "aggregate_metrics": json.loads((folder / (arm + "-scores.json")).read_text()),
            },
            update=True,
        )
        for row in rows:
            r = results[row["id"]]
            source = attachments[row["id"]]
            png = prepare(
                Image.open(io.BytesIO(source.data)),
                jev_limits=manifest["image_condition"] == "matched" or arm == "jev_multimodal",
            )
            assert hashlib.sha256(png).hexdigest() == r["input_image_sha256"], (
                "Reconstructed input differs from logged input"
            )
            actual = source
            if png != source.data:
                actual = bt.Attachment(
                    data=png,
                    filename=arm + "-" + source.reference["filename"],
                    content_type="image/png",
                )
                assert actual.upload()["upload_status"] == "done"
            scores = {"valid_output": int(r["status"] == "ok")}
            if r["status"] == "ok":
                scores.update(
                    {
                        k: float((p >= 0.5) == row["labels"][k])
                        for k, p in r["probabilities"].items()
                    }
                )
                scores["label_accuracy"] = sum(scores[k] for k in row["labels"]) / len(
                    row["labels"]
                )
            exp.log(
                id=ids[row["id"]],
                dataset_record_id=ids[row["id"]],
                input={"image": actual, "source_image": source},
                expected=row["labels"],
                output={
                    "probabilities": r.get("probabilities"),
                    "labels": {k: p >= 0.5 for k, p in r.get("probabilities", {}).items()},
                    "status": r["status"],
                },
                scores=scores,
                metadata={k: v for k, v in r.items() if k not in ("response", "probabilities")},
                metrics={"latency_ms": r["latency_ms"]},
            )
        exp.flush()
        links["experiments"][arm] = exp.summarize(summarize_scores=False).experiment_url
    # Read back attachments from the remote dataset and verify all image bytes.
    remote = bt.init_dataset(project=project, name="ladi-validation-pilot")
    verified = 0
    for row in remote.fetch():
        if row["id"] in ids.values():
            image = row["input"]["image"]
            if isinstance(image, dict):
                image = bt.ReadonlyAttachment(image)
            assert hashlib.sha256(image.data).hexdigest() == row["metadata"]["image_sha256"]
            verified += 1
    assert verified == len(rows)
    links["verified_remote_images"] = verified
    (folder / "braintrust-links.json").write_text(json.dumps(links, indent=2))
    print(json.dumps(links, indent=2))
    return links


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--env-file", type=Path)
    args = p.parse_args()
    key = os.environ.get("BRAINTRUST_API_KEY")
    if args.env_file:
        for line in args.env_file.read_text().splitlines():
            if line.startswith("BRAINTRUST_API_KEY="):
                key = line.split("=", 1)[1].strip().strip(chr(34)).strip(chr(39))
    if not key:
        raise RuntimeError("BRAINTRUST_API_KEY required")
    publish(args.folder, key)
