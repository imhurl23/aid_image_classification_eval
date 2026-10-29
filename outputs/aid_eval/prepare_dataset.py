"""Download frozen AID inputs; optionally publish them to your Braintrust project."""

import argparse
import hashlib
import io
import json
import uuid
from pathlib import Path

import braintrust as bt
import requests
from credentials import add_credential_arguments, keys_load, validate_keys
from PIL import Image

BASE = Path(__file__).resolve().parent
CONFIG = json.loads((BASE / "config.json").read_text())
SPLITS = {
    "pilot": (BASE / "cases.json", "aid-validation-pilot-30", "validation"),
    "test": (BASE / "test-2000-rerun-01/cases.json", "aid-test-2000", "test"),
}


def verify_image(raw, case):
    if hashlib.sha256(raw).hexdigest() != case["sha256"]:
        raise ValueError(f"Image hash mismatch: {case['id']}")
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        if list(image.size) != case["dimensions"]:
            raise ValueError(f"Image dimensions mismatch: {case['id']}")
        return {"JPEG": "image/jpeg", "PNG": "image/png"}[image.format]


def prepare(cases, images_dir):
    images_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for case in cases:
        path = images_dir / case["image"]
        if path.exists():
            raw = path.read_bytes()
            mime = verify_image(raw, case)
        else:
            source = f"data/{case['label']}/{case['id']}"
            response = requests.get(
                f"https://huggingface.co/datasets/{CONFIG['dataset']}/resolve/"
                f"{CONFIG['revision']}/{source}",
                timeout=90,
            )
            response.raise_for_status()
            raw = response.content
            mime = verify_image(raw, case)
            path.write_bytes(raw)
        prepared.append((case, path, mime))
    return prepared


def publish(prepared, project, name, split, keys):
    validate_keys(keys)
    bt.login(api_key=keys["BRAINTRUST_API_KEY"])
    dataset = bt.init_dataset(project=project, name=name)
    existing = list(dataset.fetch())
    if existing:
        raise ValueError("Dataset is not empty; choose a new --dataset name")
    for case, path, mime in prepared:
        attachment = bt.Attachment(data=str(path), filename=case["image"], content_type=mime)
        if attachment.upload()["upload_status"] != "done":
            raise RuntimeError(f"Attachment upload failed: {case['id']}")
        dataset.insert(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, CONFIG["revision"] + case["id"])),
            input={"image": attachment},
            expected=case["label"],
            metadata={
                "source_id": case["id"],
                "sha256": case["sha256"],
                "dimensions": case["dimensions"],
                "split": split,
            },
        )
    dataset.flush()
    print(f"Published {len(prepared)} images to {project}/{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default="pilot")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true", help="Upload images and dataset rows")
    parser.add_argument("--project", help="Required with --publish")
    parser.add_argument("--dataset", help="Name of an empty dataset; defaults to the split name")
    add_credential_arguments(parser)
    args = parser.parse_args()
    if args.publish and not args.project:
        parser.error("--publish requires --project")
    keys = keys_load(args.env_file, args.liquid_key_file) if args.publish else None
    if args.publish:
        validate_keys(keys)
    manifest, name, split = SPLITS[args.split]
    prepared = prepare(json.loads(manifest.read_text()), args.images_dir)
    print(f"Verified {len(prepared)} images in {args.images_dir}")
    if args.publish:
        publish(prepared, args.project, args.dataset or name, split, keys)


if __name__ == "__main__":
    main()
