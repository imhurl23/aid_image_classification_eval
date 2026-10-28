"""Freeze a balanced, label-stratified controlled image transformation experiment."""

import argparse
import concurrent.futures as cf
import hashlib
import json
import time
import uuid
from pathlib import Path

import braintrust as bt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

BASE = Path(__file__).resolve().parent
OUT = BASE / "controlled-300-v1"
CONDITIONS = [
    "control",
    "blur_1",
    "blur_2",
    "contrast_0.5",
    "contrast_1.5",
    "shift_left_60",
    "shift_right_60",
]
SEED = "aid-controlled-20261022-v1"


def transform(image, condition):
    if condition == "control":
        return image.copy()
    if condition.startswith("blur_"):
        return image.filter(ImageFilter.GaussianBlur(float(condition.split("_")[1])))
    if condition.startswith("contrast_"):
        return ImageEnhance.Contrast(image).enhance(float(condition.split("_")[1]))
    arr = np.asarray(image)
    padded = np.pad(arr, ((0, 0), (60, 60), (0, 0)), mode="reflect")
    start = 120 if condition == "shift_left_60" else 0
    return Image.fromarray(padded[:, start : start + 600])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    allcases = json.loads((BASE / "test-2000-rerun-01/cases.json").read_text())
    selected = []
    for label in sorted({c["label"] for c in allcases}):
        selected += sorted(
            [c for c in allcases if c["label"] == label],
            key=lambda c: hashlib.sha256((SEED + c["id"]).encode()).hexdigest(),
        )[:10]
    protocol = {
        "selection": "10 per class from test, lowest SHA256(seed+filename); independent of predictions",
        "seed": SEED,
        "images": 300,
        "conditions": CONDITIONS,
        "variant_rows": 2100,
        "image_policy": "All controls and transformed variants are 600x600 RGB lossless PNG; control preserves decoded RGB pixels; no resizing.",
        "transformations": {
            "blur": "Pillow GaussianBlur radius 1 or 2 pixels",
            "contrast": "Pillow ImageEnhance.Contrast factors 0.5 and 1.5",
            "shift": "60 pixels left/right with reflected fill; crops 60 original boundary pixels, so this is not pure object-position isolation",
        },
        "analysis": "Paired accuracy changes vs fresh control; McNemar tests; pairwise differences of change via paired randomization; Holm across 30 within-model and 60 between-model tests.",
        "notes": [
            "Equal-class balanced subset; accuracy is not directly comparable to the full test split.",
            "Ground truth held fixed; stronger transforms may remove discriminative evidence.",
            "Single stochastic response per model/image/condition; repeat on independent data to confirm.",
        ],
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    cases = []
    for c in selected:
        source = BASE / "test-2000-rerun-01/images" / c["image"]
        raw = source.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == c["sha256"]
        with Image.open(source) as original:
            original = original.convert("RGB")
            assert original.size == (600, 600)
            for condition in CONDITIONS:
                name = Path(c["id"]).stem + "__" + condition + ".png"
                path = OUT / "images" / name
                variant = transform(original, condition)
                variant.save(path, format="PNG")
                with Image.open(path) as check:
                    assert check.size == (600, 600)
                    if condition == "control":
                        assert np.array_equal(np.asarray(check), np.asarray(original))
                cases.append(
                    {
                        "id": name,
                        "image": name,
                        "label": c["label"],
                        "original_id": c["id"],
                        "condition": condition,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "original_sha256": c["sha256"],
                        "dimensions": [600, 600],
                    }
                )
    assert len(cases) == 2100
    # Randomize row order deterministically so condition/class do not track request time.
    cases.sort(key=lambda c: hashlib.sha256(("order:" + SEED + c["id"]).encode()).hexdigest())
    (OUT / "cases.json").write_text(json.dumps(cases, indent=2))
    keys = dict(
        l.split("=", 1)
        for l in args.env_file.read_text().splitlines()
        if "=" in l and not l.lstrip().startswith("#")
    )
    bt.login(api_key=keys["BRAINTRUST_API_KEY"].strip().strip('"').strip("'"))
    ds = bt.init_dataset(
        project="aid-single-label-comparison",
        name="aid-controlled-300-v1",
        metadata=protocol,
        description="Balanced 300 original test images x 7 controlled conditions; lossless PNG; paired robustness evaluation.",
    )
    refpath = OUT / "attachments.jsonl"
    refs = (
        {r["id"]: r["reference"] for r in map(json.loads, refpath.read_text().splitlines())}
        if refpath.exists()
        else {}
    )

    def upload(c):
        for attempt in range(5):
            try:
                a = bt.Attachment(
                    data=str(OUT / "images" / c["image"]),
                    filename=c["image"],
                    content_type="image/png",
                )
                assert a.upload()["upload_status"] == "done"
                return c["id"], a.reference
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)

    with refpath.open("a") as f, cf.ThreadPoolExecutor(max_workers=12) as pool:
        for fut in cf.as_completed([pool.submit(upload, c) for c in cases if c["id"] not in refs]):
            name, ref = fut.result()
            refs[name] = ref
            f.write(json.dumps({"id": name, "reference": ref}) + "\n")
            f.flush()
            if len(refs) % 100 == 0:
                print("uploaded", len(refs), "of", len(cases), flush=True)
    for c in cases:
        ds.insert(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, SEED + c["id"])),
            input={"image": refs[c["id"]]},
            expected=c["label"],
            metadata={
                "source_id": c["id"],
                "original_id": c["original_id"],
                "condition": c["condition"],
                "scene_class": c["label"],
                "sha256": c["sha256"],
                "original_sha256": c["original_sha256"],
                "split": "test-balanced-subset",
                "dimensions": [600, 600],
            },
            tags=[c["condition"], c["label"]],
        )
    ds.flush()
    fresh = bt.init_dataset(project="aid-single-label-comparison", name="aid-controlled-300-v1")
    remote = list(fresh.fetch())
    assert len(remote) == 2100
    version = fresh.version
    result = {
        "dataset_version": version,
        "dataset_url": fresh.summarize().dataset_url,
        "rows": len(remote),
    }
    (OUT / "braintrust.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
