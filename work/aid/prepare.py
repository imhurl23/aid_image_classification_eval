import concurrent.futures
import hashlib
import io
import json
from pathlib import Path

import requests
from PIL import Image

OUT = Path("outputs/aid_eval")
(OUT / "images").mkdir(exist_ok=True)
rev = "d9b5531544523f29d365a77a007b2ace3cea6882"
splitrev = "e6767964e40f567a18eac6e6ca5fce397e4ce411"


def get(url):
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    return r


info = get("https://huggingface.co/api/datasets/blanchon/AID/revision/" + rev).json()
paths = {
    Path(x["rfilename"]).name: x["rfilename"]
    for x in info["siblings"]
    if x["rfilename"].endswith(".jpg")
}
splits = {
    s: get(
        f"https://huggingface.co/datasets/isaaccorley/aid/resolve/{splitrev}/aid-{s}.txt"
    ).text.splitlines()
    for s in ["train", "val", "test"]
}
assert (
    not set(splits["train"]) & set(splits["test"])
    and not set(splits["val"]) & set(splits["test"])
    and not set(splits["train"]) & set(splits["val"])
)
(OUT / "split-manifest.json").write_text(json.dumps(splits, indent=2))
selected = {}
for name in sorted(
    splits["val"], key=lambda x: hashlib.sha256(("aid-pilot-2026:" + x).encode()).hexdigest()
):
    path = paths[name]
    label = path.split("/")[1]
    selected.setdefault(label, path)
assert len(selected) == 30
labels = sorted(selected)


def fetch(item):
    label, path = item
    raw = get(f"https://huggingface.co/datasets/blanchon/AID/resolve/{rev}/{path}").content
    im = Image.open(io.BytesIO(raw))
    assert im.size == (600, 600)
    filename = Path(path).name
    (OUT / "images" / filename).write_bytes(raw)
    return {
        "id": filename,
        "label": label,
        "image": filename,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_path": path,
        "dimensions": [600, 600],
    }


with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    rows = list(pool.map(fetch, sorted(selected.items())))
(OUT / "cases.json").write_text(json.dumps(rows, indent=2))
(OUT / "config.json").write_text(
    json.dumps(
        {
            "project": "aid-single-label-comparison",
            "dataset": "blanchon/AID",
            "revision": rev,
            "split_source": "isaaccorley/aid",
            "split_revision": splitrev,
            "split": "val",
            "split_is_official": False,
            "labels": labels,
            "pilot_images": 30,
            "selection": "minimum sha256(aid-pilot-2026:filename) per class in validation",
            "image_policy": "same original 600x600 JPEG bytes; no client resizing",
            "baseline_status": "pending checkpoint training-ID verification",
        },
        indent=2,
    )
)
print("Prepared", len(rows), "images", labels)
