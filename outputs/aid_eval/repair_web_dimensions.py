"""Apply shared minimum-side padding and invalidate all five predictions for affected images."""

import hashlib
import json
import uuid
from pathlib import Path

import braintrust as bt
import web_aid_eval as w
from PIL import Image


def main():
    p = w.OUT
    if (p / "padding-repair.json").exists():
        raise RuntimeError("Padding already prepared; resume web_aid_eval.py instead")
    cases = json.loads((p / "cases.json").read_text())
    affected = [c for c in cases if min(c["dimensions"]) < 280]
    done = list(map(json.loads, (p / "published.jsonl").read_text().splitlines()))
    assert len(done) == len(cases) == 1000, "Wait for the original run to finish"
    keys = w.keys_load()
    bt.login(api_key=keys["BRAINTRUST_API_KEY"])
    logger = bt.init_logger(project=w.PROJECT)
    ds = bt.init_dataset(project=w.PROJECT, name=w.BATCH)
    (p / "cases-before-padding.json").write_text(json.dumps(cases, indent=2))
    ids = {c["id"] for c in affected}
    mapping = []
    newrefs = []
    for c in affected:
        oldsha = c["sha256"]
        oldimage = c["image"]
        im = Image.open(p / "images" / oldimage).convert("RGB")
        size = (max(280, im.width), max(280, im.height))
        canvas = Image.new("RGB", size, (127, 127, 127))
        offset = ((size[0] - im.width) // 2, (size[1] - im.height) // 2)
        canvas.paste(im, offset)
        target = p / "images" / (c["id"] + "-padded.png")
        canvas.save(target, format="PNG")
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        c.update(
            image=target.name,
            sha256=sha,
            dimensions=list(size),
            pre_padding_sha256=oldsha,
            pre_padding_image=oldimage,
            padding={"fill": [127, 127, 127], "offset": list(offset), "minimum_side": 280},
        )
        a = bt.Attachment(
            data=str(target), filename=c["id"][:16] + "-padded.png", content_type="image/png"
        )
        assert a.upload()["upload_status"] == "done"
        newrefs.append({"id": c["id"], "reference": a.reference})
        rid = str(uuid.uuid5(uuid.NAMESPACE_URL, w.BATCH + "/review/" + c["id"]))
        # Update image and provenance only; do not overwrite any human's expected category.
        logger.update_span(
            id=rid,
            input={"image": a.reference},
            metadata={
                "sha256": sha,
                "dimensions": list(size),
                "image": target.name,
                "padding": c["padding"],
                "pre_padding_sha256": oldsha,
            },
        )
        ds.update(
            id=c["id"],
            input={"image": a.reference},
            metadata={
                "sha256": sha,
                "dimensions": list(size),
                "image": target.name,
                "padding": c["padding"],
                "pre_padding_sha256": oldsha,
            },
        )
        mapping.append(
            {"id": c["id"], "old_sha256": oldsha, "new_sha256": sha, "new_dimensions": list(size)}
        )
    ds.flush()
    bt.flush()
    preds = list(map(json.loads, (p / "predictions.jsonl").read_text().splitlines()))
    old = [r for r in preds if r["id"] in ids]
    assert len(old) == len(affected) * 5
    (p / "pre-padding-predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in old))
    (p / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in preds if r["id"] not in ids)
    )
    (p / "published.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in done if r["id"] not in ids)
    )
    refs = list(map(json.loads, (p / "attachments.jsonl").read_text().splitlines()))
    (p / "attachments.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in refs if r["id"] not in ids)
        + "".join(json.dumps(r) + "\n" for r in newrefs)
    )
    (p / "cases.json").write_text(json.dumps(cases, indent=2))
    plan = json.loads((p / "protocol.json").read_text())
    plan["image_policy"] += (
        " Images with a side below 280 are centered on neutral-gray padding to minimum 280 per side, without further resizing; all five models rerun on the same padded bytes."
    )
    (p / "protocol.json").write_text(json.dumps(plan, indent=2))
    (p / "padding-repair.json").write_text(json.dumps(mapping, indent=2))
    print("Prepared shared padding for", len(mapping), "images")


if __name__ == "__main__":
    main()
