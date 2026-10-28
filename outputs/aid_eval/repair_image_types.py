"""Publish correctly typed attachments without changing source bytes or dataset row IDs."""

import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path

import braintrust as bt
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, type=Path)
    args = parser.parse_args()
    keys = dict(
        line.split("=", 1)
        for line in args.env_file.read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )
    bt.login(api_key=keys["BRAINTRUST_API_KEY"].strip().strip('"').strip("'"))
    base = Path(__file__).resolve().parent
    config = json.loads((base / "config.json").read_text())
    for name, source in [
        ("aid-validation-pilot-30", base),
        ("aid-test-2000", base / "test-2000-rerun-01"),
    ]:
        ds = bt.init_dataset(project=config["project"], name=name)
        rows = list(ds.fetch())
        before = ds.version

        def repair(row):
            path = source / "images" / row["metadata"]["source_id"]
            raw = path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == row["metadata"]["sha256"]
            with Image.open(io.BytesIO(raw)) as im:
                mime = Image.MIME[im.format]
            old = row["input"]["image"]
            ref = old.reference if hasattr(old, "reference") else old
            if ref["content_type"] == mime:
                return None
            attachment = bt.Attachment(data=raw, filename=path.stem + ".png", content_type=mime)
            assert attachment.upload()["upload_status"] == "done"
            ds.insert(
                id=row["id"],
                input={"image": attachment},
                expected=row["expected"],
                metadata=row["metadata"],
            )
            return row["id"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            changed = [x for x in pool.map(repair, rows) if x]
        ds.flush()
        print(
            json.dumps({"dataset": name, "prior_version": before, "corrected": len(changed)}),
            flush=True,
        )


if __name__ == "__main__":
    main()
