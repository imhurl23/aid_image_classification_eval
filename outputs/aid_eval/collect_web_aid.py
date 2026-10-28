"""Collect attributed web aerial-image candidates, without inventing ground truth."""

import concurrent.futures as cf
import hashlib
import html
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parent
OUT = BASE / "web-aid-v1"
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": "AIDWebEvaluation/1.0 (research classification; attribution retained; https://github.com/imhurl23/aid_image_classification_eval)"
}
QUERIES = {
    "Airport": "airport",
    "BareLand": "bare land",
    "BaseballField": "baseball field",
    "Beach": "beach",
    "Bridge": "bridge",
    "Center": "city center",
    "Church": "church",
    "Commercial": "shopping commercial",
    "DenseResidential": "dense residential city",
    "Desert": "desert",
    "Farmland": "farmland",
    "Forest": "forest",
    "Industrial": "industrial factory",
    "Meadow": "meadow grassland",
    "MediumResidential": "residential neighborhood",
    "Mountain": "mountain",
    "Park": "park",
    "Parking": "parking lot",
    "Playground": "playground",
    "Pond": "pond",
    "Port": "port harbor",
    "RailwayStation": "railway station",
    "Resort": "resort",
    "River": "river",
    "School": "school campus",
    "SparseResidential": "rural residential village",
    "Square": "town square",
    "Stadium": "stadium",
    "StorageTanks": "storage tanks",
    "Viaduct": "viaduct",
}


def fetch(url, params=None):
    for i in range(5):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=45)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(30, 2 ** (i + 1)))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == 4:
                raise
            time.sleep(2**i)
    raise RuntimeError("HTTP retry exhausted")


def clean(s):
    return html.unescape(re.sub("<[^>]+>", "", s or "")).strip()


def search(term, offset=0):
    d = fetch(
        API,
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": term + " filetype:bitmap",
            "gsrnamespace": 6,
            "gsrlimit": 50,
            "gsroffset": offset,
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": 600,
        },
    ).json()
    return sorted(d.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))


def download(p):
    try:
        info = p.get("imageinfo", [{}])[0]
        ext = info.get("extmetadata", {})
        if min(info.get("width", 0), info.get("height", 0)) < 300:
            return None
        if max(info["width"], info["height"]) / min(info["width"], info["height"]) > 4:
            return None
        if not info.get("url", "").split("?", 1)[0].lower().endswith((".jpg", ".jpeg", ".png")):
            return None
        license = clean(ext.get("LicenseShortName", {}).get("value", ""))
        if not license:
            return None
        url = info.get("thumburl", info["url"])
        raw = fetch(url).content
        with Image.open(io.BytesIO(raw)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((600, 600), Image.Resampling.LANCZOS)
            if min(im.size) < 150:
                return None
            padding = None
            if min(im.size) < 280:
                size = (max(280, im.width), max(280, im.height))
                offset = ((size[0] - im.width) // 2, (size[1] - im.height) // 2)
                canvas = Image.new("RGB", size, (127, 127, 127))
                canvas.paste(im, offset)
                im = canvas
                padding = {"fill": [127, 127, 127], "offset": list(offset), "minimum_side": 280}
            arr = np.asarray(im.resize((9, 8)).convert("L"))
            bits = (arr[:, 1:] > arr[:, :-1]).ravel()
            dh = sum(int(x) << i for i, x in enumerate(bits))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            data = buf.getvalue()
            sha = hashlib.sha256(data).hexdigest()
            filename = sha + ".png"
            (OUT / "images" / filename).write_bytes(data)
            return {
                "id": sha,
                "image": filename,
                "sha256": sha,
                "dhash": dh,
                "dimensions": list(im.size),
                "source_dimensions": [info["width"], info["height"]],
                "source_title": p["title"],
                "source_page": info["descriptionurl"],
                "source_url": info["url"],
                "download_url": url,
                "license": license,
                "license_url": ext.get("LicenseUrl", {}).get("value"),
                "author": clean(ext.get("Artist", {}).get("value")),
                "source_description": clean(ext.get("ImageDescription", {}).get("value"))[:2000],
                "commons_page_id": p["pageid"],
                "review_status": "unreviewed",
                "expected": None,
                "padding": padding,
            }
    except Exception as e:
        return {"download_error": type(e).__name__, "title": p.get("title")}


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    path = OUT / "candidates.jsonl"
    cases = list(map(json.loads, path.read_text().splitlines())) if path.exists() else []
    ids = {c["id"] for c in cases}
    pages = {c["commons_page_id"] for c in cases}
    hashes = [c["dhash"] for c in cases]
    plan = {
        "target_images": 1000,
        "source": "Wikimedia Commons public image search",
        "categories_are_search_targets_not_labels": True,
        "image_policy": "Download Commons width-600 thumbnail where available; preserve aspect ratio, orient from EXIF, RGB lossless PNG, longest side <=600, no upscaling/cropping; identical bytes to all five models. Original source URL and dimensions retained. Neutral-gray padding ensures each side is at least 280 pixels; no stretching.",
        "duplicates": "Reject exact PNG SHA256 matches, repeated Commons page IDs, and dHash Hamming distance <=2.",
        "human_review": "True category is blank. Review once per image; permit Ambiguous, Other, NotAerial, Unusable. No accuracy until human labels exist.",
    }
    (OUT / "protocol.json").write_text(json.dumps(plan, indent=2))
    errors = 0
    with path.open("a") as f, cf.ThreadPoolExecutor(max_workers=4) as pool:
        for n, (label, term) in enumerate(QUERIES.items()):
            target = 34 if n < 10 else 33
            have = sum(c["search_category"] == label for c in cases)
            for variant in [
                "aerial " + term,
                "drone " + term,
                "satellite " + term,
                "aerial " + term.split()[0],
            ]:
                if have >= target:
                    break
                try:
                    candidates = search(variant)
                except Exception as e:
                    print(
                        json.dumps({"search_error": type(e).__name__, "query": variant}), flush=True
                    )
                    continue
                candidates = [p for p in candidates if p["pageid"] not in pages]
                for start in range(0, len(candidates), 4):
                    if have >= target:
                        break
                    for c in pool.map(download, candidates[start : start + 4]):
                        if not c or "download_error" in c:
                            errors += 1
                            continue
                        if have >= target:
                            continue
                        if c["id"] in ids or any((c["dhash"] ^ h).bit_count() <= 2 for h in hashes):
                            continue
                        c.update(search_category=label, search_query=variant)
                        cases.append(c)
                        ids.add(c["id"])
                        pages.add(c["commons_page_id"])
                        hashes.append(c["dhash"])
                        have += 1
                        f.write(json.dumps(c) + "\n")
                        f.flush()
            print(
                json.dumps(
                    {
                        "category": label,
                        "collected": have,
                        "target": target,
                        "total": len(cases),
                        "skipped_or_failed": errors,
                    }
                ),
                flush=True,
            )
    (OUT / "cases.json").write_text(json.dumps(cases, indent=2))
    print("COLLECTION_COMPLETE", len(cases), flush=True)


if __name__ == "__main__":
    main()
