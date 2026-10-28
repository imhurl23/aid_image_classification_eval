"""Exploratory image-property associations, adjusted for class and other properties."""

import concurrent.futures
import itertools
import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from analyze_subgroups import holm
from PIL import Image
from scipy.ndimage import laplace

BASE = Path(__file__).resolve().parent
MODELS = ["luna", "jev", "clef", "d1", "d1-ultra"]
FEATURES = [
    "brightness",
    "contrast",
    "sharpness",
    "entropy",
    "edge_density",
    "detail_offset",
    "central_detail",
    "clipped_fraction",
]


def features(case):
    with Image.open(BASE / "test-2000-rerun-01/images" / case["image"]) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=float) / 255
    gray = rgb @ np.array([0.2126, 0.7152, 0.0722])
    gy, gx = np.gradient(gray)
    edge = np.hypot(gx, gy)
    h, w = gray.shape
    y, x = np.mgrid[:h, :w]
    weight = edge.sum()
    cx = (edge * x).sum() / weight / (w - 1) if weight else 0.5
    cy = (edge * y).sum() / weight / (h - 1) if weight else 0.5
    hist = np.histogram(gray, bins=256, range=(0, 1))[0] / gray.size
    vals = {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(np.log1p(laplace(gray).var() * 255**2)),
        "entropy": float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum()),
        "edge_density": float((edge > 0.05).mean()),
        "detail_offset": float(np.hypot(cx - 0.5, cy - 0.5) / np.sqrt(0.5)),
        "central_detail": float(edge[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].sum() / weight)
        if weight
        else 0.25,
        "clipped_fraction": float(((gray <= 0.02) | (gray >= 0.98)).mean()),
    }
    return {"id": case["id"], "class": case["label"], **vals}


def main():
    cases = json.loads((BASE / "test-2000-rerun-01/cases.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(features, cases))
    (BASE / "native-fresh-01-image-features.json").write_text(json.dumps(rows, indent=2))
    raw = np.array([[r[f] for f in FEATURES] for r in rows])
    mean = raw.mean(axis=0)
    sd = raw.std(axis=0)
    z = (raw - mean) / sd
    labels = sorted({c["label"] for c in cases})
    dummy = np.array([[int(c["label"] == label) for label in labels] for c in cases])
    X = np.column_stack([dummy, z])
    assert np.linalg.matrix_rank(X) == X.shape[1]
    correct = {}
    for m in MODELS:
        records = [
            json.loads(x)
            for x in (BASE / f"native-fresh-01-{m}" / f"{m}.jsonl").read_text().splitlines()
        ]
        r = {x["id"]: x for x in records}
        assert len(r) == len(cases) == len(records)
        correct[m] = np.array(
            [int(r[c["id"]]["status"] == "ok" and r[c["id"]]["label"] == c["label"]) for c in cases]
        )
        assert all(r[c["id"]]["input_sha256"] == c["sha256"] for c in cases)
    tests = []
    targets = [(m, correct[m], "accuracy") for m in MODELS] + [
        (a + "_minus_" + b, correct[a] - correct[b], "paired_difference")
        for a, b in itertools.combinations(MODELS, 2)
    ]
    for name, y, kind in targets:
        fit = sm.OLS(y, X).fit(cov_type="HC3")
        ci = fit.conf_int()
        for i, f in enumerate(FEATURES):
            j = len(labels) + i
            tests.append(
                {
                    "target": name,
                    "kind": kind,
                    "feature": f,
                    "effect_pp_per_sd": float(fit.params[j] * 100),
                    "unadjusted_95pct_ci_pp": (ci[j] * 100).tolist(),
                    "p": float(fit.pvalues[j]),
                }
            )
    holm(tests)
    quantiles = {}
    for i, f in enumerate(FEATURES):
        cut = np.quantile(raw[:, i], [0.25, 0.75])
        masks = [raw[:, i] <= cut[0], raw[:, i] >= cut[1]]
        quantiles[f] = {
            "cutoffs": cut.tolist(),
            "n": [int(x.sum()) for x in masks],
            "unadjusted_accuracy": {
                m: [float(correct[m][mask].mean()) for mask in masks] for m in MODELS
            },
        }
    # VIF after removing class means: diagnose overlap among measured properties.
    within = z - dummy @ np.linalg.lstsq(dummy, z, rcond=None)[0]
    vif = {}
    for i, f in enumerate(FEATURES):
        y = within[:, i]
        pred = (
            np.delete(within, i, axis=1)
            @ np.linalg.lstsq(np.delete(within, i, axis=1), y, rcond=None)[0]
        )
        vif[f] = float((y @ y) / ((y - pred) @ (y - pred)))
    result = {
        "images": len(cases),
        "features": FEATURES,
        "feature_means": dict(zip(FEATURES, mean.tolist())),
        "feature_sd": dict(zip(FEATURES, sd.tolist())),
        "method": "Linear probability regressions with 30 scene-class fixed effects and all eight standardized image properties; HC3 robust SE. Paired model differences regress correctness(A)-correctness(B) on the same design. Holm correction across all 120 coefficient tests.",
        "tests": tests,
        "significant_tests": [t for t in tests if t["significant"]],
        "within_class_vif": vif,
        "descriptive_quartiles": quantiles,
        "caveats": [
            "Exploratory associations, not causal effects or interventions.",
            "Detail centroid and central edge share are proxies, not detected object positions.",
            "Sharpness/entropy can reflect scene texture, not only acquisition quality.",
            "Only the same 2,000 test images; no independent validation.",
            "HC3 assumes independence between images; unknown geographic correlation is not modeled.",
            "Confidence intervals shown are unadjusted; significance uses Holm-adjusted p values.",
        ],
    }
    (BASE / "native-fresh-01-image-properties.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Image-property analysis",
        "",
        result["method"],
        "",
        "Each effect is an accuracy change in percentage points per one dataset-wide standard deviation, holding class and the other measured properties constant.",
        "",
        "| Target | Property | Effect (pp/SD) | Adjusted p |",
        "|---|---|---:|---:|",
    ]
    for t in sorted(result["significant_tests"], key=lambda t: t["p_holm"]):
        lines.append(
            f"| {t['target']} | {t['feature']} | {t['effect_pp_per_sd']:+.2f} | {t['p_holm']:.4g} |"
        )
    lines += [
        "",
        "## Feature definitions",
        "",
        "- Brightness: mean Rec.709 luminance, 0–1.",
        "- Contrast: standard deviation of luminance.",
        "- Sharpness: log(1 + variance of luminance Laplacian × 255²).",
        "- Entropy: 256-bin luminance Shannon entropy in bits.",
        "- Edge density: fraction with luminance gradient magnitude above 0.05.",
        "- Detail offset: distance of edge-weighted centroid from image center, normalized by corner distance.",
        "- Central detail: fraction of total gradient magnitude in the central 50% × 50% of the frame.",
        "- Clipped fraction: luminance pixels at or below 0.02 or at or above 0.98.",
        "",
        "## Limitations",
        "",
    ] + ["- " + c for c in result["caveats"]]
    (BASE / "native-fresh-01-image-properties.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"significant_tests": result["significant_tests"], "vif": vif}, indent=2))


if __name__ == "__main__":
    main()
