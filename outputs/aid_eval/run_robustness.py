"""Verify the definitions-matched robustness protocol; --execute runs fresh inference."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import native_eval as native
from credentials import add_credential_arguments, keys_load, validate_keys

BASE = Path(__file__).resolve().parent
ROOT = BASE / "controlled-definitions-v1"
IMAGES = BASE / "controlled-300-v1/images"


def preflight():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    prompt = (ROOT / "prompt.txt").read_text()
    if hashlib.sha256(prompt.encode()).hexdigest() != protocol["prompt_sha256"]:
        raise ValueError("Frozen definitions prompt changed")
    if (
        hashlib.sha256((ROOT / "cases.json").read_bytes()).hexdigest()
        != protocol["case_manifest_sha256"]
    ):
        raise ValueError("Frozen case manifest changed")
    cases = json.loads((ROOT / "cases.json").read_text())
    if len(cases) != len({c["id"] for c in cases}) or len(cases) != 2100:
        raise ValueError("Expected exactly 2,100 unique cases")
    groups = {}
    for case in cases:
        raw = (IMAGES / case["image"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != case["sha256"]:
            raise ValueError("Image differs from frozen manifest: " + case["id"])
        groups.setdefault(case["original_id"], set()).add(case["condition"])
    if len(groups) != 300 or any(c != set(protocol["conditions"]) for c in groups.values()):
        raise ValueError("Each of 300 sources must have a control and all six transformations")
    dummy = {
        name: "offline"
        for name in (
            "OPENAI_API_KEY",
            "TYPESAFE_API_KEY",
            "CLOUDFLARE_ACCOUNT_ID",
            "CLOUDFLARE_AUTH_TOKEN",
            "LIQUID_AI_API_KEY",
        )
    }
    for model in protocol["models"]:
        _, _, body = native.request_payload(model, raw, dummy, instructions=prompt)
        question = body["questions"][0] if model == "luna" else body["questions"]["scene"]
        if question["instructions"] != prompt:
            raise ValueError("Definitions missing from " + model)
    report = {
        "images_verified": len(cases),
        "sources": len(groups),
        "prompt_sha256": protocol["prompt_sha256"],
        "provider_payloads_verified": protocol["models"],
        "api_calls": 0,
    }
    (ROOT / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--models", nargs="+", choices=list(native.MODELS), default=list(native.MODELS)
    )
    add_credential_arguments(parser)
    args = parser.parse_args()
    protocol = preflight()
    if not args.execute:
        return
    if len(args.models) != len(set(args.models)):
        parser.error("Duplicate models")
    validate_keys(keys_load(args.env_file, args.liquid_key_file), args.models)
    for model in args.models:
        folder = BASE / (protocol["experiment"] + "-" + model)
        if (folder / "native-manifest.json").exists() != args.resume:
            parser.error("Run state conflicts with --resume: " + model)
        if not args.resume and folder.exists() and any(folder.iterdir()):
            parser.error("Run directory is not empty: " + model)
    children = []
    for model in args.models:
        command = [
            sys.executable,
            str(BASE / "native_eval.py"),
            "--models",
            model,
            "--run-name",
            protocol["experiment"] + "-" + model,
            "--dataset",
            protocol["dataset"],
            "--dataset-version",
            protocol["dataset_version"],
            "--cases-file",
            str(ROOT / "cases.json"),
            "--protocol-file",
            str(ROOT / "protocol.json"),
            "--instructions-file",
            str(ROOT / "prompt.txt"),
            "--images-dir",
            str(IMAGES),
            "--max-concurrency",
            str(protocol["max_concurrency_per_model"]),
            "--base-experiment",
            "",
        ]
        for flag, path in (
            ("--env-file", args.env_file),
            ("--liquid-key-file", args.liquid_key_file),
        ):
            if path:
                command.extend([flag, str(path.resolve())])
        if args.resume:
            command.append("--resume")
        with (ROOT / f"{model}.log").open("a" if args.resume else "w") as log:
            children.append(
                (model, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
            )
    codes = {model: child.wait() for model, child in children}
    (ROOT / "run-status.json").write_text(json.dumps({"exit_codes": codes}, indent=2) + "\n")
    print(json.dumps({"exit_codes": codes}), flush=True)
    raise SystemExit(int(any(codes.values())))


if __name__ == "__main__":
    main()
