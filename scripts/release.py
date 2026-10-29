"""Check the public working tree and optionally archive it without Git history."""

import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATHS = (
    "outputs/aid_eval/team-travel-v1/",
    "outputs/aid_eval/team_travel_eval.py",
    "outputs/aid_eval/assign_model_reviews.py",
    "outputs/aid_eval/specialist/papers/",
    "outputs/aid_eval/specialist/author-request.txt",
    "outputs/aid_eval/web-aid-v1/before-label-once-review.json",
    "outputs/aid_eval/web-aid-v1/review-queue-verification.json",
    "outputs/aid_eval/aid-definitions-v1-shared-errors/",
)
PATTERNS = {
    "credential": re.compile(
        r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
        r"|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
        r"|[?&](?:X-Amz-Signature|X-Goog-Signature)=",
        re.IGNORECASE,
    ),
    "personal home path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
}


def candidate_files():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sorted(set(filter(None, result.stdout.decode().split("\0"))))


def problems(name, data, names):
    path = Path(name)
    errors = []
    if name.startswith(PRIVATE_PATHS) or (
        path.name.startswith(".env") and path.name != ".env.example"
    ):
        errors.append("private/local file")
    if path.suffix.lower() in {".pdf", ".pem", ".key", ".jpg", ".jpeg", ".pth", ".ckpt"}:
        errors.append("source image, downloaded paper, key, or checkpoint")
    if path.suffix.lower() == ".png" and not (
        name.startswith("outputs/aid_eval/controlled-definitions-v1/")
        and path.name.endswith("-deltas.png")
    ):
        errors.append("unapproved image binary")
    if len(data) > 50 * 1024 * 1024:
        errors.append("file exceeds 50 MiB")
    if path.suffix.lower() == ".png":
        return errors
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return errors + ["unexpected binary file"]
    for label, pattern in PATTERNS.items():
        if pattern.search(content):
            errors.append(label)
    try:
        if path.suffix == ".json":
            json.loads(content)
        elif path.suffix == ".jsonl":
            for line in content.splitlines():
                if line.strip():
                    json.loads(line)
    except ValueError:
        errors.append("invalid JSON/JSONL")
    if path.suffix == ".md":
        for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", content):
            if ":" in target or target.startswith("#"):
                continue
            target = target.split("#")[0]
            destination = (ROOT / path.parent / target).resolve()
            if (
                not destination.is_relative_to(ROOT)
                or str(destination.relative_to(ROOT)) not in names
            ):
                errors.append(f"local link outside release: {target}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Write a .tar.gz snapshot and SHA256 file")
    args = parser.parse_args()
    names = candidate_files()
    contents = {}
    errors = []
    for name in names:
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            errors.append(f"{name}: missing file or symlink")
            continue
        data = path.read_bytes()
        errors.extend(f"{name}: {problem}" for problem in problems(name, data, names))
        contents[name] = data
    if errors:
        raise SystemExit("Release check failed:\n" + "\n".join(errors))
    print(f"Release checks passed: {len(contents)} files")
    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.archive, "w:gz") as archive:
            for name, data in contents.items():
                info = tarfile.TarInfo("aid-image-classification-eval/" + name)
                info.size = len(data)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(data))
        digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
        args.archive.with_suffix(args.archive.suffix + ".sha256").write_text(
            f"{digest}  {args.archive.name}\n"
        )
        print(f"Created {args.archive} (no Git history or ignored files)")


if __name__ == "__main__":
    main()
