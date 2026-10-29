# Public release

The working tree contains the public source and benchmark audit exports. Personal
gallery files, reviewer snapshots, downloaded PDFs, and local scratch are ignored
and retained locally. The four original research commits have been rewritten to
remove these files, legacy reviewer IDs, and personal home paths. Recovery backups
are kept outside this repository and must remain private.

Commit the reviewed release changes before publishing the cleaned branch. If a
remote still has the original history, updating it requires a coordinated force
push; do not merge the original commits back into the cleaned branch.

Alternatively, build a snapshot without Git metadata:

```sh
python -m unittest discover -s outputs/aid_eval -p 'test_*.py'
ruff check .
ruff format --check .
python scripts/release.py --archive dist/aid-image-classification-eval.tar.gz
```

The archive includes tracked working-tree files plus nonignored new files, after
checking for private file paths, credential patterns, personal home paths, broken
local Markdown links, and invalid JSON/JSONL. It excludes Git history and ignored
local files. A SHA256 checksum is written alongside it. Review `git status` before
building so unrelated new files are not included. These checks are guardrails,
not a comprehensive secret-detection or rights audit.

Extract the archive into a new directory and initialize a new Git repository there
for publication. Keep pre-cleanup backups private. The release script
does not publish, push, rewrite history, or delete local source material.

The code is MIT-licensed. Third-party dataset and model terms are described in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Saved metrics retain their original
protocols and limitations. Historical Braintrust links may require workspace access.
