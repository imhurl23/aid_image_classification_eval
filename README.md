# AID image classification eval

Compare OpenAI Decisions Luna and TypeSafe Jev on AID single-label aerial scenes, with Braintrust image review and specialist-baseline analysis.

- [AID runner and protocol](outputs/aid_eval/README.txt)
- [Pilot runner](outputs/aid_eval/eval.py)
- [Full test runner](outputs/aid_eval/full_test.py)
- [Full test results](outputs/aid_eval/test-2000/results.txt)
- [Specialist analysis](outputs/aid_eval/specialist/)
- [Earlier LADI work](outputs/ladi_eval/README.txt)
- [Dataset preparation](work/aid/prepare.py)

Run scripts from this repository root. Credentials are external and are not committed; some existing full-run scripts reference the sibling classifier_eval project's .env. Pilot scripts accept --env-file.

All original images, checkpoints, downloaded source repositories, and local environments have been moved here but are ignored by Git. Existing links at the original Codex workspace preserve old paths and virtual-environment entry points. This is a local repository; no remote has been configured.
