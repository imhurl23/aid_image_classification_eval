LADI v2a decision comparison

Status: evaluation implementation with a bounded validation integration pilot.
The official test split has not been evaluated. Five pilot images cannot rank models.

Arms: OpenAI Decisions gpt-6-luna; TypeSafe jev-1.14.0 with image header;
MITLL/LADI-v2-classifier-small-reference and large-reference.
Deployment checkpoints are excluded from the primary comparison because they trained on all splits.

Image conditions:
- native (primary): supplied LADI image resolution to Luna and MITLL; Jev resizes
  with aspect ratio preserved to fit 1568 pixels. MITLL's own image processor still applies.
- matched (secondary): all arms receive the same resized pixels before native preprocessing.
Both rotate EXIF orientation into pixels, remove metadata and encode lossless PNG.
Input hashes and dimensions are recorded. This distinguishes practical system
performance from a comparison controlling the supplied resolution. It does not
force identical internal tokenization/crops across architectures.

Dependencies: numpy requests Pillow datasets pyarrow huggingface_hub transformers torch.
API keys: OPENAI_API_KEY and TYPESAFE_API_KEY in environment or explicit --env-file.
Never write keys into results.

Download the pinned source with hf download, e.g.:
hf download MITLL/LADI-v2-dataset --repo-type dataset --revision 5f2dbfe8c466d32edafd1bab847ec5252309acdb --include 'data/validation/*' --local-dir SOURCE

Run from this directory:
python run.py --source SOURCE --split validation --limit 5 --image-condition native --out pilot-native --arms decisions_luna jev_multimodal mitll_small_reference mitll_large_reference
Repeat with --image-condition matched and a new output directory for the controlled comparison.
All four arms must use identical source revision, split and row IDs. For partial
source downloads only contiguous shards from zero are accepted. Output directories
must be empty to prevent accidental overwrites. Each prediction is flushed immediately.

The current rubric is operational/provisional. Review damage definitions and pilot
errors against the dataset annotation guidance using validation only, then freeze
rubric, image policy, checkpoints and thresholds before test inference. No tuning
on test labels. API arms use independent predicate/Noul questions, one batch per image.
Local reference classifiers use sigmoid probabilities over all twelve named labels.

Scoring: fixed 0.5 threshold, macro/micro F1, per-label precision/recall/F1/support,
Brier score, exact match, coverage, and observed latency p50/p95. Undefined label
metrics are zero with support reported. Failures are retained and never become negatives.
Quality metrics condition on complete valid images; compare coverage and use a common
valid-ID intersection for paired model comparisons when failures differ. Latency
includes network/retries for APIs and preprocessing/inference for local models, excluding
model load; it is not a hardware-normalized speed comparison. Usage is retained, but
cost calculation, paired bootstrap intervals and statistical ranking are not implemented.

Sources:
https://huggingface.co/datasets/MITLL/LADI-v2-dataset
https://github.com/LADI-Dataset/ladi-overview
https://typesafe.ai/blog/jev-1-14
https://docs.typesafe.ai/images
https://docs.typesafe.ai/models
https://developers.openai.com/api/docs/guides/decisions
