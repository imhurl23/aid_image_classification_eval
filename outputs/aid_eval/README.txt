AID single-label comparison

Project: aid-single-label-comparison
Task: choose exactly one of 30 AID aerial-scene labels per image.
Models: gpt-6-luna via Decisions Choice; jev-1.14.0 via TypeSafe Choice with image header.
Inputs: identical original 600x600 JPEG bytes. No client resizing, crops, or label-bearing filenames in requests.

Pilot: one deterministic validation image per class, 30 images / 60 API requests.
This is a balanced integration and rubric-review pilot, NOT a reliable estimate of final performance.
Image files, expected labels, probabilities, predictions, latencies and exact served versions are logged to Braintrust.

Sources are pinned in config.json. Image mirror: blanchon/AID. Split manifests:
isaaccorley/aid, deterministic per-class 60/20/20. These are community splits,
not official AID splits. All 10,000 filenames are saved in split-manifest.json;
train/validation/test disjointness checked. No test images were used in this pilot.
The original AID paper uses random train/test splits and repeated runs, so paper
numbers must not be presented as directly comparable scores on this pilot.

Scoring: single-label accuracy, macro F1, 30x30 confusion matrix, multiclass log loss,
valid-output coverage. API latency recorded per image. Probabilities may have rounded
sums: allow total mass within 0.02 of 1 and normalize; preserve raw response and mass.
Malformed choices or incomplete/nonfinite distributions fail validation. The pilot's
two initial rounding flags were corrected offline without rerunning model calls.

Specialist baseline remains pending. The downloadable JavideuS/aid-image-classification
checkpoint card reports a 90/10 split; its repository has weights but no training/test
filename manifest or model-definition code. It cannot yet be assumed held out on
our split. Do not compare its published 92.8% with these pilot scores. A valid next
step is to recover its exact test manifest or train a baseline on our frozen training
partition. No official ready-to-run checkpoint has been established for this split.

Run or resume:
python eval.py --env-file /path/to/credentials.env
Requires requests, braintrust, Pillow. Existing prediction rows are reused; publishing
uses stable row IDs and experiment names. Do not edit instructions or labels and reuse
cached predictions; start a separately versioned run for changed configurations.

Review pilot confusions before freezing a larger test run. The original LADI Braintrust
project is preserved separately and is not the AID project.

Sources:
https://captain-whu.github.io/AID/
https://huggingface.co/datasets/blanchon/AID
https://huggingface.co/datasets/isaaccorley/aid
https://huggingface.co/JavideuS/aid-image-classification
https://developers.openai.com/api/docs/guides/decisions
https://docs.typesafe.ai/images

Full test run completed: 2,000 test images, 4,000 model outputs, both published to Braintrust.
Full results, image checks, frozen protocol and experiment links are in test-2000/.
Luna: 74.30% accuracy; Jev: 79.40%; estimated combined API cost $0.2885.
Full-run entry point: python full_test.py (resumes saved outputs and attachment references).

2026-10-19 baseline update: original source recovered at JavideuS/aerial-vision;
checkpoint strictly loaded and smoke-tested on 30 images. Original held-out
filename manifest is still unavailable. See specialist/findings.txt and adapter.py.
