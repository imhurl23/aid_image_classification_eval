import copy
import hashlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import native_eval as native
import numpy as np
from analyze_robustness import bootstrap_indices, ece, validate_manifest
from PIL import Image


class RobustnessProtocolTests(unittest.TestCase):
    def test_definitions_reach_every_provider_without_changing_other_payload_fields(self):
        prompt = (Path(__file__).parent / "controlled-definitions-v1/prompt.txt").read_text()
        keys = {
            k: "test"
            for k in (
                "OPENAI_API_KEY",
                "TYPESAFE_API_KEY",
                "CLOUDFLARE_ACCOUNT_ID",
                "CLOUDFLARE_AUTH_TOKEN",
                "LIQUID_AI_API_KEY",
            )
        }
        for fmt in ("PNG", "JPEG"):
            image = io.BytesIO()
            Image.new("RGB", (600, 600)).save(image, format=fmt)
            for model in native.MODELS:
                with (
                    self.subTest(model=model, format=fmt),
                    patch("uuid.uuid4", return_value="fixed"),
                ):
                    before = native.request_payload(model, image.getvalue(), keys)
                    after = native.request_payload(
                        model, image.getvalue(), keys, instructions=prompt
                    )
                    self.assertEqual(before[:2], after[:2])
                    question = (
                        after[2]["questions"][0]
                        if model == "luna"
                        else after[2]["questions"]["scene"]
                    )
                    self.assertEqual(question["instructions"], prompt)
                    question["instructions"] = native.protocol.INSTRUCTIONS
                    self.assertEqual(before[2], after[2])

    def test_mixed_prompts_and_dataset_versions_are_rejected(self):
        manifest = {
            "instructions": "definitions",
            "labels": ["A"],
            "dataset": "controlled",
            "dataset_version": "1",
            "trials": 1,
            "transform_protocol": {"prompt_sha256": hashlib.sha256(b"definitions").hexdigest()},
        }
        validate_manifest(manifest, manifest)
        for field, value in [
            ("instructions", "old prompt"),
            ("dataset_version", "2"),
            ("trials", 2),
        ]:
            other = copy.deepcopy(manifest)
            other[field] = value
            with self.assertRaises(ValueError):
                validate_manifest(other, manifest)
        other = copy.deepcopy(manifest)
        other["instructions"] = "wrong definitions"
        with self.assertRaises(ValueError):
            validate_manifest(other, other)

    def test_bootstrap_preserves_class_counts_and_pairing(self):
        ids = ["a", "b", "c", "d"]
        labels = dict(zip(ids, ["A", "A", "B", "B"]))
        indices = bootstrap_indices(ids, labels, reps=100)
        self.assertTrue(np.all((indices < 2).sum(axis=1) == 2))
        values = np.array([1, 0, 1, 0])
        self.assertTrue(np.all((values - values)[indices].mean(axis=1) == 0))

    def test_calibration_excludes_failed_predictions(self):
        confidence = np.array([0.9, 0.1, 0.0])
        correct = np.array([1.0, 0.0, 0.0])
        valid = np.array([True, True, False])
        self.assertAlmostEqual(ece(confidence, correct, valid)[0], 0.1)
        self.assertAlmostEqual(ece(confidence, correct, valid, np.array([[0, 1, 2]]))[0], 0.1)


if __name__ == "__main__":
    unittest.main()
