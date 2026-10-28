import io
import unittest
from unittest.mock import Mock, patch

import native_eval as n
from PIL import Image


class NativeEvalTests(unittest.TestCase):
    def image(self, fmt):
        b = io.BytesIO()
        Image.new("RGB", (600, 600)).save(b, format=fmt)
        return b.getvalue()

    def test_image_mime_and_frozen_options(self):
        keys = {
            "OPENAI_API_KEY": "test",
            "TYPESAFE_API_KEY": "test",
            "CLOUDFLARE_ACCOUNT_ID": "test",
            "CLOUDFLARE_API_KEY": "test",
            "LIQUID_AI_API_KEY": "test",
        }
        for fmt, mime in [("PNG", "image/png"), ("JPEG", "image/jpeg")]:
            raw = self.image(fmt)
            for arm in n.MODELS:
                url, headers, body = n.request_payload(arm, raw, keys)
                if arm == "luna":
                    self.assertTrue(
                        body["input"][0]["content"][0]["image_url"].startswith("data:" + mime)
                    )
                    self.assertEqual(
                        [c["value"] for c in body["questions"][0]["choices"]], n.LABELS
                    )
                elif arm in ("clef", "d1", "d1-ultra"):
                    self.assertTrue(body["images"][0].startswith("data:" + mime))
                    self.assertEqual(list(body["questions"]["scene"]["criteria"]), n.LABELS)
                else:
                    self.assertEqual(
                        next(iter(body["assets"].values()))["format"],
                        "png" if fmt == "PNG" else "jpg",
                    )

    def test_cloudflare_envelope_and_failure_scoring(self):
        answer = {
            "type": "choice",
            "choice": n.LABELS[0],
            "probabilities": {x: int(i == 0) for i, x in enumerate(n.LABELS)},
        }
        response = Mock(status_code=200, ok=True)
        response.json.return_value = {
            "success": True,
            "result": {
                "model": "clef",
                "answers": {"scene": answer},
                "usage": {"input_tokens": 1000},
            },
        }
        with patch.object(n.requests, "post", return_value=response):
            result = n.predict(
                "clef",
                self.image("PNG"),
                {
                    "CLOUDFLARE_ACCOUNT_ID": "test",
                    "CLOUDFLARE_API_KEY": "test",
                    "LIQUID_AI_API_KEY": "test",
                },
                {},
            )
        self.assertEqual(n.accuracy(result, n.LABELS[0]), 1)
        self.assertAlmostEqual(result["estimated_api_cost_usd"], 0.00024)
        self.assertEqual(n.accuracy({"status": "api_error"}, n.LABELS[0]), 0)
        self.assertEqual(n.valid_output({"status": "invalid_output"}), 0)

    def test_refusal_retains_usage_and_is_not_a_valid_category(self):
        response = Mock(status_code=200, ok=True)
        response.json.return_value = {
            "model": "gpt-6-luna",
            "answers": [{"type": "refusal", "name": "scene"}],
            "usage": {"input_tokens": 250, "output_tokens": 0},
        }
        with patch.object(n.requests, "post", return_value=response):
            result = n.predict("luna", self.image("PNG"), {"OPENAI_API_KEY": "test"}, {})
        self.assertEqual(result["status"], "refusal")
        self.assertEqual(n.valid_output(result), 0)
        self.assertEqual(n.accuracy(result, "Airport"), 0)
        self.assertAlmostEqual(result["estimated_api_cost_usd"], 0.000025)

    def test_native_trial_hooks_and_scorers(self):
        trials = []

        def task(input, hooks):
            trials.append(hooks.trial_index)
            return {"status": "ok", "label": input}

        result = n.bt.Eval(
            "local-test",
            data=[{"input": "Airport", "expected": "Airport"}],
            task=task,
            scores=[n.accuracy, n.valid_output],
            trial_count=2,
            no_send_logs=True,
        )
        self.assertEqual(sorted(trials), [0, 1])
        self.assertEqual(len(result.results), 2)
        for row in result.results:
            self.assertEqual(row.scores["accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
