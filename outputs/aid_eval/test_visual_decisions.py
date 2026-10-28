import base64
import io
import itertools
import unittest

import visual_decisions as v
from analyze_robustness import sign_flip_p
from PIL import Image


class DecisionTests(unittest.TestCase):
    def test_payload_preserves_pair_order_and_bytes(self):
        raws = []
        for color in ["red", "blue"]:
            buf = io.BytesIO()
            Image.new("RGB", (600, 600), color).save(buf, format="PNG")
            raws.append(buf.getvalue())
        keys = {
            k: "test"
            for k in [
                "OPENAI_API_KEY",
                "TYPESAFE_API_KEY",
                "CLOUDFLARE_ACCOUNT_ID",
                "CLOUDFLARE_API_KEY",
                "LIQUID_AI_API_KEY",
            ]
        }
        for model in v.MODELS:
            for ordered in (raws, raws[::-1]):
                _, _, body = v.request_payload(model, ordered, keys, v.SPECS["paired-sharpness"])
                if model == "luna":
                    encoded = [x["image_url"].split(",", 1)[1] for x in body["input"][0]["content"]]
                elif model == "jev":
                    encoded = [body["assets"][body["state"][k]]["data_base64"] for k in ["A", "B"]]
                else:
                    encoded = [x.split(",", 1)[1] for x in body["images"]]
                self.assertEqual([base64.b64decode(x) for x in encoded], ordered)

    def test_cluster_sign_flip_against_enumeration(self):
        d = [-2, -1, 0, 2, 2]
        sums = [
            abs(sum(x * s for x, s in zip(d, signs)))
            for signs in itertools.product([-1, 1], repeat=len(d))
        ]
        expected = sum(x >= abs(sum(d)) for x in sums) / len(sums)
        self.assertAlmostEqual(sign_flip_p(d), expected)


if __name__ == "__main__":
    unittest.main()
