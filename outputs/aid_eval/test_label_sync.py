import unittest

from sync_image_labels import label_of, model_row_id, updates_for


class LabelSync(unittest.TestCase):
    def setUp(self):
        self.master = {"id": "source", "metadata": {"id": "image"}}
        self.rows = [
            {
                "id": "image",
                "model": m,
                "status": "ok",
                "label": "Forest",
                "inference_log_id": "original",
            }
            for m in ["luna", "jev", "clef", "d1", "d1-ultra"]
        ]

    def test_one_label_five_rows_and_original(self):
        events = updates_for(self.master, self.rows, "Forest")
        self.assertEqual(len(events), 6)
        self.assertEqual(len({e["id"] for e in events}), 6)
        self.assertTrue(all(e["expected"]["web_scene_category"] == "Forest" for e in events))
        self.assertTrue(all(e["scores"]["human_category_accuracy"] == 1 for e in events[:5]))
        self.assertTrue(all("output" not in e and "input" not in e for e in events))

    def test_correction_changes_score_and_removal_clears_it(self):
        self.assertEqual(
            updates_for(self.master, self.rows, "Park")[0]["scores"]["human_category_accuracy"], 0
        )
        for label in (None, "Ambiguous", "Unusable", "Other", "NotAerial"):
            self.assertIsNone(
                updates_for(self.master, self.rows, label)[0]["scores"]["human_category_accuracy"]
            )

    def test_refusal_not_correct(self):
        self.rows[0]["status"] = "refusal"
        self.assertEqual(
            updates_for(self.master, self.rows, "Forest")[0]["scores"]["human_category_accuracy"], 0
        )

    def test_invalid_label_and_wrong_image_rejected(self):
        with self.assertRaises(ValueError):
            updates_for(self.master, self.rows, "guess")
        self.rows[0]["id"] = "different"
        with self.assertRaises(AssertionError):
            updates_for(self.master, self.rows, "Forest")


if __name__ == "__main__":
    unittest.main()
