import unittest
from unittest.mock import Mock

from web_aid_eval import sync_reviewed_dataset


class ReviewedDatasetTests(unittest.TestCase):
    def test_removed_and_excluded_labels_leave_next_dataset(self):
        dataset = Mock()
        dataset.fetch.return_value = [{"id": "kept"}, {"id": "removed"}, {"id": "excluded"}]
        row = {"id": "review-id", "metadata": {"id": "kept"}, "input": {"image": "attachment"}}
        sync_reviewed_dataset(dataset, [(row, "Forest")])
        self.assertEqual(
            [call.args[0] for call in dataset.delete.call_args_list], ["removed", "excluded"]
        )
        dataset.insert.assert_called_once_with(
            id="kept",
            input=row["input"],
            expected="Forest",
            metadata={
                "id": "kept",
                "human_review_log_id": "review-id",
                "review_status": "human_labeled",
            },
        )
        dataset.flush.assert_called_once()

    def test_no_eligible_labels_clears_previous_membership(self):
        dataset = Mock()
        dataset.fetch.return_value = [{"id": "previously-labeled"}]
        sync_reviewed_dataset(dataset, [])
        dataset.delete.assert_called_once_with("previously-labeled")
        dataset.insert.assert_not_called()
        dataset.flush.assert_called_once()


if __name__ == "__main__":
    unittest.main()
