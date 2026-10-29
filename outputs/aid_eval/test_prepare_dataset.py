import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import prepare_dataset as p
from PIL import Image


class PrepareDatasetTests(unittest.TestCase):
    def fixture(self):
        buffer = io.BytesIO()
        Image.new("RGB", (600, 600)).save(buffer, "PNG")
        raw = buffer.getvalue()
        return raw, {
            "id": "airport_1.jpg",
            "image": "airport_1.jpg",
            "label": "Airport",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "dimensions": [600, 600],
        }

    def test_detects_png_bytes_despite_jpg_extension_and_reuses_cache(self):
        raw, case = self.fixture()
        with tempfile.TemporaryDirectory() as directory, patch.object(p.requests, "get") as get:
            get.return_value = Mock(content=raw)
            prepared = p.prepare([case], Path(directory))
            self.assertEqual(prepared[0][2], "image/png")
            p.prepare([case], Path(directory))
            get.assert_called_once()
            prepared[0][1].write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                p.prepare([case], Path(directory))

    def test_rejects_changed_source_without_writing_cache(self):
        _, case = self.fixture()
        with tempfile.TemporaryDirectory() as directory, patch.object(p.requests, "get") as get:
            get.return_value = Mock(content=b"unexpected source")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                p.prepare([case], Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_publication_refuses_to_modify_existing_dataset(self):
        with patch.object(p.bt, "login"), patch.object(p.bt, "init_dataset") as init:
            init.return_value.fetch.return_value = [{"id": "existing"}]
            with self.assertRaisesRegex(ValueError, "not empty"):
                p.publish([], "example", "pilot", "validation", {"BRAINTRUST_API_KEY": "test"})
            init.return_value.insert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
