import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from credentials import keys_load, validate_keys


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "credentials.env"

    def test_environment_only_does_not_read_files(self):
        os.environ["LIQUID_API_KEY"] = "alias"
        with patch.object(Path, "read_text", side_effect=AssertionError("Unexpected file read")):
            self.assertEqual(keys_load()["LIQUID_AI_API_KEY"], "alias")

    def test_explicit_file_supports_export_quotes_comments_and_precedence(self):
        os.environ["OPENAI_API_KEY"] = "environment"
        self.path.write_text('# comment\nexport OPENAI_API_KEY="file=value" # comment\nEMPTY=\n')
        keys = keys_load(self.path)
        self.assertEqual(keys["OPENAI_API_KEY"], "file=value")
        self.assertEqual(keys["EMPTY"], "")

    def test_file_environment_variable_and_explicit_override(self):
        os.environ["AID_ENV_FILE"] = str(self.path)
        self.path.write_text("OPENAI_API_KEY=configured\n")
        self.assertEqual(keys_load()["OPENAI_API_KEY"], "configured")
        other = self.path.with_name("other.env")
        other.write_text("OPENAI_API_KEY=explicit\n")
        self.assertEqual(keys_load(other)["OPENAI_API_KEY"], "explicit")

    def test_shell_file_reads_only_literal_liquid_assignments(self):
        self.path.write_text(
            'source /does/not/exist\nexport OTHER=$(false)\nexport LIQUID_AI_KEY="literal"\n'
        )
        self.assertEqual(keys_load(liquid_key_file=self.path)["LIQUID_AI_API_KEY"], "literal")

    def test_expressions_and_bad_quotes_are_rejected_without_echoing_secrets(self):
        for value in ("$(echo secret)", "`echo secret`", "$SECRET", '"secret', "secret extra"):
            with self.subTest(value=value):
                self.path.write_text("LIQUID_AI_API_KEY=" + value)
                with self.assertRaises(ValueError) as error:
                    keys_load(self.path)
                self.assertNotIn("secret", str(error.exception))

    def test_missing_credentials_and_cloudflare_aliases(self):
        with self.assertRaisesRegex(ValueError, "BRAINTRUST_API_KEY"):
            validate_keys({})
        keys = {"BRAINTRUST_API_KEY": "test", "CLOUDFLARE_ACCOUNT_ID": "test"}
        with self.assertRaisesRegex(ValueError, "CLOUDFLARE_AUTH_TOKEN"):
            validate_keys(keys, ["clef"])
        for name in ("CLOUDFLARE_AUTH_TOKEN", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY"):
            validate_keys({**keys, name: "test"}, ["clef"])
        validate_keys(keys, [])


if __name__ == "__main__":
    unittest.main()
