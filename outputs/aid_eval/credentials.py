"""Load explicit credential files as data, without evaluating shell expressions."""

import os
import re
import shlex
from pathlib import Path

LIQUID_KEYS = ("LIQUID_AI_API_KEY", "LIQUID_AI_KEY", "LIQUID_API_KEY")
ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def _read_assignments(path, allowed=None):
    values = {}
    path = Path(path).expanduser()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = ASSIGNMENT.fullmatch(line)
        if allowed is not None and (not match or match[1] not in allowed):
            continue
        if not match:
            raise ValueError(f"Expected a literal assignment at {path}:{number}")
        try:
            parts = shlex.split(match[2], comments=True)
        except ValueError:
            raise ValueError(f"Invalid quoting at {path}:{number}") from None
        if len(parts) > 1 or any(c in match[2] for c in ("$", "`")):
            raise ValueError(f"Expected a literal value at {path}:{number}")
        values[match[1]] = parts[0] if parts else ""
    return values


def keys_load(env_file=None, liquid_key_file=None):
    """Explicit files override the environment; no files are read by default.

    AID_ENV_FILE and AID_LIQUID_KEY_FILE provide the same configuration to
    maintenance scripts that do not have credential command-line options.
    """
    keys = dict(os.environ)
    env_file = env_file or keys.get("AID_ENV_FILE")
    liquid_key_file = liquid_key_file or keys.get("AID_LIQUID_KEY_FILE")
    if env_file:
        keys.update(_read_assignments(env_file))
    if liquid_key_file:
        liquid = _read_assignments(liquid_key_file, allowed=LIQUID_KEYS)
        for value in liquid.values():
            keys["LIQUID_AI_API_KEY"] = value
    if not keys.get("LIQUID_AI_API_KEY"):
        keys["LIQUID_AI_API_KEY"] = keys.get("LIQUID_AI_KEY") or keys.get("LIQUID_API_KEY")
    return keys


def validate_keys(keys, models=()):
    """Fail before dataset access or inference if required credentials are absent."""
    required = {"BRAINTRUST_API_KEY"}
    for model in models:
        required.update(
            {
                "luna": ("OPENAI_API_KEY",),
                "jev": ("TYPESAFE_API_KEY",),
                "clef": ("CLOUDFLARE_ACCOUNT_ID",),
                "d1": ("LIQUID_AI_API_KEY",),
                "d1-ultra": ("LIQUID_AI_API_KEY",),
            }[model]
        )
    missing = sorted(name for name in required if not keys.get(name))
    if "clef" in models and not any(
        keys.get(name)
        for name in ("CLOUDFLARE_AUTH_TOKEN", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY")
    ):
        missing.append("CLOUDFLARE_AUTH_TOKEN (or CLOUDFLARE_API_TOKEN / CLOUDFLARE_API_KEY)")
    if missing:
        raise ValueError("Missing credentials: " + ", ".join(missing))


def add_credential_arguments(parser):
    parser.add_argument(
        "--env-file", type=Path, help="Literal KEY=value file; defaults to AID_ENV_FILE"
    )
    parser.add_argument(
        "--liquid-key-file",
        type=Path,
        help="Read literal Liquid key assignments only; never source shell files",
    )
