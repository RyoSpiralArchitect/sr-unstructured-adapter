"""Frozen evidence must survive Git's Windows-style checkout unchanged."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required to exercise checkout conversion")
def test_git_checkout_preserves_frozen_artifact_bytes(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copyfile(repo / ".gitattributes", checkout / ".gitattributes")
    expected = {}
    for source in sorted((repo / "docs/validation").rglob("*")):
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        name = source.relative_to(repo)
        destination = checkout / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected[name] = source.read_bytes()
        destination.write_bytes(expected[name])
    assert expected
    control = checkout / "ordinary.txt"
    control.write_bytes(b"ordinary text\nsecond line\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_ATTR_NOSYSTEM"] = "1"

    def git(*args):
        return subprocess.run(
            ["git", "-c", "core.autocrlf=true", "-c", "core.safecrlf=false",
             "-c", f"core.attributesFile={os.devnull}", *args],
            cwd=checkout, env=env, check=True, capture_output=True, timeout=30,
        )

    git("init", "--quiet")
    git("add", "--", ".gitattributes", "docs/validation", "ordinary.txt")
    for name in expected:
        (checkout / name).unlink()
    control.unlink()
    git("checkout-index", "--all", "--force")

    # Confirm that CRLF conversion really ran, while the frozen artifacts kept
    # their exact bytes rather than relying on a permissive hash validator.
    assert control.read_bytes() == b"ordinary text\r\nsecond line\r\n"
    for name, data in expected.items():
        assert (checkout / name).read_bytes() == data, name.as_posix()
