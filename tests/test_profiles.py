from __future__ import annotations

from pathlib import Path

import pytest

from sr_adapter.profiles import (
    LLMPolicy,
    ProcessingProfile,
    ProfileStore,
    load_processing_profile,
    resolve_profile,
)


def test_load_processing_profile_default() -> None:
    profile = load_processing_profile()
    assert isinstance(profile, ProcessingProfile)
    assert profile.name == "balanced"
    assert profile.llm_policy.max_confidence == pytest.approx(0.6)
    assert profile.metadata["description"].startswith("Balanced")


def test_profile_store_reads_custom_file(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "profiles"
    config_dir.mkdir()
    (config_dir / "custom.yaml").write_text(
        """
name: custom
layout_profile: custom-layout
layout_batch_size: 12
text_batch_size: 20
stream_normalize: false
warm_runtime: false
default_deadline_ms: 1500
metadata:
  description: Custom test profile
llm:
  enabled: true
  max_confidence: 0.3
  limit_block_types:
    - paragraph
  max_blocks: 2
  metadata:
    cadence: test
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("SR_ADAPTER_PROFILE_PATH", str(config_dir))
    store = ProfileStore()
    profile = store.load("custom")

    assert profile.name == "custom"
    assert profile.layout_profile == "custom-layout"
    assert profile.llm_policy.limit_block_types == ("paragraph",)
    assert profile.llm_policy.metadata["cadence"] == "test"


def test_resolve_profile_accepts_instance() -> None:
    policy = LLMPolicy(enabled=False)
    profile = ProcessingProfile(name="manual", llm_policy=policy)
    resolved = resolve_profile(profile)
    assert resolved is profile
