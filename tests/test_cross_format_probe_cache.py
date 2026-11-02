"""Prepared reuse evidence stays bound to the recorded protocol and outcomes."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest

from sr_adapter.knowledge import sha256
from sr_adapter.ontology_ingest import ingest_ontology_source
from test_docx_relation_sources import package, paragraph
from test_ontology_ingest import proposed


@pytest.fixture
def cached_probe(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    probe = importlib.import_module("probe_cross_format_ontology")
    monkeypatch.setattr("httpx.Client", lambda *args, **kwargs: pytest.fail("Reuse made a network call"))
    path = tmp_path / "policy.docx"
    data = package(paragraph("Review requires approval."))
    path.write_bytes(data)
    receipt = ingest_ontology_source(path, tmp_path / "ontology.sqlite", collection="cross-format",
        model="fixture", complete=proposed, adapter_id="cache-fixture", execute=True)
    assert receipt["status"] == "complete", receipt
    source = {"path": str(path), "source_name": path.name, "format": "docx", "sha256": sha256(data)}
    protocol = {"version": 1, "model": probe.MODEL, "questions": probe.QUESTIONS, "sources": [source]}
    outcomes = [{"source": source, "receipt": receipt}]
    probe.save(tmp_path / "protocol.json", protocol)
    probe.save(tmp_path / "ingestions.json", outcomes)
    path.unlink()
    baseline = probe.collect(tmp_path)
    assert len(baseline[2]) == 1
    return probe, tmp_path, baseline


def test_unchanged_cached_evidence_remains_sqlite_only(cached_probe):
    probe, root, baseline = cached_probe
    assert probe.collect(root) == baseline
    assert probe.reuse(root) == {"status": "prepared", "model_requests": 0}
    assert not (root / "reuse.submitted").exists()
    readback = json.loads((root / "readback.json").read_bytes())
    assert readback["sources"][0]["snapshot_id"] == baseline[1][0]["receipt"]["snapshot_id"]


@pytest.mark.parametrize("change", ["status", "snapshot", "new_recovery", "protocol"])
def test_changed_outcomes_or_protocol_reject_before_paid_reuse(cached_probe, change):
    probe, root, baseline = cached_probe
    protocol, outcomes = deepcopy(baseline[:2])
    if change == "protocol":
        protocol["questions"]["draft_qualifier"] = "A changed question after preparation."
        probe.save(root / "protocol.json", protocol)
    elif change == "new_recovery":
        outcomes[0]["receipt"]["run_id"] = "f" * 32
        outcomes[0]["receipt"]["recovered_from_run_id"] = baseline[1][0]["receipt"]["run_id"]
        probe.save(root / "recoveries.json", outcomes)
    else:
        receipt = outcomes[0]["receipt"]
        receipt["status" if change == "status" else "snapshot_id"] = "failed" if change == "status" else "b" * 64
        probe.save(root / "ingestions.json", outcomes)
    with pytest.raises(ValueError, match="Prepared reuse evidence changed"):
        probe.reuse(root, execute=True)
    assert not (root / "reuse.submitted").exists()
    assert not (root / "reuse-request.json").exists()
