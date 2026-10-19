"""A matching value is insufficient when roles, rows or required context differ."""
import copy
import importlib.util
import json
from pathlib import Path

from sr_adapter.knowledge import sha256
from sr_adapter.narrative import NarrativeOntology, render_narrative, save_narrative
from sr_adapter.relation_sources import read_source

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data/knowledge/realistic-relations"
spec = importlib.util.spec_from_file_location("relation_evidence_probe", ROOT / "scripts/probe_relation_evidence.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def make_bundles(tmp_path, transform=None):
    gold = json.loads((FIXTURES / "gold.json").read_bytes())
    ontology = NarrativeOntology.model_validate_json((FIXTURES / "ontology.json").read_bytes())
    bundles = {}
    for doc in gold["documents"]:
        path = FIXTURES / doc["path"]
        data = path.read_bytes()
        source = read_source(data, path.name)
        units = {u["locator"]["line_start"]: u for u in source["units"]}
        rows = [{**{k: row[k] for k in probe.FIELDS}, "evidence": [
            {"unit_id": units[n]["id"], "quote": units[n]["text"]} for n in row["required_lines"]]}
            for row in doc["assertions"]]
        if transform:
            transform(doc["path"], rows)
        raw = {"status": "completed", "model": "gpt-5.5", "output": [{"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": json.dumps({"assertions": rows}, ensure_ascii=False)}]}]}
        manifest, files = render_narrative(data=data, source_name=path.name, ontology=ontology, model="gpt-5.5", response=raw)
        bundles[path.name] = tmp_path / path.name
        save_narrative(bundles[path.name], manifest, files)
    return bundles


def test_fixed_gold_has_complete_scoped_evidence(tmp_path):
    result = probe.score(FIXTURES / "gold.json", make_bundles(tmp_path))
    assert result["tuple_totals"] == {"tp": 17, "fp": 0, "fn": 0, "predicted": 17, "expected": 17}
    assert result["context_complete"] == result["context_expected"] == 17


def test_wrong_role_can_preserve_every_literal_value(tmp_path):
    def swap(name, rows):
        if name == "operations-review.md":
            rows[0]["object"], rows[1]["object"] = rows[1]["object"], rows[0]["object"]
    result = probe.score(FIXTURES / "gold.json", make_bundles(tmp_path, swap))
    md = result["documents"][0]
    assert md["literal_value_metrics"]["fp"] == md["literal_value_metrics"]["fn"] == 0
    assert md["tuple_metrics"]["fp"] == md["tuple_metrics"]["fn"] == 2
    assert md["context_complete"] == 8


def test_correct_numbers_without_unit_and_period_context_do_not_pass(tmp_path):
    def omit(name, rows):
        if name == "operations-review.md":
            rows[6]["evidence"] = rows[6]["evidence"][-1:]
    result = probe.score(FIXTURES / "gold.json", make_bundles(tmp_path, omit))
    assert result["tuple_totals"]["tp"] == 17
    assert result["context_complete"] == 16


def test_duplicate_rows_cannot_pool_evidence_to_pass(tmp_path):
    def duplicate(name, rows):
        if name == "operations-review.md":
            extra = copy.deepcopy(rows[0])
            extra["evidence"] = extra["evidence"][-1:]
            rows.append(extra)
    result = probe.score(FIXTURES / "gold.json", make_bundles(tmp_path, duplicate))
    assert result["tuple_totals"]["fp"] == 1
    assert result["context_complete"] == 16


def test_saved_live_comparison_replays_and_preserves_all_artifacts():
    root = ROOT / "docs/validation/relation-context-2026-10-19"
    manifest = json.loads((root / "manifest.json").read_bytes())
    assert set(manifest["files"]) == {p.relative_to(root).as_posix() for p in root.rglob("*")
                                      if p.is_file() and p != root / "manifest.json"}
    for name, expected in manifest["files"].items():
        data = (root / name).read_bytes()
        assert expected == {"sha256": sha256(data), "bytes": len(data)}
    summary = json.loads((root / "summary.json").read_bytes())
    assert summary["planned_requests"] == summary["completed_requests"] == 4
    for arm in ("baseline", "contextual"):
        bindings = {name: root / arm / name / "bundle" for name in ("operations-review.md", "procurement-ledger.csv")}
        actual = probe.score(root / "fixtures/gold.json", bindings)
        assert actual == json.loads((root / f"{arm}-score.json").read_bytes())
        total = 0
        for path in bindings.values():
            receipt = json.loads((path.parent / "receipt.json").read_bytes())
            raw = json.loads((path.parent / "extraction-response.json").read_bytes())
            assert receipt["status"] == "complete" and len(receipt["requests"]) == 1
            assert receipt["requests"][0]["usage"] == raw["usage"]
            assert sha256((path.parent / "extraction-response.json").read_bytes()) == receipt["requests"][0]["wire_response_sha256"]
            total += raw["usage"]["total_tokens"]
        assert total == summary["arms"][arm]["usage"]["total_tokens"]
