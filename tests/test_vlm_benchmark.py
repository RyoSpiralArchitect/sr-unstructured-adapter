"""Paired-input controls, strict scoring, and no-retry live-run boundaries."""

import copy
import importlib.util
import json
from pathlib import Path

import httpx
import pytest


spec = importlib.util.spec_from_file_location("vlm_benchmark", Path(__file__).parents[1] / "scripts/vlm_benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "page.png").write_bytes(b"fixed-image-bytes")
    (root / "input.pdf").write_bytes(b"fixed-pdf-bytes")
    bench.write_json(root / "text.json", {"pages": [{"page": 2, "text": "上限 12,500円。担当は未定。"}]})
    bench.write_json(root / "adapter.json", {"blocks": [{"text": "上限 12500円。担当は未定。"}]})
    case = {"id": "table", "kind": "synthetic", "images": [{**bench.file_record(root / "page.png", root), "page": 2}],
            "pdf": bench.file_record(root / "input.pdf", root),
            "text": bench.file_record(root / "text.json", root),
            "adapter": bench.file_record(root / "adapter.json", root),
            "fields": [
                {"id": "limit", "question": "上限は何円か。", "accepted": ["12500円"], "page": 2,
                 "evidence": ["上限 12,500円。"], "anchors": ["12,500"]},
                {"id": "owner", "question": "担当の氏名は何か。", "accepted": ["UNKNOWN"], "page": 2,
                 "evidence": ["担当は未定。"], "anchors": ["未定"]},
                {"id": "date", "question": "発効日はいつか。", "accepted": ["UNKNOWN"], "page": None,
                 "evidence": [], "anchors": []},
            ]}
    manifest = {"status": "frozen", "visual_review": {"note": "unit fixture"}, "cases": [case]}
    bench.write_json(root / "manifest.json", manifest)
    return root, case, manifest


def good_answer():
    return {"limit": {"value": "１２，５００ 円", "page": 2, "quote": "12,500円"},
            "owner": {"value": "UNKNOWN", "page": 2, "quote": "担当は未定。"},
            "date": {"value": "UNKNOWN", "page": None, "quote": ""}}


def response(answer=None, reason="stop", model=bench.MODEL):
    return {"model": model, "choices": [{"finish_reason": reason,
            "message": {"content": json.dumps(good_answer() if answer is None else answer)}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 40}}


def test_questions_images_and_sampling_identical_no_gold_leakage(corpus):
    root, case, _ = corpus
    case = copy.deepcopy(case)
    case["fields"][0]["accepted"] = ["SECRET_GOLD_SENTINEL"]
    case["fields"][0]["evidence"] = ["HIDDEN_ANNOTATION"]
    payloads = [bench.build_payload(case, root, arm) for arm in bench.ARMS]
    for payload in payloads:
        assert "SECRET_GOLD_SENTINEL" not in str(payload)
        assert "HIDDEN_ANNOTATION" not in str(payload)
        assert payload["messages"][0] == payloads[0]["messages"][0]
        assert payload["messages"][1]["content"][:-1] == payloads[0]["messages"][1]["content"][:-1]
        assert payload["max_completion_tokens"] == payloads[0]["max_completion_tokens"]
        assert payload["seed"] == payloads[0]["seed"]
        assert payload["store"] is False
    assert len({p["messages"][1]["content"][-1]["text"] for p in payloads}) == 3


def test_correct_values_and_source_quotes(corpus):
    result = bench.grade(corpus[1], response())
    assert result["valid"]
    assert all(f["grounded_correct"] for f in result["fields"])


@pytest.mark.parametrize("value", ["12500", "12500ドル", "-12500円", "125000円"])
def test_units_signs_and_magnitude_are_not_normalized_away(corpus, value):
    answer = good_answer()
    answer["limit"]["value"] = value
    assert not bench.grade(corpus[1], response(answer))["fields"][0]["value_correct"]


@pytest.mark.parametrize("page,quote", [(1, "12,500円"), (2, "未定"), (2, "これは12500円です")])
def test_wrong_page_or_unverifiable_quote_does_not_get_source_credit(corpus, page, quote):
    answer = good_answer()
    answer["limit"].update(page=page, quote=quote)
    result = bench.grade(corpus[1], response(answer))["fields"][0]
    assert result["value_correct"] and not result["grounded_correct"]


@pytest.mark.parametrize("value", ["田中", "未定"])
def test_noncanonical_unknown_is_counted_without_claiming_fabrication(corpus, value):
    answer = good_answer()
    answer["owner"]["value"] = value
    result = bench.grade(corpus[1], response(answer))["fields"][1]
    assert result["unknown_format_mismatch"] and not result["value_correct"]


@pytest.mark.parametrize("reason", ["length", "content_filter", None])
def test_correct_json_with_incomplete_response_has_no_credit(corpus, reason):
    result = bench.grade(corpus[1], response(reason=reason))
    assert not result["valid"]
    assert not any(f["value_correct"] for f in result["fields"])


def test_different_model_is_not_a_matched_observation(corpus):
    assert not bench.grade(corpus[1], response(model="another-model"))["valid"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "number", "bool_page"])
def test_invalid_answer_shape_fails_closed(corpus, mutation):
    answer = good_answer()
    if mutation == "missing":
        answer.pop("date")
    elif mutation == "extra":
        answer["extra"] = answer["date"]
    elif mutation == "number":
        answer["limit"]["value"] = 12500
    else:
        answer["limit"]["page"] = True
    assert not bench.grade(corpus[1], response(answer))["valid"]


def test_duplicate_keys_rejected_even_when_last_value_would_pass(corpus):
    raw = response()
    text = raw["choices"][0]["message"]["content"]
    raw["choices"][0]["message"]["content"] = '{"limit":null,' + text[1:]
    assert not bench.grade(corpus[1], raw)["valid"]


def test_hash_changes_rejected_before_inference(corpus):
    root, case, _ = corpus
    (root / "page.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        bench.build_payload(case, root, "vision")


def test_escaping_corpus_paths_rejected(corpus):
    root, case, _ = corpus
    case["images"][0]["path"] = "../elsewhere.png"
    with pytest.raises(ValueError, match="escapes"):
        bench.build_payload(case, root, "vision")


def test_schedule_has_matched_seed_and_full_denominator(corpus):
    rows = bench.schedule([corpus[1]], 2)
    assert len(rows) == 6
    for repeat in range(2):
        group = [r for r in rows if r["repeat"] == repeat]
        assert {r["arm"] for r in group} == set(bench.ARMS)
        assert len({r["seed"] for r in group}) == 1
    summary = bench.summarize(corpus[2], rows)
    for arm in summary["arms"].values():
        assert arm["field_denominator"] == 6
        assert arm["value_correct"] == 0
        assert arm["prompt_tokens_total"] is None
    for comparison in summary["paired"].values():
        assert comparison["counts"] == {"unavailable": 6}


def test_live_calls_require_opt_in_and_respect_budget(corpus, tmp_path, monkeypatch):
    root, _, _ = corpus
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    with pytest.raises(ValueError, match="allow-live-api"):
        bench.run(root, tmp_path / "no-opt-in")
    with pytest.raises(ValueError, match="exceeds"):
        bench.run(root, tmp_path / "over-budget", allow_live=True, max_requests=1)
    assert not (tmp_path / "over-budget").exists()


@pytest.mark.parametrize("change", ["unreviewed", "rejected", "duplicate_case", "no_images"])
def test_invalid_corpus_cannot_cross_provider_boundary(corpus, tmp_path, monkeypatch, change):
    root, case, manifest = corpus
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    if change == "unreviewed":
        manifest.pop("visual_review")
    elif change == "rejected":
        (root / "REJECTED.txt").write_text("bad rendering")
    elif change == "duplicate_case":
        manifest["cases"].append(case)
    else:
        case["images"] = []
    bench.write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError):
        bench.run(root, tmp_path / "result", allow_live=True)
    assert not (tmp_path / "result").exists()


def test_transport_failure_stops_all_requests_and_preserves_plan(corpus, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")

    class BrokenClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            calls.append(1)
            raise httpx.ReadTimeout("sensitive transport message must not be saved")

    monkeypatch.setattr(httpx, "Client", BrokenClient)
    result = bench.run(corpus[0], tmp_path / "result", allow_live=True)
    assert len(calls) == 1
    assert len(result["results"]) == 6
    assert [r["status"] for r in result["results"]] == ["failed"] + ["skipped"] * 5
    assert "sensitive transport" not in str(result)
    assert "unit-test-placeholder" not in str(result)
    assert result["summary"]["arms"]["vision"]["field_denominator"] == 6
    with pytest.raises(FileExistsError):
        bench.run(corpus[0], tmp_path / "result", allow_live=True)
    assert len(calls) == 1


def test_frozen_pilot_has_reviewable_annotations_and_source_hashes():
    spec = json.loads((Path(__file__).parents[1] / "data/vlm_benchmark/pilot-v1.json").read_text("utf-8"))
    assert len({c["id"] for c in spec["cases"]}) == 8
    assert sum(len(c["fields"]) for c in spec["cases"]) == 48
    for case in spec["cases"]:
        assert len({f["id"] for f in case["fields"]}) == len(case["fields"])
        if case["kind"] == "digital_agency":
            assert len(case["source_sha256"]) == 64
        for field in case["fields"]:
            assert field["accepted"]
            if field["page"] is None:
                assert field["accepted"] == ["UNKNOWN"]
            else:
                assert field["page"] in case["source_pages"]
                assert field["anchors"] and field["evidence"]
