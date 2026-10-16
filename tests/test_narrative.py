import copy
import csv
import io
import json
from types import SimpleNamespace

import httpx
import pytest

from sr_adapter.knowledge import sha256
from sr_adapter.narrative import (NarrativeOntology, extraction_request, narrative_cards,
    render_narrative, save_narrative, verify_narrative, verify_narrative_payloads)
from sr_adapter.narrative_runtime import extract_narrative
from sr_adapter.relation_sources import read_source
from sr_adapter.tables import canonical_json


@pytest.fixture
def ontology():
    return NarrativeOntology.model_validate({"namespace": "urn:sr:test:", "classes": {"Project": "案件", "Organization": "組織"},
        "predicates": [{"id": "managedBy", "label": "担当", "subject_type": "Project", "object_type": "Organization", "description": "主担当"}]})


def response(assertions, **extra):
    return {"status": "completed", "model": "gpt-5.5-2026-04-24", "output": [{"type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": json.dumps({"assertions": assertions}, ensure_ascii=False)}]}], **extra}


def candidate(data=b'A is managed by B.', status="asserted", object="B"):
    source = read_source(data, "source.txt")
    return {"subject": "A", "predicate": "managedBy", "object": object, "status": status,
        "evidence": [{"unit_id": source["units"][0]["id"], "quote": source["units"][0]["text"]}]}


def bundle(ontology, row=None, data=b'A is managed by B.'):
    return render_narrative(data=data, source_name="source.txt", ontology=ontology, model="gpt-5.5",
                            response=response([row or candidate(data)]))


def test_table_reproduces_original_evidence_and_candidate_scope(ontology, tmp_path):
    manifest, files = bundle(ontology)
    save_narrative(tmp_path / "bundle", manifest, files)
    assert verify_narrative(tmp_path / "bundle") == (manifest, files)
    row = next(csv.DictReader(io.StringIO(files["relations.csv"].decode())))
    assert row["subject"] == "A" and row["object"] == "B"
    assert row["interpretation"] == "model_candidate"
    assert row["grounding"] == "source_text_verified"
    assert json.loads(row["evidence_json"])[0]["quote"] == "A is managed by B."
    assert "semantic validation: not performed" in narrative_cards(files)[0]["content"]


@pytest.mark.parametrize("status,object,text", [
    ("denied", "B", b'A is not managed by B.'), ("unknown", None, b'A manager is unknown.'),
    ("ambiguous", None, b'A may be managed by B or C; no decision.')])
def test_nonpositive_status_is_preserved(ontology, status, object, text):
    _, files = bundle(ontology, candidate(text, status, object), text)
    row = json.loads(files["assertions.json"])[0]
    assert row["status"] == status and row["object"] == object
    assert narrative_cards(files)[0]["assertion_status"] == status


@pytest.mark.parametrize("patch", [
    {"predicate": "invented"}, {"subject": "C"}, {"object": "C"}, {"object": None},
    {"status": "unknown"}, {"status": "ambiguous"}, {"subject": " A "},
    {"evidence": [{"unit_id": "not-present", "quote": "A is managed by B."}]},
    {"evidence": [{"unit_id": "text:1", "quote": "A is managed by C."}]},
])
def test_invalid_candidate_cannot_be_published(ontology, patch):
    row = candidate()
    row.update(patch)
    with pytest.raises(ValueError):
        bundle(ontology, row)


def test_removing_condition_from_evidence_is_rejected(ontology):
    text = b'If approved tomorrow, A is managed by B.'
    row = candidate(text)
    row["evidence"][0]["quote"] = "A is managed by B."
    with pytest.raises(ValueError, match="full source unit"):
        bundle(ontology, row, text)


def test_quote_match_does_not_pretend_semantic_validation(ontology):
    row = candidate()
    row["subject"], row["object"] = "B", "A"
    manifest, files = bundle(ontology, row)
    assert manifest["semantic_validation"] == "not_performed"
    assert json.loads(files["assertions.json"])[0]["interpretation"] == "model_candidate"


def test_rehashed_table_edit_fails_semantic_regeneration(ontology):
    _, files = bundle(ontology)
    files["relations.csv"] = files["relations.csv"].replace(b',A,', b',C,')
    manifest = json.loads(files["manifest.json"])
    manifest["files"]["relations.csv"] = {"sha256": sha256(files["relations.csv"]), "bytes": len(files["relations.csv"])}
    files["manifest.json"] = canonical_json(manifest) + b"\n"
    with pytest.raises(ValueError, match="reproduce"):
        verify_narrative_payloads(files)


@pytest.mark.parametrize("raw", [response([], status="incomplete"), response([], error={"message": "bad"}),
    {"status": "completed", "model": "gpt-5.5", "output": [{"type": "message", "role": "assistant", "content": [{"type": "refusal"}]}]}])
def test_incomplete_or_refused_response_is_not_a_bundle(ontology, raw):
    with pytest.raises(ValueError):
        render_narrative(data=b'A is managed by B.', source_name="source.txt", ontology=ontology, model="gpt-5.5", response=raw)


def test_duplicate_candidates_and_duplicate_evidence_rejected(ontology):
    row = candidate()
    for rows in ([row, row], [{**row, "evidence": row["evidence"] * 2}]):
        with pytest.raises(ValueError, match="Duplicate"):
            render_narrative(data=b'A is managed by B.', source_name="source.txt", ontology=ontology, model="gpt-5.5", response=response(rows))


def test_identical_mentions_in_distinct_sources_are_not_joined(ontology):
    a = bundle(ontology)[1]
    b = render_narrative(data=b'A is managed by B.', source_name="another.txt", ontology=ontology,
        model="gpt-5.5", response=response([candidate()]))[1]
    assert json.loads(a["assertions.json"])[0]["subject_mention_id"] != json.loads(b["assertions.json"])[0]["subject_mention_id"]


def test_source_traversal_and_added_files_rejected(ontology, tmp_path):
    manifest, files = bundle(ontology)
    files["extra"] = b"arbitrary"
    with pytest.raises(ValueError):
        verify_narrative_payloads(files)
    root = tmp_path / "bundle"
    root.mkdir()
    manifest["files"] = {"../secret": {}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        verify_narrative(root)


def test_schema_has_dynamic_predicates_and_no_secrets(ontology):
    request = extraction_request(read_source(b'A is managed by B.', 'source.txt'), ontology, "deployment")
    assert request["store"] is False
    assert request["text"]["format"]["schema"]["$defs"]["Candidate"]["properties"]["predicate"]["enum"] == ["managedBy"]


def test_live_runtime_records_before_submit_and_never_overwrites(ontology, tmp_path):
    source = tmp_path / 'source.txt'
    source.write_bytes(b'A is managed by B.')
    out = tmp_path / "attempt"
    def handler(request):
        receipt = json.loads((out / 'receipt.json').read_bytes())
        assert receipt['requests'][0]['status'] == 'submitting'
        assert (out / 'extraction-request.json').exists()
        return httpx.Response(200, json=response([candidate()]))
    credential = SimpleNamespace(get_token=lambda _: SimpleNamespace(token='fake'))
    kwargs = dict(ontology=ontology, endpoint='https://unit.services.ai.azure.com/api/projects/test', model='gpt-5.5',
        destination=out, execute=True, credential=credential, transport=httpx.MockTransport(handler))
    assert extract_narrative(source, **kwargs)['status'] == 'complete'
    verify_narrative(out / 'bundle')
    with pytest.raises(FileExistsError):
        extract_narrative(source, **kwargs)


@pytest.mark.parametrize('mode,expected', [('timeout', 'unknown'), ('server', 'unknown'), ('rejected', 'failed'), ('invalid', 'failed')])
def test_failed_extraction_preserves_receipt_without_bundle(ontology, tmp_path, mode, expected):
    source = tmp_path / 'source.txt'
    source.write_bytes(b'A is managed by B.')
    calls = []
    def handler(request):
        calls.append(request)
        if mode == 'timeout':
            raise httpx.ReadTimeout('not logged', request=request)
        if mode == 'server':
            return httpx.Response(500, json={'error': 'failed'})
        if mode == 'rejected':
            return httpx.Response(400, json={'error': 'failed'})
        return httpx.Response(200, json=response([{**candidate(), 'subject': 'invented'}]))
    out = tmp_path / 'attempt'
    result = extract_narrative(source, ontology=ontology, endpoint='https://unit.services.ai.azure.com/api/projects/test',
        model='gpt-5.5', destination=out, execute=True, credential=SimpleNamespace(get_token=lambda _: SimpleNamespace(token='fake')),
        transport=httpx.MockTransport(handler))
    assert result['status'] == expected
    assert len(calls) == 1 and not (out / 'bundle').exists()
    assert 'not logged' not in (out / 'receipt.json').read_text()


def test_dry_run_is_offline_and_bounded(ontology, tmp_path):
    source = tmp_path / 'source.txt'
    source.write_bytes(b'A is managed by B.')
    result = extract_narrative(source, ontology=ontology, endpoint='https://unit.services.ai.azure.com/api/projects/test',
        model='gpt-5.5', destination=tmp_path / 'attempt')
    assert result['status'] == 'prepared' and result['requests'] == []


def test_image_transcription_is_not_source_text_verification(ontology):
    from PIL import Image
    data = io.BytesIO()
    Image.new('RGB', (20, 20), 'white').save(data, format='PNG')
    text = {'lines': ['A is managed by B.']}
    source = read_source(data.getvalue(), 'image.png', transcription=text)
    row = candidate()
    row['evidence'][0]['unit_id'] = source['units'][0]['id']
    transcript = copy.deepcopy(response([]))
    transcript['output'][0]['content'][0]['text'] = json.dumps(text)
    manifest, files = render_narrative(data=data.getvalue(), source_name='image.png', ontology=ontology,
        model='gpt-5.5', response=response([row]), transcription_response=transcript)
    assert verify_narrative_payloads(files) == manifest
    assert narrative_cards(files)[0]['grounding'] == 'model_transcription_unverified'
    assert json.loads(files['assertions.json'])[0]['evidence'][0]['locator']['kind'] == 'image'


def test_image_runtime_preserves_both_calls_and_replays(ontology, tmp_path):
    from PIL import Image
    image = tmp_path / 'input.png'
    Image.new('RGB', (12, 12), 'white').save(image)
    transcript = {'lines': ['A is managed by B.']}
    image_source = read_source(image.read_bytes(), image.name, transcription=transcript)
    row = candidate()
    row['evidence'][0]['unit_id'] = image_source['units'][0]['id']
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            assert body['input'][1]['content'][0]['type'] == 'input_image'
            raw = response([])
            raw['output'][0]['content'][0]['text'] = json.dumps(transcript)
            return httpx.Response(200, json=raw)
        assert body['input'][1]['content'][0]['type'] == 'input_text'
        assert 'model_transcription_unverified' in body['input'][1]['content'][0]['text']
        return httpx.Response(200, json=response([row]))
    result = extract_narrative(image, ontology=ontology, endpoint='https://unit.services.ai.azure.com/api/projects/test',
        model='gpt-5.5', destination=tmp_path / 'attempt', execute=True,
        credential=SimpleNamespace(get_token=lambda _: SimpleNamespace(token='fake')), transport=httpx.MockTransport(handler))
    assert result['status'] == 'complete' and result['planned_model_requests'] == len(calls) == 2
    manifest, files = verify_narrative(tmp_path / 'attempt/bundle')
    assert manifest['assertion_count'] == 1 and files['source.bin'] == image.read_bytes()
