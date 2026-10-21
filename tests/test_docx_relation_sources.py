"""Synthetic DOCX source positions and ordinary narrative-path compatibility."""
from __future__ import annotations

import csv
import io
import json
from html import escape
from types import SimpleNamespace
import zipfile

import httpx
import pytest

from sr_adapter.narrative import NarrativeOntology, verify_narrative
from sr_adapter.narrative_runtime import extract_narrative
from sr_adapter.relation_sources import read_source

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PR = 'http://schemas.openxmlformats.org/package/2006/relationships'
CT = 'http://schemas.openxmlformats.org/package/2006/content-types'


def package(body, extra=None):
    parts = {
        '[Content_Types].xml': f'<Types xmlns="{CT}"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>',
        '_rels/.rels': f'<Relationships xmlns="{PR}"><Relationship Id="document" Type="{R}/officeDocument" Target="word/document.xml"/></Relationships>',
        'word/document.xml': f'<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>{body}</w:body></w:document>',
        **(extra or {}),
    }
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    return out.getvalue()


def paragraph(text):
    return '<w:p><w:r><w:t xml:space="preserve">' + escape(text) + '</w:t></w:r></w:p>'


def cell(text='', props='', body=None):
    return f'<w:tc><w:tcPr>{props}</w:tcPr>{paragraph(text) if body is None else body}</w:tc>'


def row(cells, props=''):
    return f'<w:tr><w:trPr>{props}</w:trPr>{"".join(cells)}</w:tr>'


def table(rows, columns=2):
    return '<w:tbl><w:tblGrid>' + '<w:gridCol w:w="100"/>' * columns + '</w:tblGrid>' + ''.join(rows) + '</w:tbl>'


def test_docx_paragraphs_cells_and_nested_tables_keep_xml_order_and_stable_ids():
    nested = table([row([cell('Nested A'), cell('Nested B')])])
    grid = table([row([cell(body=paragraph('Before nested') + nested + paragraph('After nested')), cell('Peer')])])
    data = package(paragraph('Introduction') + paragraph('') + grid + paragraph('Conclusion'))
    result = read_source(data, 'document.DOCX')
    units = result['units']
    assert [u['text'] for u in units] == ['Introduction', 'Before nested', 'Nested A', 'Nested B', 'After nested', 'Peer', 'Conclusion']
    assert units[0]['locator'] == {'kind': 'docx', 'part': 'word/document.xml', 'block': 1, 'paragraph': 1}
    assert units[-1]['locator']['block'] == 4
    assert units[-1]['locator']['paragraph'] == 3
    assert units[1]['locator']['table'] == 1
    assert units[2]['locator']['table'] == 2
    assert units[2]['locator']['parent_cell']['table'] == 1
    assert units[4]['locator']['block'] == 3
    assert units[4]['locator']['paragraph'] == 2
    assert len({u['id'] for u in units}) == len(units)
    assert result == read_source(data, 'document.DOCX')
    assert result['media_type'].endswith('wordprocessingml.document')
    assert result['needs_transcription'] is False


def test_docx_grid_spans_header_context_empty_cells_and_vertical_merge():
    data = package(table([
        row([cell('Parent', '<w:gridSpan w:val="2"/>')], '<w:tblHeader/>'),
        row([cell('Project'), cell('Owner')], '<w:tblHeader w:val="on"/>'),
        row([cell('A', '<w:vMerge w:val="restart"/>'), cell('B')]),
        row([cell('', '<w:vMerge/>'), cell('C')]),
    ]))
    units = read_source(data, 'table.docx')['units']
    assert [u['text'] for u in units] == ['Parent', 'Project', 'Owner', 'A', 'B', '', 'C']
    assert units[0]['locator']['cell']['column_span'] == 2
    assert units[4]['locator']['cell']['column'] == 2
    assert [h['text'] for h in units[4]['locator']['header_context']] == ['Parent', 'Owner']
    assert units[3]['locator']['cell']['vertical_merge'] == 'restart'
    assert units[5]['locator']['cell']['vertical_merge'] == 'continue'
    assert units[5]['text'] == ''  # No filling merged cells with invented repeat text.


def test_docx_omitted_grid_columns_and_legacy_horizontal_merges_are_explicit():
    grid = table([
        row([cell('A', '<w:hMerge w:val="restart"/>'), cell('', '<w:hMerge/>')], '<w:gridBefore w:val="1"/>'),
        row([cell('B')], '<w:gridAfter w:val="2"/>'),
    ], columns=3)
    units = read_source(package(grid), 'table.docx')['units']
    assert [u['locator']['cell']['column'] for u in units] == [2, 3, 1]
    assert units[0]['locator']['cell']['horizontal_merge'] == 'restart'
    assert units[1]['locator']['cell']['horizontal_merge'] == 'continue'
    assert all(u['locator']['header_context'] == [] for u in units)


def test_docx_row_and_cell_content_controls_preserve_grid_headers_and_order():
    def control(content):
        return '<w:sdt><w:sdtPr/><w:sdtContent>' + content + '</w:sdtContent></w:sdt>'
    header = row([cell('Project'), cell('Owner')], '<w:tblHeader/>')
    ordinary = table([header, row([cell('A'), cell('B')]), row([cell('C'), cell('D')])])
    wrapped = table([control(header), control(row([control(cell('A')), control(cell('B'))])),
                     row([cell('C'), control(cell('D'))])])
    expected = read_source(package(ordinary), 'ordinary.docx')
    result = read_source(package(wrapped), 'wrapped.docx')
    assert [(u['text'], u['locator']) for u in result['units']] == [
        (u['text'], u['locator']) for u in expected['units']]
    assert result['warnings'] == expected['warnings']


def test_docx_hyperlink_text_fields_revisions_and_content_controls_without_fetch(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Network is not permitted for DOCX parsing')
    monkeypatch.setattr('socket.create_connection', forbidden)
    body = ('<w:sdt><w:sdtPr/><w:sdtContent><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>'
            '<w:r><w:t>A </w:t><w:tab/></w:r><w:hyperlink r:id="link"><w:r><w:t>owns B</w:t></w:r></w:hyperlink>'
            '<w:r><w:br/><w:instrText> DO NOT EMIT FIELD CODE </w:instrText><w:t>Cached result</w:t></w:r>'
            '<w:del><w:r><w:delText>deleted</w:delText></w:r></w:del><w:ins><w:r><w:t>inserted</w:t></w:r></w:ins>'
            '</w:p></w:sdtContent></w:sdt>')
    extra = {'word/_rels/document.xml.rels': f'<Relationships xmlns="{PR}"><Relationship Id="link" Type="{R}/hyperlink" Target="https://example.invalid/private" TargetMode="External"/></Relationships>'}
    result = read_source(package(body, extra), 'links.docx')
    assert result['units'][0]['text'] == 'A \towns B\nCached resultinserted'
    assert result['units'][0]['locator']['paragraph_style'] == 'Heading1'
    assert any('revisions' in w for w in result['warnings'])


def test_docx_reports_non_body_content_without_guessing_pages():
    body = paragraph('Visible') + '<w:p><w:r><w:drawing><w:t>drawing text</w:t></w:drawing><w:footnoteReference w:id="1"/></w:r></w:p>'
    result = read_source(package(body), 'drawings.docx')
    assert [u['text'] for u in result['units']] == ['Visible']
    assert all('page' not in u['locator'] for u in result['units'])
    assert any('drawings' in w for w in result['warnings'])
    assert any('footnotes' in w for w in result['warnings'])


@pytest.mark.parametrize('body', [
    table([row([cell('A')])]),
    table([row([cell('A', '<w:gridSpan w:val="0"/>'), cell('B')])]),
    table([row([cell('A', '<w:gridSpan w:val="3"/>')])]),
    table([row([cell('A', '<w:vMerge w:val="maybe"/>'), cell('B')])]),
    table([row([cell('A'), cell('B')], '<w:gridBefore w:val="-1"/>')]),
])
def test_docx_invalid_grid_rejects_without_partial_units(body):
    with pytest.raises(ValueError, match='DOCX table'):
        read_source(package(paragraph('Before invalid table') + body), 'bad.docx')


@pytest.mark.parametrize('kind,target,mode', [
    ('image', 'https://example.invalid/image.png', 'External'),
    ('hyperlink', 'https://example.invalid/', 'Internal'),
    ('image', '../../escape.png', 'Internal'),
    ('image', 'missing.png', 'Internal'),
])
def test_docx_unsupported_external_or_invalid_internal_relationships_reject(kind, target, mode):
    extra = {'word/_rels/document.xml.rels': f'<Relationships xmlns="{PR}"><Relationship Id="rId1" Type="{R}/{kind}" Target="{target}" TargetMode="{mode}"/></Relationships>'}
    with pytest.raises(ValueError, match='DOCX'):
        read_source(package(paragraph('Visible'), extra), 'bad.docx')


def test_docx_shared_package_safeguards_and_strict_namespaces(monkeypatch):
    from sr_adapter import relation_sources
    body = paragraph('Strict text')
    original = package(body)
    with zipfile.ZipFile(io.BytesIO(original)) as archive:
        parts = {name: archive.read(name).replace(W.encode(), b'http://purl.oclc.org/ooxml/wordprocessingml/main').replace(R.encode(), b'http://purl.oclc.org/ooxml/officeDocument/relationships') for name in archive.namelist()}
    assert read_source(package('', parts), 'strict.docx')['units'][0]['text'] == 'Strict text'
    with pytest.raises(ValueError, match='unsafe entry'):
        read_source(package(body, {'../escape.xml': '<x/>'}), 'bad.docx')
    document = f'<!DOCTYPE w:document [<!ENTITY x "secret">]><w:document xmlns:w="{W}"><w:body>{body}</w:body></w:document>'
    with pytest.raises(ValueError, match='document type'):
        read_source(package('', {'word/document.xml': document}), 'dtd.docx')
    monkeypatch.setattr(relation_sources, '_MAX_ZIP_BYTES', 20)
    with pytest.raises(ValueError, match='32 MiB'):
        read_source(original, 'bounded.docx')


def test_docx_runtime_request_relation_export_and_replay_use_existing_api(tmp_path):
    ontology = NarrativeOntology.model_validate({'namespace': 'urn:sr:docx-test:', 'classes': {'Project': 'Project', 'Team': 'Team'},
        'predicates': [{'id': 'managedBy', 'label': 'Managed by', 'subject_type': 'Project', 'object_type': 'Team', 'description': 'Responsible team'}]})
    path = tmp_path / 'source.docx'
    path.write_bytes(package(table([row([cell('Project'), cell('Owner')], '<w:tblHeader/>'), row([cell('A'), cell('B')])])))
    units = read_source(path.read_bytes(), path.name)['units']
    evidence = [{'unit_id': u['id'], 'quote': u['text']} for u in units]
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        raw = {'status': 'completed', 'model': 'gpt-5.5', 'output': [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': json.dumps({'assertions': [{'subject': 'A', 'predicate': 'managedBy', 'object': 'B', 'status': 'asserted', 'evidence': evidence}]})}]}]}
        return httpx.Response(200, json=raw)
    result = extract_narrative(path, ontology=ontology, endpoint='https://unit.services.ai.azure.com/api/projects/test', model='gpt-5.5', destination=tmp_path / 'run', execute=True,
        credential=SimpleNamespace(get_token=lambda _: SimpleNamespace(token='fake')), transport=httpx.MockTransport(handler))
    assert result['status'] == 'complete'
    assert result['planned_model_requests'] == len(requests) == 1
    assert 'header_context' in requests[0]['input'][1]['content'][0]['text']
    manifest, files = verify_narrative(tmp_path / 'run/bundle')
    assert manifest['assertion_count'] == 1
    exported = next(csv.DictReader(io.StringIO(files['relations.csv'].decode())))
    assert (exported['subject'], exported['object']) == ('A', 'B')
    assert json.loads(exported['evidence_json'])[-1]['locator']['cell']['column'] == 2


def test_docx_cli_prepares_offline_request_without_format_specific_flag(tmp_path, capsys):
    from sr_adapter.cli import main
    path = tmp_path / 'source.docx'
    path.write_bytes(package(paragraph('A is managed by B.')))
    ontology = tmp_path / 'ontology.json'
    ontology.write_text(json.dumps({'namespace': 'urn:sr:docx-test:', 'classes': {'Project': 'Project', 'Team': 'Team'},
        'predicates': [{'id': 'managedBy', 'label': 'Managed by', 'subject_type': 'Project', 'object_type': 'Team', 'description': 'Responsible team'}]}))
    out = tmp_path / 'planned'
    assert main(['knowledge', 'narrative', str(path), '--ontology', str(ontology), '--project-endpoint',
                 'https://unit.services.ai.azure.com/api/projects/test', '--model', 'gpt-5.5', '--out', str(out)]) == 0
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['status'] == 'prepared' and receipt['requests'] == []
    assert receipt['planned_model_requests'] == 1
    assert 'prepared' in capsys.readouterr().out
