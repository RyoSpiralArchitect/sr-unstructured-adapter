import io
import json
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sr_adapter.cli import main
from sr_adapter.fabric import FabricTarget, verify_fabric
from sr_adapter.fabric_runtime import fabric_status, publish_fabric
from sr_adapter.fabric_tabular import export_tabular
from sr_adapter.knowledge import sha256
from sr_adapter.tables import canonical_json
from test_fabric import FabricFixture

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def box(identity, text):
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{identity}" name="box"/></p:nvSpPr>'
            '<p:spPr><a:xfrm><a:off x="-10" y="20"/><a:ext cx="100" cy="200"/></a:xfrm></p:spPr>'
            f'<p:txBody><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp>')


def connector(identity, start=None, end=None):
    refs = ''.join(f'<a:{tag} id="{value}" idx="2"/>' for tag, value in (("stCxn", start), ("endCxn", end)) if value)
    return (f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{identity}" name="line"/>'
            f'<p:cNvCxnSpPr>{refs}</p:cNvCxnSpPr></p:nvCxnSpPr>'
            '<p:spPr><a:ln><a:headEnd type="triangle"/><a:tailEnd/></a:ln></p:spPr></p:cxnSp>')


def deck(tmp_path, *, duplicate=False, external=False):
    group = ('<p:grpSp><p:nvGrpSpPr><p:cNvPr id="10" name="group"/></p:nvGrpSpPr>'
             '<p:grpSpPr><a:xfrm><a:off x="1" y="2"/><a:ext cx="30" cy="40"/>'
             '<a:chOff x="0" y="0"/><a:chExt cx="3" cy="4"/></a:xfrm></p:grpSpPr>'
             + box("3", 'Ops &amp; 支援') + '</p:grpSp>')
    table = ('<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="8" name="table"/></p:nvGraphicFramePr>'
             '<a:graphic><a:graphicData><a:tbl><a:tr><a:tc gridSpan="2"><a:txBody><a:p><a:r><a:t>0</a:t>'
             '</a:r></a:p></a:txBody></a:tc><a:tc hMerge="1"><a:txBody><a:p/></a:txBody></a:tc>'
             '</a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>')
    bodies = [box("2", "A") + group + connector("4", "2", "3") + connector("5")
              + connector("6", "2") + connector("7", "2", "999") + table,
              box("2", "UNSELECTED PRIVATE PAGE"), box("2", "C")]
    if duplicate:
        bodies[0] += box("2", "duplicate")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('ppt/presentation.xml', f'<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst>'
            + ''.join(f'<p:sldId id="{n}" r:id="r{n}"/>' for n in (1, 2, 3))
            + '</p:sldIdLst><p:sldSz cx="1000" cy="700"/></p:presentation>')
        archive.writestr('ppt/_rels/presentation.xml.rels',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + ''.join(f'<Relationship Id="r{n}" Type="{R}/slide" Target="slides/slide{n}.xml"'
                      + (' TargetMode="External"' if external and n == 1 else '') + '/>' for n in (1, 2, 3))
            + '</Relationships>')
        for n, body in enumerate(bodies, 1):
            archive.writestr(f'ppt/slides/slide{n}.xml', f'<p:sld xmlns:p="{P}" xmlns:a="{A}" show="{int(n != 1)}">'
                             f'<p:cSld><p:spTree>{body}</p:spTree></p:cSld></p:sld>')
    source = tmp_path / 'input.pptx'
    source.write_bytes(stream.getvalue())
    return source


@pytest.fixture
def target():
    return FabricTarget(workspace_id="11111111-1111-1111-1111-111111111111",
                        lakehouse_id="22222222-2222-2222-2222-222222222222", lakehouse_name="test")


@pytest.fixture
def exported(tmp_path, target):
    folder = tmp_path / 'tabular'
    export_tabular(deck(tmp_path), folder, target=target, slides=[1, 3], page_labels=['56', '57'])
    return folder


def read(files, kind):
    return pq.read_table(pa.BufferReader(files[f'tables/{kind}.parquet'])).to_pylist()


def test_scalar_tables_keep_native_identity_and_unresolved_connections(exported, target):
    manifest, files, plan = verify_fabric(exported, target=target)
    pages = sorted(read(files, 'pages'), key=lambda p: p['slide'])
    assert [(p['slide'], p['page_label'], p['hidden']) for p in pages] == [(1, '56', True), (3, '57', False)]
    assert pages[0]['attached_count'] == pages[0]['partial_count'] == pages[0]['unattached_count'] == pages[0]['dangling_count'] == 1
    objects = {(r['slide'], r['shape_id']): r for r in read(files, 'objects')}
    assert objects[1, '3']['parent_object_id'] == objects[1, '10']['object_id']
    assert objects[1, '3']['text'] == 'Ops & 支援'
    assert objects[1, '3']['x_emu'] == -10  # Parent-local EMU, no invented global coordinates.
    assert objects[1, '2']['object_id'] != objects[3, '2']['object_id']
    edges = {r['endpoint_status']: r for r in read(files, 'connections')}
    assert edges['attached']['start_text'] == 'A' and edges['attached']['end_text'] == 'Ops & 支援'
    assert edges['attached']['head_arrow'] == 'triangle' and edges['attached']['tail_arrow'] == 'none'
    assert edges['partial']['end_object_id'] is None
    assert edges['dangling']['end_shape_id'] == '999' and edges['dangling']['end_object_id'] is None
    assert all(r['interpretation'] == 'native_connector_only' for r in edges.values())
    cells = read(files, 'cells')
    assert [c['text'] for c in cells] == ['0', ''] and cells[1]['horizontal_merge'] is True
    assert cells[0]['column_span'] == 2
    assert 'source/slide-2.xml' not in files and not any(name.endswith('.pptx') for name in files)
    assert not any(b'UNSELECTED PRIVATE PAGE' in data for data in files.values())
    for table in plan['tables']:
        schema = pq.read_table(pa.BufferReader(files[table['parquet_file']])).schema
        assert all(not name.endswith('_json') for name in schema.names)
        assert all(pa.types.is_string(f.type) or pa.types.is_integer(f.type) or pa.types.is_boolean(f.type) for f in schema)
    assert manifest['tables']['connections']['rows'] == 4


def test_tabular_publish_uses_existing_receipts_and_exact_delta_readback(exported, target, tmp_path):
    service = FabricFixture(target)
    receipt = tmp_path / 'receipt.json'
    result = publish_fabric(exported, target=target, receipt=receipt, **service.kwargs)
    assert result['status'] == 'pending' and len(service.loads) == 4
    service.state = 'Succeeded'
    assert fabric_status(exported, target=target, receipt=receipt, **service.kwargs)['status'] == 'committed'
    service.requests.clear()
    assert publish_fabric(exported, target=target, receipt=tmp_path / 'again.json', **service.kwargs)['status'] == 'committed'
    assert not any(r.method == 'POST' for r in service.requests)


def test_rehashed_table_tamper_does_not_pass_source_reconstruction(exported, target):
    path = exported / 'tables/connections.parquet'
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]['start_text'] = 'invented'
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    manifest = json.loads((exported / 'manifest.json').read_bytes())
    manifest['files']['tables/connections.parquet'] = {'bytes': path.stat().st_size, 'sha256': sha256(path.read_bytes())}
    (exported / 'manifest.json').write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError, match='Parquet differs'):
        verify_fabric(exported, target=target)


@pytest.mark.parametrize('slides,labels', [([], None), ([0], None), ([4], None), ([1, 1], None), ([1, 3], ['56'])])
def test_invalid_selection_rejects_before_creating_export(tmp_path, target, slides, labels):
    with pytest.raises(ValueError):
        export_tabular(deck(tmp_path), tmp_path / 'out', target=target, slides=slides, page_labels=labels)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('kwargs', [{'duplicate': True}, {'external': True}])
def test_ambiguous_shape_or_external_selected_slide_rejects(tmp_path, target, kwargs):
    with pytest.raises(ValueError):
        export_tabular(deck(tmp_path, **kwargs), tmp_path / 'out', target=target, slides=[1])


def test_cli_tabular_export_and_explicit_publication_gate(tmp_path, target, capsys):
    source = deck(tmp_path)
    config = tmp_path / 'target.json'
    config.write_text(target.model_dump_json())
    args = ['knowledge', 'fabric', str(source), '--mode', 'tabular', '--slides', '1', '3', '--page-labels', '56', '57',
            '--target', str(config), '--out', str(tmp_path / 'out')]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'complete'
    assert main(['knowledge', 'fabric-publish', str(tmp_path / 'out'), '--target', str(config),
                 '--receipt', str(tmp_path / 'receipt.json')]) == 1
    assert not (tmp_path / 'receipt.json').exists()


def test_source_selection_path_cannot_escape_export(exported, target):
    path = exported / 'manifest.json'
    manifest = json.loads(path.read_bytes())
    manifest['files']['../outside'] = {}
    path.write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError, match='escapes'):
        verify_fabric(exported, target=target)


def test_empty_tables_keep_schemas_without_cloud_load(tmp_path, target):
    folder = tmp_path / 'out'
    export_tabular(deck(tmp_path), folder, target=target, slides=[3])
    manifest, files, plan = verify_fabric(folder, target=target)
    assert manifest['tables']['connections']['rows'] == manifest['tables']['cells']['rows'] == 0
    assert read(files, 'connections') == read(files, 'cells') == []
    assert {table['kind'] for table in plan['tables']} == {'pages', 'objects'}
