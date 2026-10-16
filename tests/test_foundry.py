"""Relation context semantics and an offline Foundry handoff, with real source bundles."""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from sr_adapter.cli import main
from sr_adapter.foundry import FoundryTarget, export_foundry, relation_cards, verify_foundry
from sr_adapter.foundry_runtime import publish_foundry
from sr_adapter.knowledge import extract_knowledge, sha256
from sr_adapter.knowledge_store import export_bundle
from sr_adapter.tables import canonical_json, extract_pdf_tables


@pytest.fixture
def source_bundle(package, tmp_path):
    path = tmp_path / 'source-bundle'
    export_bundle(package, path)
    return path


@pytest.fixture
def foundry_target():
    return FoundryTarget(search_endpoint='https://fixture.search.windows.net',
        project_endpoint='https://fixture.services.ai.azure.com/api/projects/demo',
        project_resource_id='/subscriptions/fixture/resourceGroups/demo/providers/Microsoft.CognitiveServices/accounts/fixture/projects/demo',
        project_connection_name='knowledge-read-only', model_deployment='fixture-model')


def test_every_record_is_available_in_exactly_one_row_context(package):
    cards = relation_cards(package)
    assert len(cards) == 3
    emitted = []
    for card in cards:
        facts = json.loads(card['facts_json'])
        evidence = {c['id']: c for c in json.loads(card['evidence_json'])['cells']}
        assert facts['identity_scope'] == 'source_row_mention'
        for group in ('entities', 'relations', 'observations'):
            for record in facts[group]:
                emitted.append(record['id'])
                assert set(record['evidence']) <= set(evidence) | {card['table_id']}
    expected = [item.id for group in (package.entities, package.relations, package.observations) for item in group]
    assert sorted(emitted) == sorted(expected) and len(emitted) == len(set(emitted))
    assert '"Alpha" --ownedBy [source column "Owner"]--> "Ops"' in cards[0]['content']
    assert '"Ops" --ownedBy' not in cards[0]['content']
    assert 'UNKNOWN (no asserted value)' in cards[1]['content']
    assert 'unresolved entity --Team [source column "Owner"] = UNKNOWN' in cards[2]['content']
    for card in cards:
        facts = json.loads(card['facts_json'])
        for assertion in facts['relations'] + facts['observations']:
            assert 'assertion_id=' + assertion['id'] in card['content']


def test_equal_labels_do_not_merge_rows(pdf_factory, mapping):
    package = extract_knowledge(extract_pdf_tables(pdf_factory([
        ['Project', 'Owner', 'Budget'], ['A', 'Ops', '1'], ['B', 'Ops', '2']])), mapping)
    cards = relation_cards(package)
    owners = [[e['id'] for e in json.loads(c['facts_json'])['entities'] if e['label'] == 'Ops'][0] for c in cards]
    assert len(set(owners)) == 2 and cards[0]['id'] != cards[1]['id']


def test_spanning_values_are_audit_evidence_not_assigned_facts(pdf_factory, mapping, rows):
    package = extract_knowledge(extract_pdf_tables(pdf_factory(rows, spans=[((1, 1), (1, 2))])), mapping)
    cards = relation_cards(package)
    assert 'AMBIGUOUS' in cards[0]['content'] and 'AMBIGUOUS' in cards[1]['content']
    assert '--ownedBy' not in cards[0]['content'] and '--ownedBy' not in cards[1]['content']
    assert any(c['row_span'] == 2 for c in json.loads(cards[1]['evidence_json'])['cells'])


def test_foundry_export_has_search_fields_native_mcp_and_snapshot_filter(source_bundle, foundry_target, tmp_path):
    output = tmp_path / 'foundry'
    report = export_foundry([source_bundle], output, target=foundry_target)
    manifest, files = verify_foundry(output)
    assert report == manifest and report['card_count'] == 3
    assert report['schema_version'] == 2
    assert json.loads(files['project-connection.json'])['properties']['isSharedToAll'] is False
    index = json.loads(files['search-index.json'])
    fields = {f['name']: f for f in index['fields']}
    assert fields['id']['key'] and fields['content']['searchable']
    assert fields['facts_json']['retrievable'] and not fields['facts_json']['searchable']
    source = json.loads(files['knowledge-source.json'])['searchIndexParameters']
    assert source['baseFilter'] == "snapshot_id eq '" + manifest['snapshot_id'] + "'"
    assert {'facts_json', 'evidence_json', 'source_sha256', 'page'} <= {f['name'] for f in source['sourceDataFields']}
    base = json.loads(files['knowledge-base.json'])
    assert base['outputMode'] == 'extractiveData' and base['retrievalReasoningEffort'] == {'kind': 'minimal'}
    agent = json.loads(files['prompt-agent.json'])['definition']
    assert agent['kind'] == 'prompt' and agent['model'] == 'fixture-model'
    assert len(agent['tools']) == 1 and agent['tools'][0]['allowed_tools'] == ['knowledge_base_retrieve']
    assert agent['tools'][0]['server_url'].endswith('/mcp?api-version=2026-08-01-preview')
    assert agent['tools'][0]['project_connection_id'] == 'knowledge-read-only'
    assert json.loads(files['retrieval-request.json'])['knowledgeSourceParams'][0]['includeReferenceSourceData']
    with pytest.raises(FileExistsError):
        export_foundry([source_bundle], output, target=foundry_target)


def test_recorded_v1_handoff_remains_verifiable():
    folder = Path(__file__).resolve().parents[1] / 'docs/validation/foundry-context-2026-10-16/handoff'
    manifest, files = verify_foundry(folder)
    assert manifest['schema_version'] == 1 and manifest['card_count'] == 8
    assert json.loads(files['project-connection.json'])['properties']['isSharedToAll'] is True


def test_project_connection_readback_uses_service_persisted_scope(configured_export, foundry_target, tmp_path):
    service = AzureFixture()
    original_handle = service.handle

    def azure_project_response(request):
        response = original_handle(request)
        if '/connections/' in str(request.url) and response.status_code in (200, 201):
            body = response.json()
            body['properties']['isSharedToAll'] = False
            return httpx.Response(response.status_code, json=body)
        return response

    service.handle = azure_project_response
    result = publish_fixture(configured_export, foundry_target, tmp_path / 'receipt.json', service)
    assert result['status'] == 'configured'


def test_unconfigured_export_never_claims_a_deployment(source_bundle, tmp_path):
    output = tmp_path / 'foundry'
    assert main(['knowledge', 'foundry', str(source_bundle), '--out', str(output)]) == 0
    manifest, files = verify_foundry(output)
    assert not json.loads(files['deployment-plan.json'])['configured']
    assert manifest['cloud_validation'] == 'not_performed_by_export'
    assert json.loads(files['target.json']) is None


def test_rehashed_tamper_still_fails_semantic_verification(source_bundle, tmp_path):
    output = tmp_path / 'foundry'
    export_foundry([source_bundle], output)
    manifest = json.loads((output / 'manifest.json').read_bytes())
    path = output / 'prompt-agent.json'
    payload = json.loads(path.read_bytes())
    payload['definition']['instructions'] = 'Invent missing relations'
    data = canonical_json(payload) + b'\n'
    path.write_bytes(data)
    manifest['files']['prompt-agent.json'] = {'sha256': sha256(data), 'bytes': len(data)}
    (output / 'manifest.json').write_bytes(canonical_json(manifest))
    with pytest.raises(ValueError, match='does not reproduce'):
        verify_foundry(output)


def test_duplicate_packages_and_empty_mapping_results_do_not_create_output(source_bundle, package, tmp_path):
    output = tmp_path / 'foundry'
    with pytest.raises(ValueError, match='distinct'):
        export_foundry([source_bundle, source_bundle], output)
    assert not output.exists()
    payload = package.mapping.model_dump()
    payload['entities'][0]['column'] = 'Absent header'
    missing = extract_knowledge(package.source, type(package.mapping).model_validate(payload))
    bundle = tmp_path / 'missing'
    export_bundle(missing, bundle)
    with pytest.raises(ValueError, match='No mapped'):
        export_foundry([bundle], output)
    assert not output.exists()


@pytest.mark.parametrize('endpoint', ['http://fixture.search.windows.net', 'https://unexpected.example',
    'https://secret@fixture.search.windows.net', 'https://fixture.search.windows.net?api-key=secret'])
def test_target_does_not_accept_credential_bearing_or_non_search_endpoints(foundry_target, endpoint):
    with pytest.raises(ValueError):
        FoundryTarget.model_validate({**foundry_target.model_dump(), 'search_endpoint': endpoint})


def test_oversized_context_is_rejected_without_truncation(package):
    source = package.source.model_copy(deep=True)
    source.tables[0].cells[3].text = 'X' * 70000
    large = extract_knowledge(source, package.mapping)
    with pytest.raises(ValueError, match='64 KiB'):
        relation_cards(large)


def test_target_rejects_different_project_or_invalid_agent_name(foundry_target):
    for change in ({'project_endpoint': 'https://fixture.services.ai.azure.com/api/projects/other'},
                   {'agent_name': 'name_with_underscore'}, {'agent_name': 'name-'}, {'agent_name': 'a' * 64}):
        with pytest.raises(ValueError):
            FoundryTarget.model_validate({**foundry_target.model_dump(), **change})


class AzureFixture:
    """Offline HTTP contract fixture; these acknowledgements are not cloud evidence."""

    def __init__(self, fault=None):
        self.resources, self.requests, self.scopes = {}, [], []
        self.fault = fault
        self.agent_version = 0

    def get_token(self, scope):
        self.scopes.append(scope)
        return SimpleNamespace(token='fixture-secret')

    def handle(self, request):
        assert request.headers['authorization'] == 'Bearer fixture-secret'
        self.requests.append(request)
        if self.fault:
            response = self.fault(request)
            if response is not None:
                return response
        url = str(request.url)
        if request.method == 'GET':
            return httpx.Response(200, json=self.resources[url]) if url in self.resources else httpx.Response(404)
        body = json.loads(request.content)
        if request.method == 'PUT':
            assert request.headers['if-none-match'] == '*'
            assert url not in self.resources
            self.resources[url] = {**body, '@odata.etag': 'fixture'}
            return httpx.Response(201, json=self.resources[url])
        if '/docs/index?' in url:
            return httpx.Response(200, json={'value': [
                {'key': doc['id'], 'status': True, 'statusCode': 201} for doc in body['value']]})
        assert request.url.path.endswith('/agents')
        self.agent_version += 1
        return httpx.Response(200, json={'object': 'agent', 'name': body['name'],
            'versions': {'latest': {'name': body['name'], 'version': str(self.agent_version)}}})


@pytest.fixture
def configured_export(source_bundle, foundry_target, tmp_path):
    folder = tmp_path / 'foundry'
    export_foundry([source_bundle], folder, target=foundry_target)
    return folder


def publish_fixture(folder, target, receipt, service):
    result = publish_foundry(folder, target=target, receipt=receipt,
        credential=service, transport=httpx.MockTransport(service.handle))
    assert json.loads(receipt.read_bytes()) == result
    assert 'fixture-secret' not in receipt.read_text() and 'sensitive-response' not in receipt.read_text()
    return result


def test_publication_verifies_configuration_and_every_document_then_pins_agent_version(configured_export, foundry_target, tmp_path):
    service = AzureFixture()
    result = publish_fixture(configured_export, foundry_target, tmp_path / 'receipt.json', service)
    assert result['status'] == 'configured' and result['retrieval_validation'] == 'not_run'
    assert result['agent_reference'] == {'type': 'agent_reference', 'name': 'sr-relations', 'version': '1'}
    assert [row['status'] for row in result['operations']] == [
        'created_and_verified', 'documents_accepted', 'created_and_verified', 'created_and_verified',
        'created_and_verified', 'created']
    assert set(service.scopes) == {'https://search.azure.com/.default', 'https://management.azure.com/.default',
                                  'https://ai.azure.com/.default'}
    # A separately requested publication verifies existing resources, reuploads
    # deterministic IDs, and explicitly creates a new agent version.
    second = publish_fixture(configured_export, foundry_target, tmp_path / 'second.json', service)
    assert second['status'] == 'configured' and second['agent_reference']['version'] == '2'
    assert all(row['status'] == 'already_present_verified' for row in second['operations'] if row['method'] == 'PUT')
    assert len([r for r in service.requests if r.method == 'PUT']) == 4


def test_existing_different_configuration_stops_without_any_write(configured_export, foundry_target, tmp_path):
    service = AzureFixture(lambda request: httpx.Response(200, json={'name': 'foreign-index'})
                           if request.method == 'GET' else None)
    result = publish_fixture(configured_export, foundry_target, tmp_path / 'receipt.json', service)
    assert result['status'] == 'conflict'
    assert [r.method for r in service.requests] == ['GET']


@pytest.mark.parametrize('fault, expected', [('partial', 'partial'), ('missing', 'unknown'), ('duplicate', 'unknown'),
    ('timeout', 'unknown'), (401, 'failed'), (429, 'failed'), (503, 'unknown'), (202, 'pending')])
def test_indexing_failures_stop_before_knowledge_or_agent_creation(configured_export, foundry_target, tmp_path, fault, expected):
    def fail(request):
        if '/docs/index?' not in str(request.url):
            return None
        if fault == 'timeout':
            raise httpx.ReadTimeout('sensitive-response')
        if isinstance(fault, int):
            return httpx.Response(fault, text='sensitive-response')
        docs = json.loads(request.content)['value']
        values = [{'key': doc['id'], 'status': True, 'statusCode': 201} for doc in docs]
        if fault == 'partial':
            values[-1].update(status=False, statusCode=400, errorMessage='sensitive-response')
        elif fault == 'missing':
            values.pop()
        else:
            values[-1] = values[0]
        return httpx.Response(207, json={'value': values})
    service = AzureFixture(fail)
    result = publish_fixture(configured_export, foundry_target, tmp_path / 'receipt.json', service)
    assert result['status'] == expected and result['operations'][-1]['status'] == expected
    assert [row['name'] for row in result['operations']] == ['index', 'documents']
    assert sum(r.method == 'POST' for r in service.requests) == 1


def test_agent_timeout_has_one_attempt_and_no_false_version(configured_export, foundry_target, tmp_path):
    def fail(request):
        if request.url.path.endswith('/agents'):
            raise httpx.ReadTimeout('sensitive-response')
    service = AzureFixture(fail)
    result = publish_fixture(configured_export, foundry_target, tmp_path / 'receipt.json', service)
    assert result['status'] == 'unknown' and 'agent_reference' not in result
    assert sum(r.url.path.endswith('/agents') for r in service.requests) == 1
    assert result['operations'][-2]['status'] == 'created_and_verified'


def test_publication_requires_matching_target_explicit_execution_and_new_receipt(configured_export, foundry_target, tmp_path):
    receipt = tmp_path / 'receipt.json'
    target_file = configured_export / 'target.json'
    assert main(['knowledge', 'foundry-publish', str(configured_export), '--target', str(target_file),
                 '--receipt', str(receipt)]) == 1
    assert not receipt.exists()
    service = AzureFixture()
    other = FoundryTarget.model_validate({**foundry_target.model_dump(), 'search_endpoint': 'https://other.search.windows.net'})
    with pytest.raises(ValueError, match='explicit target'):
        publish_fixture(configured_export, other, receipt, service)
    assert not receipt.exists() and not service.requests
    receipt.write_text('existing receipt')
    with pytest.raises(FileExistsError):
        publish_fixture(configured_export, foundry_target, receipt, service)
    assert receipt.read_text() == 'existing receipt' and not service.requests and not service.scopes
