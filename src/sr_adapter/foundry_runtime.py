"""Explicit publication of a verified Foundry handoff to existing Azure resources."""

from __future__ import annotations

import json
from pathlib import Path
import uuid

import httpx

from .foundry import FoundryTarget, verify_foundry
from .knowledge import sha256
from .knowledge_store import _reserve_receipt, atomic_json


def _contains(actual, expected):
    """Allow service-added metadata but never silently replace different settings."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and _contains(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(_contains(a, b) for a, b in zip(actual, expected))
    return actual == expected


def publish_foundry(folder: str | Path, *, target: FoundryTarget, receipt: str | Path,
                    credential=None, transport=None) -> dict:
    """Create matching named configuration and upload deterministic search records.

    There is no resource-group/service provisioning, role assignment, rollback or
    automatic retry. Agent creation adds a version, so an ambiguous request must
    be reconciled before another explicit attempt. 'configured' does not mean
    that permission-dependent retrieval or model answering has been validated.
    """
    manifest, files = verify_foundry(folder)
    plan = json.loads(files['deployment-plan.json'])
    if not plan['configured'] or json.loads(files['target.json']) != target.model_dump(mode='json'):
        raise ValueError('Pass the explicit target that matches this configured export')
    report = {'destination': 'foundry', 'snapshot_id': manifest['snapshot_id'], 'status': 'running',
        'configured_target': target.model_dump(mode='json'), 'operations': [], 'retrieval_validation': 'not_run'}
    path = _reserve_receipt(receipt, report)
    owned = credential is None
    try:
        if owned:
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
        with httpx.Client(transport=transport, timeout=60, follow_redirects=False) as client:
            for operation in plan['operations']:
                row = {'name': operation['name'], 'method': operation['method'], 'status': 'running',
                    'request_id': str(uuid.uuid4()), 'request_sha256': sha256(files[operation['body_file']])}
                report['operations'].append(row)
                atomic_json(path, report)
                scope = ('https://management.azure.com/.default' if operation['name'] == 'connection' else
                         'https://ai.azure.com/.default' if operation['name'] == 'agent' else 'https://search.azure.com/.default')
                token = credential.get_token(scope).token
                headers = {'Authorization': 'Bearer ' + token, 'x-ms-client-request-id': row['request_id']}
                body = json.loads(files[operation['body_file']])
                if operation['method'] == 'PUT':
                    existing = client.get(operation['url'], headers=headers)
                    if existing.status_code != 404:
                        existing.raise_for_status()
                        expected = {'properties': body['properties']} if operation['name'] == 'connection' else body
                        if not _contains(existing.json(), expected):
                            row.update(status='conflict', http_status=existing.status_code)
                            report['status'] = 'conflict'
                            atomic_json(path, report)
                            return report
                        row.update(status='already_present_verified', http_status=existing.status_code)
                        atomic_json(path, report)
                        continue
                    # Search honors this creation condition. Connection names must
                    # also be dedicated: ARM does not promise this concurrency guard.
                    headers['If-None-Match'] = '*'
                response = client.request(operation['method'], operation['url'], json=body, headers=headers)
                row['http_status'] = response.status_code
                row['response_sha256'] = sha256(response.content)
                response.raise_for_status()
                # Accepted provisioning is not completed configuration.
                if response.status_code == 202:
                    row['status'], report['status'] = 'pending', 'pending'
                    atomic_json(path, report)
                    return report
                if operation['name'] == 'documents':
                    result = response.json().get('value', [])
                    expected_ids = {doc['id'] for doc in body['value']}
                    actual_ids = [item.get('key') for item in result]
                    if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
                        raise ValueError('Search indexing acknowledgement omitted or duplicated document keys')
                    row['documents'] = [{'id': item['key'], 'accepted': item.get('status') is True,
                                         'status_code': item.get('statusCode')} for item in result]
                    if not all(item['accepted'] and item['status_code'] in (200, 201) for item in row['documents']):
                        row['status'], report['status'] = 'partial', 'partial'
                        atomic_json(path, report)
                        return report
                    row['status'] = 'documents_accepted'
                elif operation['name'] == 'agent':
                    result = response.json()
                    # POST /agents returns AgentObject, not AgentVersionObject.
                    latest = result.get('versions', {}).get('latest', {})
                    if result.get('name') != target.agent_name or not isinstance(latest.get('version'), str) or not latest['version']:
                        raise ValueError('Agent response lacks the expected name and version')
                    report['agent_reference'] = {'type': 'agent_reference', 'name': result['name'], 'version': latest['version']}
                    row['status'] = 'created'
                else:
                    confirmed = client.get(operation['url'], headers={k: v for k, v in headers.items() if k != 'If-None-Match'})
                    confirmed.raise_for_status()
                    expected = {'properties': body['properties']} if operation['name'] == 'connection' else body
                    if not _contains(confirmed.json(), expected):
                        raise ValueError('Created configuration did not match on readback')
                    row['status'] = 'created_and_verified'
                atomic_json(path, report)
        report['status'] = 'configured'
    except httpx.HTTPStatusError as exc:
        report.update(status='failed' if 400 <= exc.response.status_code < 500 else 'unknown',
                      http_status=exc.response.status_code, error_type=type(exc).__name__)
        if report['operations']:
            report['operations'][-1]['status'] = report['status']
    except Exception as exc:
        report.update(status='unknown', error_type=type(exc).__name__)
        if report['operations']:
            report['operations'][-1]['status'] = report['status']
    finally:
        if owned and credential is not None:
            credential.close()
    atomic_json(path, report)
    return report
