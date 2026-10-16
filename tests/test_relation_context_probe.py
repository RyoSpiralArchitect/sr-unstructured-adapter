"""Guard the context probe's evidence-aware score against false passes."""

import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('relation_probe', Path(__file__).resolve().parents[1] / 'scripts' / 'probe_relation_context.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_scoring_requires_correct_direction_value_and_record_evidence():
    gold = [dict(id='q', status='observed', value='Ops', evidence_ids=['relation-1'])]
    good = dict(id='q', status='observed', value='Ops', evidence_ids=['relation-1'])
    assert probe.score({'answers': [good]}, gold, ['relation-1'])[0]['strict']
    for change in [{'value': 'Alpha'}, {'status': 'unknown'}, {'evidence_ids': ['cell-1']},
                   {'evidence_ids': []}, {'evidence_ids': ['relation-1', 'invented']}]:
        assert not probe.score({'answers': [{**good, **change}]}, gold, ['relation-1'])[0]['strict']
    assert not probe.score({'answers': [good, good]}, gold, ['relation-1'])[0]['strict']
    assert not probe.score({'answers': [good, {'id': 'unrequested'}]}, gold, ['relation-1'])[0]['strict']


def test_unknown_is_not_a_negative_and_missing_answers_do_not_pass():
    gold = [dict(id='q', status='unknown', value=None, evidence_ids=['observation-1'])]
    good = dict(id='q', status='unknown', value=None, evidence_ids=['observation-1'])
    assert probe.score({'answers': [good]}, gold, ['observation-1'])[0]['strict']
    for payload in ({'answers': [{**good, 'status': 'not_established'}]}, {'answers': []}, None, {'answers': 'malformed'}):
        assert not probe.score(payload, gold, ['observation-1'])[0]['strict']
