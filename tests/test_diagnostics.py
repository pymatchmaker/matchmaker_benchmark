import hashlib
import json
from argparse import Namespace
from pathlib import Path
import pytest
from matchmaker_eval.diagnostics import CONFIGS, select_cases, materialize, merge


def test_sharding_is_exhaustive_and_disjoint():
    m = {'cases': [dict(id=str(i), suite='perturbations', input_type='audio') for i in range(41)]}
    config = CONFIGS['perturbations-audio-arzt']
    shards = [select_cases(m, config, s, 4) for s in range(4)]
    assert sorted(i for batch in shards for i, _ in batch) == list(range(41))
    assert max(map(len, shards))-min(map(len, shards)) == 1


def test_duplicate_cases_rejected():
    c = dict(id='a', suite='perturbations', input_type='audio')
    with pytest.raises(ValueError, match='duplicated'):
        select_cases({'cases':[c,c]}, CONFIGS['perturbations-audio-arzt'])


def test_chunked_input_verified(tmp_path):
    chunks = [b'first', b'second']
    parts = []
    for i, data in enumerate(chunks):
        p = tmp_path/f'p{i}'; p.write_bytes(data)
        parts.append(dict(path=p.name, sha256=hashlib.sha256(data).hexdigest()))
    m={'files':{'audio.flac':dict(parts=parts,sha256=hashlib.sha256(b''.join(chunks)).hexdigest())}}
    out=materialize('audio.flac',m,tmp_path,'https://invalid/')
    assert out.read_bytes() == b'firstsecond'
    m['files']['audio.flac']['sha256']='wrong'
    with pytest.raises(ValueError,match='hash mismatch'):
        materialize('audio.flac',m,tmp_path,'https://invalid/')


def test_merge_fails_incomplete_and_preserves_evidence(tmp_path):
    source=tmp_path/'source';source.mkdir()
    manifest=tmp_path/'manifest.json'
    cases=[dict(id=k,suite=c['suite'],input_type=c['input_type']) for k,c in CONFIGS.items()]
    # Unique IDs per suite/modality are enough; all output shards are absent.
    manifest.write_text(json.dumps({'cases':cases}))
    with pytest.raises(SystemExit,match='Incomplete'):
        merge(Namespace(source=source,output=tmp_path/'out',manifest=manifest,shards=4))
    assert not json.loads((tmp_path/'out/summary.json').read_text())['complete']


def test_effective_hmm_settings_can_be_saved():
    import numpy as np
    from matchmaker.matchmaker import DEFAULT_KWARGS
    from matchmaker_eval.diagnostics import json_value
    settings = dict(DEFAULT_KWARGS['midi']['hmm'])
    record = {'kwargs': settings, 'mean_b': np.float32(0.25)}
    saved = json.loads(json.dumps(record, default=json_value))
    assert saved['kwargs']['tempo_model'] == 'matchmaker.utils.tempo_models.KalmanTempoModel'
    assert saved['kwargs']['piano_range'] is True
    assert saved['mean_b'] == 0.25
    # Saving metadata must not replace the model class in the live settings.
    assert isinstance(settings['tempo_model'], type)
