import json
from pathlib import Path

import pytest

from matchmaker_eval import merge_references as merger


def test_non_piano_uses_its_own_destination():
    assert merger.output_for_fold('eval') == merger.DEFAULT_OUTPUT
    assert merger.output_for_fold('beyond_piano') == merger.REPO_ROOT / 'results/runs/beyond_piano'


def test_wrong_fold_is_rejected_before_merge(tmp_path):
    d = tmp_path / 'refshard-arzt-audio-0'
    d.mkdir()
    (d/'metrics.json').write_text(json.dumps({'fold':'beyond_piano'}))
    with pytest.raises(merger.MergeError, match="expected fold 'eval'"):
        merger.check_fold(tmp_path,'eval')
    merger.check_fold(tmp_path,'beyond_piano')


def test_a_wholly_missing_method_cannot_publish_a_smaller_benchmark(tmp_path):
    d = tmp_path / 'refshard-arzt-audio-0'
    d.mkdir()
    (d/'metrics.json').write_text(json.dumps({'fold':'eval'}))
    with pytest.raises(merger.MergeError, match="missing=\\['dixon-audio'\\]"):
        merger.check_expected_entries(tmp_path,['arzt-audio','dixon-audio'])
    merger.check_expected_entries(tmp_path,['arzt-audio'])


def test_workflows_keep_non_piano_out_of_the_piano_leaderboard():
    import yaml
    for name in ['evaluate.yml','evaluate-references.yml']:
        w = yaml.safe_load((Path(__file__).parents[1]/'.github/workflows'/name).read_text())
        steps = w['jobs']['merge']['steps']
        merger_step = next(s for s in steps if 'merge_references.py' in s.get('run',''))
        assert '--fold "$FOLD"' in merger_step['run']
        for label in ['Export the per-piece detail','Rebuild the leaderboard','Assemble the site for review','Commit the results']:
            assert next(s for s in steps if s.get('name')==label)['if'] == "env.FOLD == 'eval'"
        artifact = next(s for s in steps if s.get('name')=='Upload the complete merged results')
        assert artifact['with']['name']=='benchmark-results-${{ env.FOLD }}'


@pytest.mark.parametrize('fold,wanted,count,types', [
    ('eval','both',128,{'audio','midi'}),
    ('beyond_piano','both',48,{'audio'}),
    ('beyond_piano','audio',48,{'audio'}),
])
def test_reference_workflow_plans_only_available_modalities(monkeypatch, capsys, fold, wanted, count, types):
    import yaml
    root = Path(__file__).parents[1]
    workflow = yaml.safe_load((root/'.github/workflows/evaluate-references.yml').read_text())
    step = next(s for s in workflow['jobs']['plan']['steps'] if s.get('id')=='plan')
    program = step['run'].split("\n", 2)[2].rsplit('\nPY',1)[0]
    monkeypatch.chdir(root)
    for key,value in {'INPUT_TYPE':wanted,'ONLY':'','SHARDS':'8','FOLD':fold}.items():
        monkeypatch.setenv(key,value)
    exec(compile(program, '<reference plan>', 'exec'), {})
    values = dict(line.split('=',1) for line in capsys.readouterr().out.splitlines())
    jobs = json.loads(values['jobs'])
    assert len(jobs)==count and int(values['count'])==count
    assert {j['input_type'] for j in jobs}==types
    assert {j['shard'] for j in jobs}==set(range(8))
