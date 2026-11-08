"""Run MIDI piano OPHMM with or without global transitions, without publishing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'matchmaker_eval'):
    sys.path.insert(0, str(path))

from matchmaker import Matchmaker
from matchmaker_eval import run_submission as runner


def controlled_factory(global_transition, observations, base=Matchmaker):
    if global_transition not in ('on', 'off'):
        raise ValueError('global_transition must be on or off')

    def build(*args, **kwargs):
        if kwargs.get('method') != 'outerhmm' or kwargs.get('input_type') != 'midi':
            raise ValueError('This comparison is restricted to MIDI outerhmm')
        mm = base(*args, **kwargs)
        follower = mm.score_follower
        if global_transition == 'off':
            follower.S[:] = 0.0
        observations.append({
            'performance': str(kwargs['performance_file']),
            'n_states': int(follower.n_states),
            'global_transition': global_transition,
            'S_nonzero': int((follower.S != 0).sum()),
            'S_sum': float(follower.S.sum()),
        })
        return mm
    return build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--global-transition', choices=['on', 'off'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard', type=int)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--limit', type=int)
    p.add_argument('--no-fetch', action='store_true')
    a = p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise SystemExit('Use a new, empty output directory to preserve previous results')
    observations = []
    with patch.object(runner, 'Matchmaker', controlled_factory(a.global_transition, observations)):
        result = runner.evaluate_submission(builtin='outerhmm', input_type='midi', fold='eval',
            run_dir=a.output, shard=a.shard, num_shards=a.num_shards,
            limit=a.limit, fetch=not a.no_fetch)
    experiment = {
        'global_transition': a.global_transition,
        'changed_parameter': 'S[:] = 0' if a.global_transition == 'off' else 'none',
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'github_run_id': os.environ.get('GITHUB_RUN_ID'),
        'github_sha': os.environ.get('GITHUB_SHA'),
        'followers': observations,
    }
    result['method'] = 'outerhmm-global-' + a.global_transition
    result['submission'] = result['method'] + '-midi'
    result['metadata']['name'] = 'OPHMM (global transitions ' + a.global_transition + ')'
    result['experiment'] = experiment
    (a.output / 'metrics.json').write_text(json.dumps(result, indent=2))
    (a.output / 'experiment.json').write_text(json.dumps(experiment, indent=2))
    if result['n_failed'] or len(observations) != result['n_pieces']:
        raise SystemExit('Some executions failed; inspect metrics.json')
    if a.global_transition == 'off' and any(x['S_nonzero'] for x in observations):
        raise SystemExit('Global transitions were not disabled')

if __name__ == '__main__':
    main()
