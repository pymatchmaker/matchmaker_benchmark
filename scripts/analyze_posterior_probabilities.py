import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import rel_entr


def summarize(run_dir):
    completion = json.loads((run_dir / 'completion.json').read_text())
    if not completion['complete']:
        raise ValueError('Posterior analysis requires a complete run')
    pieces = []
    for result_path in sorted(run_dir.glob('*.json')):
        if not result_path.stem.isdigit():
            continue
        trace = np.loadtxt(run_dir / f'imm_{result_path.stem}.tsv', skiprows=1, ndmin=2)
        prior, posterior = trace[:, 4:7], trace[:, 7:10]
        np.testing.assert_allclose(prior.sum(axis=1), 1.)
        np.testing.assert_allclose(posterior.sum(axis=1), 1.)
        np.testing.assert_array_equal(posterior[trace[:, 3] == 0, 2], 0.)
        pieces.append((result_path.stem, json.loads(result_path.read_text())['tracked'], trace))
    if len(pieces) != completion['completed']:
        raise ValueError('Diagnostic count differs from completed performance count')
    contexts = {}
    for allowed in (False, True):
        groups = [trace[trace[:, 3] == allowed] for _, _, trace in pieces]
        nonempty = [group for group in groups if len(group)]
        if not nonempty:
            continue
        values = np.concatenate(nonempty)
        prior, posterior = values[:, 4:7], values[:, 7:10]
        contexts['zv_permitted' if allowed else 'zv_excluded'] = {
            'performances': len(nonempty), 'frames': len(values),
            'mean_prior': prior.mean(axis=0).tolist(),
            'mean_posterior': posterior.mean(axis=0).tolist(),
            'mean_update_kl_nats': float(rel_entr(posterior, prior).sum(axis=1).mean()),
        }
    return {'performances': len(pieces), 'tracked': sum(tracked for _, tracked, _ in pieces),
            'aggregation': 'frame-pooled; all performances, including failures',
            'context': 'ZV eligibility under the filter configuration; not ground-truth musical classes',
            'contexts': contexts}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    report = summarize(args.run_dir)
    (args.run_dir / 'posterior_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
