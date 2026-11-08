"""Frozen diagnostic suites, executed and scored with the benchmark evaluator.

The data manifest records every input hash. No paths from a developer's machine
are used at evaluation time. Jobs fail on missing inputs or incomplete results.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'matchmaker_eval'):
    sys.path.insert(0, str(p))

CONFIGS = {
    **{f'perturbations-{mode}-{m}': dict(suite='perturbations', input_type=mode, method=m)
       for mode, methods in [('audio', ['dixon', 'arzt', 'outerhmm', 'skf']),
                             ('midi', ['pthmm', 'hmm', 'outerhmm', 'dixon', 'arzt', 'SLT_OLTW'])]
       for m in methods},
    **{f'repeat_examples-audio-{m}': dict(suite='repeat_examples', input_type='audio', method=m)
       for m in ['arzt', 'arzt_multi_fold', 'outerhmm']},
    **{f'folded_batik-audio-{m}': dict(suite='folded_batik', input_type='audio', method=m,
           kwargs={'s_j': 1e-5} if m == 'outerhmm' else {})
       for m in ['dixon', 'arzt', 'skf', 'outerhmm', 'arzt_multi_fold']},
    **{f'global_skip-midi-{state}': dict(suite='global_skip', input_type='midi', method='outerhmm', disable_global_skip=state == 'off')
       for state in ['on', 'off']},
}


def json_value(value):
    """Retain model-class identity when serializing effective settings."""
    if isinstance(value, type):
        return f'{value.__module__}.{value.__qualname__}'
    return float(value)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, path.open('wb') as out:
                import shutil
                shutil.copyfileobj(response, out)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def materialize(rel, manifest, root, base_url):
    """Fetch a frozen file, reassembling large lossless audio when necessary."""
    meta = manifest['files'][rel]
    dest = root / rel
    if dest.exists() and sha(dest) == meta['sha256']:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if meta.get('parts'):
        import shutil
        with dest.open('wb') as out:
            for part in meta['parts']:
                local = root / part['path']
                if not local.exists() or sha(local) != part['sha256']:
                    download(base_url + part['path'], local)
                if sha(local) != part['sha256']:
                    raise ValueError(f'Corrupt data chunk: {part["path"]}')
                with local.open('rb') as f:
                    shutil.copyfileobj(f, out)
    else:
        download(base_url + rel, dest)
    if sha(dest) != meta['sha256']:
        raise ValueError(f'Input hash mismatch: {rel}')
    return dest


def select_cases(manifest, config, shard=0, shards=1):
    if not 1 <= shards <= 32 or not 0 <= shard < shards:
        raise ValueError('Invalid shard selection')
    cases = [c for c in manifest['cases'] if c['suite'] == config['suite'] and c['input_type'] == config['input_type']]
    ids = [c['id'] for c in cases]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicated diagnostic cases')
    return [(i, c) for i, c in enumerate(cases) if i % shards == shard]


def run_case(case, config, paths, out):
    import numpy as np
    from matchmaker import Matchmaker
    from matchmaker.matchmaker import DEFAULT_KWARGS
    from matchmaker.utils.eval import resolve_gt
    from matchmaker_eval.eval import run_evaluation
    from matchmaker_eval.run_submission import check_follower, piece_time_budget
    from matchmaker_eval.score_input import prepared_score_path
    from matchmaker_eval.utils import TOLERANCES_IN_BEATS
    from matchmaker_eval.beat_units import TRACKING_THRESHOLDS, score_meter_map, musical_beat_factors
    from matchmaker_eval.verify_tracking import check_tracking

    mode, method = config['input_type'], config['method']
    kwargs = dict(DEFAULT_KWARGS[mode][method])
    # The diagnosis tests jumps; the play-through audio registry disables them.
    if mode == 'audio' and method == 'outerhmm':
        kwargs['s_j'] = 1e-100
    kwargs.update(config.get('kwargs', {}))
    out.mkdir(parents=True, exist_ok=True)
    with piece_time_budget(2400):
        with prepared_score_path(paths['xml_score']) as (score, remapping):
            mm = Matchmaker(score_file=str(score), performance_file=str(paths[f'{mode}_performance']),
                            method=method, input_type=mode, wait=False,
                            unfold_score=case['unfold_score'], tempo=case['tempo'], kwargs=kwargs)
        check_follower(mm.score_follower)
        if config.get('disable_global_skip'):
            # S is the skip-from distribution used at every OPHMM update.
            # Zero it without changing pitch observations or local transitions.
            mm.score_follower.S[:] = 0.
        for _ in mm.run(verbose=False):
            pass
        t, b = resolve_gt(paths['match'], mm.score_part.note_array(),
                          performance=str(paths[f'{mode}_performance']) if mode == 'midi' else None)
        gt = np.column_stack((t, b))
        metrics = run_evaluation(mm, gt=gt, tolerances=TOLERANCES_IN_BEATS, domain='score',
                                 debug=True, save_dir=out, run_name='0', make_plot=False, plot_dist_matrix=False)
        wp = mm.score_follower.alignment_path.T
        meter = score_meter_map(mm.score_part)
        tracking = check_tracking(wp, gt, threshold=TRACKING_THRESHOLDS[mode], segment_duration=30.,
                                  beat_factors=musical_beat_factors(gt[:, 1], meter))
    (out/'meter_0.json').write_text(json.dumps(meter, indent=2)+'\n')
    # Save directly too: the full path and all annotation points are required.
    for name, array in [('wp_0', wp), ('gt_0', gt)]:
        np.savetxt(out/(name+'.tsv'), array, delimiter='\t', fmt='%.12g', header='perf_sec\tscore_beat', comments='')
    wp = wp[np.argsort(wp[:, 0], kind='stable')]
    ix = np.searchsorted(wp[:, 0], gt[:, 0], side='right')-1
    valid = ix >= 0
    error = np.full(len(gt), np.nan)
    error[valid] = np.abs(gt[valid, 1]-wp[ix[valid], 1])
    intervals = {}
    if 'repeat_metadata' in case:
        m = case['repeat_metadata']
        masks = {'before': gt[:, 0] < m['replay_start_sec'],
                 'replay': (gt[:, 0] >= m['replay_start_sec']) & (gt[:, 0] < m['replay_end_sec']),
                 'after': gt[:, 0] >= m['replay_end_sec']}
        for name, mask in masks.items():
            intervals[name] = dict(onsets=int(mask.sum()), within_1_beat=float(np.mean(error[mask] <= 1)),
                                   median_beat_error=float(np.nanmedian(error[mask])))
    return dict(case_id=case['id'], category=case['category'], method=method, input_type=mode,
                unfold_score=case['unfold_score'], tempo=case['tempo'], kwargs=kwargs,
                disable_global_skip=config.get('disable_global_skip', False),
                tracked=bool(tracking['tracked']), max_deviation=float(tracking['max_deviation']),
                tracking_criterion={k: tracking[k] for k in ('error_unit', 'threshold', 'window_duration')},
                mean_b=float(np.nanmean(error)), n_events=int(valid.sum()), n_annotated_onsets=len(gt),
                metrics=metrics, intervals=intervals, score_voice_id_remapping=remapping)


def run(args):
    import contextlib
    import traceback
    from matchmaker_eval.run_submission import environment_info
    pin = json.loads((ROOT/'data/diagnostics.json').read_text())
    base = f'https://raw.githubusercontent.com/{pin["repository"]}/{pin["commit"]}/diagnostics/'
    args.cache.mkdir(parents=True, exist_ok=True)
    mp = args.cache/'manifest.json'
    if not mp.exists() or sha(mp) != pin['manifest_sha256']:
        download(base+'manifest.json', mp)
    if sha(mp) != pin['manifest_sha256']:
        raise ValueError('Diagnostic manifest hash mismatch')
    manifest = json.loads(mp.read_text())
    config = CONFIGS[args.config]
    selected = select_cases(manifest, config, args.shard, args.shards)
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(config=args.config, definition=config, shard=args.shard, shards=args.shards,
                  data=pin, environment=environment_info(), github_run_id=os.environ.get('GITHUB_RUN_ID'),
                  github_run_attempt=os.environ.get('GITHUB_RUN_ATTEMPT'), expected=[c['id'] for _, c in selected], pieces=[], failures=[])
    for i, case in selected:
        dest = args.output/f'{i:03d}'
        dest.mkdir(exist_ok=True)
        try:
            keys = ['xml_score', 'match', config['input_type']+'_performance']
            paths = {k: materialize(case['paths'][k], manifest, args.cache, base) for k in keys}
            with (dest/'run.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                record = run_case(case, config, paths, dest)
            record['input_hashes'] = {k: manifest['files'][case['paths'][k]]['sha256'] for k in keys}
            record['directory'] = dest.name
            (dest/'result.json').write_text(json.dumps(record, indent=2, default=json_value)+'\n')
            result['pieces'].append(record)
            print(args.config, case['id'], record['tracked'], record['mean_b'], flush=True)
        except Exception:
            error = traceback.format_exc()
            (dest/'error.txt').write_text(error)
            result['failures'].append(dict(case_id=case['id'], error=error))
            print(error, flush=True)
        (args.output/'metrics.json').write_text(json.dumps(result, indent=2, default=json_value)+'\n')
    # Empty shards are valid when a suite contains fewer cases than shards.
    (args.output/'metrics.json').write_text(json.dumps(result, indent=2, default=json_value)+'\n')
    if result['failures']:
        raise SystemExit(f'{len(result["failures"])} diagnostic cases failed')


def merge(args):
    import shutil
    from collections import defaultdict
    import numpy as np
    manifest = json.loads(args.manifest.read_text())
    records = [json.loads(p.read_text()) for p in args.source.glob('*/metrics.json')]
    summary = {}; errors = []
    args.output.mkdir(parents=True, exist_ok=True)
    for name, config in CONFIGS.items():
        rs = [r for r in records if r['config'] == name]
        expected = {c['id'] for _, c in select_cases(manifest, config)}
        found = [p['case_id'] for r in rs for p in r['pieces']]
        if len(rs) != args.shards or {r['shard'] for r in rs} != set(range(args.shards)) or set(found) != expected or len(found) != len(expected) or any(r['failures'] for r in rs):
            errors.append(name)
        cells = defaultdict(list)
        for r in rs:
            for p in r['pieces']:
                cells[p['category']].append(p)
        summary[name] = {category: dict(total=len(ps), tracked=sum(p['tracked'] for p in ps),
                            mean_b_tracked=float(np.mean([p['mean_b'] for p in ps if p['tracked']])) if any(p['tracked'] for p in ps) else None,
                            mean_b_all=float(np.mean([p['mean_b'] for p in ps]))) for category, ps in cells.items()}
    provenance = {(r['data']['commit'], r['environment']['benchmark_commit'], r['environment'].get('matchmaker_commit'), r['github_run_id']) for r in records}
    if len(provenance) != 1:
        errors.append('inconsistent provenance')
    for src in args.source.iterdir():
        if src.is_dir():
            shutil.copytree(src, args.output/'raw'/src.name, dirs_exist_ok=True)
    shutil.copyfile(args.manifest, args.output/'data_manifest.json')
    (args.output/'summary.json').write_text(json.dumps(dict(complete=not errors, incomplete=errors, configurations=summary,
        aggregation='For perturbations: arithmetic mean of per-condition MeanAE over tracked conditions; single cases also retain all-case error.',
        provenance=[list(p) for p in provenance]), indent=2)+'\n')
    if errors:
        raise SystemExit('Incomplete diagnostic results: '+', '.join(errors))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--config', choices=CONFIGS, required=True)
    r.add_argument('--shard', type=int, default=0)
    r.add_argument('--shards', type=int, default=1)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--cache', type=Path, default=Path('diagnostic-data'))
    m = sub.add_parser('merge')
    m.add_argument('--source', type=Path, required=True)
    m.add_argument('--output', type=Path, required=True)
    m.add_argument('--manifest', type=Path, required=True)
    m.add_argument('--shards', type=int, required=True)
    args = p.parse_args()
    (run if args.command == 'run' else merge)(args)

if __name__ == '__main__':
    main()
