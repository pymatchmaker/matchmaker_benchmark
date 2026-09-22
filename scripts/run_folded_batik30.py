"""Evaluate any audio method on the paper's 30 folded Batik repeats (unfold_score=False).

Inputs (cohort and audited folded ground truth) are the frozen copies under
output/final_imm_hierarchical_reproduction_20260921/protocol; the follower is
whatever `matchmaker` resolves to on PYTHONPATH.
"""
import argparse
import contextlib
import hashlib
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
PROTOCOL = BENCH / 'output/final_imm_hierarchical_reproduction_20260921/protocol'
sys.path.insert(0, str(BENCH / 'matchmaker_eval'))
from verify_tracking import check_tracking  # noqa: E402


def run_case(args):
    case, method, out = args
    from matchmaker import Matchmaker
    title = case['title']
    dest = Path(out) / title
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'run.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            row = case['row']
            data = Path.home() / 'data/batik_plays_mozart'
            score, audio = data / row['xml_score'], data / row['audio_performance']
            start = time.perf_counter()
            mm = Matchmaker(score_file=score, performance_file=audio, method=method,
                            input_type='audio', unfold_score=False, wait=False)
            setup_seconds = time.perf_counter() - start
            start = time.perf_counter()
            for _ in mm.run(verbose=False):
                pass
            run_seconds = time.perf_counter() - start
            wp = mm.score_follower.alignment_path.T
            gt = np.loadtxt(PROTOCOL / 'folded_gt' / f'{case["index"]:03d}_{title}.tsv', skiprows=1)[:, [0, 2]]
            np.savetxt(dest / 'wp.tsv', wp, delimiter='\t', header='perf_sec\tscore_beat', comments='')
            np.savetxt(dest / 'gt.tsv', gt, delimiter='\t', header='perf_sec\tscore_beat', comments='')
            tracking = check_tracking(wp, gt, threshold=2.0, segment_duration=30.0, min_fails=1)
            indices = np.searchsorted(wp[:, 0], gt[:, 0], side='right') - 1
            errors = np.full(len(gt), np.inf)
            valid = indices >= 0
            errors[valid] = abs(wp[indices[valid], 1] - gt[valid, 1])
            result = dict(title=title, index=case['index'], tracked=bool(tracking['tracked']),
                          max_deviation=tracking['max_deviation'], n_failed=tracking['n_failed'],
                          mean=float(errors.mean()), median=float(np.median(errors)),
                          le_05=float(np.mean(errors <= .5)), le_1=float(np.mean(errors <= 1)),
                          events=len(errors), frames=len(wp), setup_seconds=setup_seconds,
                          run_seconds=run_seconds, rtf=run_seconds / (gt[-1, 0] - gt[0, 0]),
                          backward_outputs=int(np.sum(np.diff(wp[:, 1]) < -2)),
                          score_sha256=hashlib.sha256(score.read_bytes()).hexdigest(),
                          method=method, unfold_score=False,
                          source=sys.modules['matchmaker'].__file__)
            (dest / 'result.json').write_text(json.dumps(result, indent=2))
            return result
        except Exception:
            error = traceback.format_exc()
            (dest / 'error.txt').write_text(error)
            return {'title': title, 'error': error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--titles', nargs='*', help='subset of piece titles')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads((PROTOCOL / 'folded_cases.json').read_text())
    if args.titles:
        cases = [c for c in cases if c['title'] in args.titles]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = []
        for r in pool.map(run_case, [(c, args.method, str(out)) for c in cases]):
            results.append(r)
            print(json.dumps(r), flush=True)
    ok = [r for r in results if 'error' not in r]
    tracked = sum(r['tracked'] for r in ok)
    summary = {'method': args.method, 'expected': len(cases), 'completed': len(ok),
               'tracked': tracked, 'errors': [r['title'] for r in results if 'error' in r]}
    if ok:
        w = np.array([r['events'] for r in ok if r['tracked']], float)
        tr = [r for r in ok if r['tracked']]
        if tr:
            summary['tracked_mean'] = float(sum(r['mean'] * r['events'] for r in tr) / w.sum())
            summary['tracked_le_1'] = float(sum(r['le_1'] * r['events'] for r in tr) / w.sum())
            summary['tracked_le_05'] = float(sum(r['le_05'] * r['events'] for r in tr) / w.sum())
        summary['rtf_mean'] = float(np.mean([r['rtf'] for r in ok]))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
