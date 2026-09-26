"""Re-score saved folded-Batik runs (scripts/run_folded_batik30.py outputs) into the
manuscript's folded table: tracking rate, common-tracked and all-tracked errors.

Each method is scored at its own tracking threshold (beats) with the same criterion and
onset errors as run_folded_batik30.py, from the wp.tsv / gt.tsv it saved per piece.

  python scripts/report_folded_table.py --out output/table2_folded_20260925 \
      --run OPHMM=output/ophmm_batik24_20260923:2 \
      --run "Multi-Fold OLTW Arzt=output/arzt_multi_fold_folded30_20260922:1" \
      --run Ours=output/imm_graph_20260924_batik24:1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH / 'matchmaker_eval'))
from verify_tracking import check_tracking  # noqa: E402

TEST_TITLES = BENCH / 'output/imm_graph_20260924_batik24'  # the 24 test movements


def onset_errors(wp, gt):
    """|predicted - annotated| beat at every annotated onset (as in run_folded_batik30.py)."""
    indices = np.searchsorted(wp[:, 0], gt[:, 0], side='right') - 1
    errors = np.full(len(gt), np.inf)
    valid = indices >= 0
    errors[valid] = abs(wp[indices[valid], 1] - gt[valid, 1])
    return errors


def pooled(errors):
    e = np.concatenate(errors)
    return dict(mean=float(e.mean()), le_05=float(100 * np.mean(e <= .5)), le_1=float(100 * np.mean(e <= 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True, help='NAME=RUN_DIR:THRESHOLD_BEATS')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    titles = sorted(p.name for p in TEST_TITLES.iterdir() if p.is_dir())
    methods = {}
    for spec in args.run:
        name, rest = spec.split('=', 1)
        run_dir, threshold = rest.rsplit(':', 1)
        errors, tracked = {}, set()
        for title in titles:
            wp = np.loadtxt(BENCH / run_dir / title / 'wp.tsv', skiprows=1)
            gt = np.loadtxt(BENCH / run_dir / title / 'gt.tsv', skiprows=1)
            errors[title] = onset_errors(wp, gt)
            if check_tracking(wp, gt, threshold=float(threshold), segment_duration=30.0, min_fails=1)['tracked']:
                tracked.add(title)
        methods[name] = dict(run_dir=run_dir, threshold=float(threshold), errors=errors, tracked=tracked)
    common = sorted(set.intersection(*(m['tracked'] for m in methods.values())))
    report = dict(titles=titles, common_tracked=common, methods={})
    for name, m in methods.items():
        report['methods'][name] = dict(
            run_dir=m['run_dir'], threshold_beats=m['threshold'], tracked=sorted(m['tracked']),
            tracking_rate=100 * len(m['tracked']) / len(titles),
            common=pooled([m['errors'][t] for t in common]) if common else None,
            all_tracked=pooled([m['errors'][t] for t in sorted(m['tracked'])]) if m['tracked'] else None)
    out = BENCH / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'paper_metrics.json').write_text(json.dumps(report, indent=2))
    print(f'common-tracked n={len(common)}: {common}')
    for name, r in report['methods'].items():
        c, a = r['common'] or {}, r['all_tracked'] or {}
        print(f"{name} ({r['threshold_beats']:g}b): TR {r['tracking_rate']:.1f} ({len(r['tracked'])}/{len(titles)})"
              f" | common {c.get('mean', float('nan')):.3f} {c.get('le_05', float('nan')):.1f} {c.get('le_1', float('nan')):.1f}"
              f" | all {a.get('mean', float('nan')):.3f} {a.get('le_05', float('nan')):.1f} {a.get('le_1', float('nan')):.1f}")


if __name__ == '__main__':
    main()
