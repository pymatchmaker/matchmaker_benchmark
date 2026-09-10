"""Measure your own follower on the validation fold.

    python matchmaker_eval/test_submission.py submissions/your-name

This is the one command a submitter needs. It downloads the 20 validation
performances if they are not already there, runs your follower over them,
and prints how it did.

It is deliberately limited to the **valid** fold. The eval fold's 146
performances are what the leaderboard is for; using them to develop against is
what the declaration in your metadata.yaml says you did not do, and the easiest
way to keep that true is not to have a convenient way to break it. See
docs/eval-protocol.md.

To measure a built-in method the same way, pass --method instead of a
directory. For anything beyond this -- sharding, other folds, the leaderboard --
use run_submission.py, which this is a thin front door onto.
"""

# Entry-point path setup — see the note in run_submission.py.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "matchmaker_eval"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import numpy as np

from matchmaker_eval.submission import SubmissionError
from run_submission import evaluate_submission

VALID_FOLD = "valid"


def summarise(metrics: dict) -> str:
    """A readable verdict, rather than a path to a JSON file."""
    overall = metrics.get("summary_all") or {}
    tracked = metrics.get("summary_tracked") or {}
    common = metrics.get("summary_common_tracked") or {}
    beat_all = (overall.get("beat") or {})
    beat_tracked = (tracked.get("beat") or {})
    beat_common = (common.get("beat") or {})
    n, total = metrics["n_tracked"], metrics["n_pieces"]

    lines = [
        "",
        f"{metrics['submission']}  ({metrics['input_type']}, {metrics.get('fold', 'validation')} fold)",
        "-" * 65,
        f"  tracking rate      {n}/{total} pieces ({overall.get('tracking_rate', 0.0):.1%})",
    ]
    if beat_tracked.get("mean") is not None:
        lines.append(
            f"  beat error (mean)  {beat_tracked['mean']:.3f} (tracked) | "
            f"{beat_all.get('mean', float('nan')):.3f} (all)"
        )
    if beat_tracked.get("median") is not None:
        lines.append(
            f"  beat error (med)   {beat_tracked['median']:.3f} (tracked) | "
            f"{beat_all.get('median', float('nan')):.3f} (all)"
        )
    if tracked.get("sparc") is not None:
        lines.append(
            f"  SPARC (mean)       {tracked['sparc']:.2f} (tracked) | "
            f"{overall.get('sparc', float('nan')):.2f} (all)"
        )
    if tracked.get("rtf") is not None:
        lines.append(f"  real-time factor   {tracked['rtf']:.4f}")

    if common:
        lines.append("-" * 65)
        lines.append(f"  [Common Tracked Subset ({common.get('selected_count', 0)} pieces)]")
        if beat_common.get("mean") is not None:
            lines.append(
                f"  common beat error  {beat_common['mean']:.3f} mean | "
                f"{beat_common.get('median', float('nan')):.3f} median"
            )
        if common.get("sparc") is not None:
            lines.append(f"  common SPARC       {common['sparc']:.2f}")
        if common.get("rtf") is not None:
            lines.append(f"  common RTF         {common['rtf']:.4f}")

    by_dataset: dict = {}
    for piece in metrics.get("pieces", []):
        entry = by_dataset.setdefault(piece.get("dataset", "?"), [0, 0])
        entry[0] += int(bool(piece.get("tracked")))
        entry[1] += 1
    if len(by_dataset) > 1:
        lines.append("")
        for dataset, (ok, seen) in sorted(by_dataset.items()):
            lines.append(f"  {dataset:18} {ok}/{seen} tracked")

    lost = [p["piece_id"] for p in metrics.get("pieces", []) if not p.get("tracked")]
    if lost:
        lines.append("")
        lines.append(f"  lost the performance on {len(lost)} piece(s):")
        for piece_id in lost[:5]:
            lines.append(f"    {piece_id}")
        if len(lost) > 5:
            lines.append(f"    ... and {len(lost) - 5} more")

    failures = metrics.get("failures") or []
    if failures:
        lines.append("")
        lines.append(f"  {len(failures)} piece(s) did not run:")
        for failure in failures[:3]:
            lines.append(f"    {failure['piece_id']}: {failure['error'][:70]}")
    return "\n".join(lines)


from matchmaker_eval.methods import available_methods
from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "submission",
        type=Path,
        nargs="?",
        default=None,
        help="your submission directory, e.g. submissions/your-name",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="measure a built-in method instead of a submission ('all' for all methods)",
    )
    parser.add_argument(
        "--input-type",
        choices=("audio", "midi"),
        default=None,
        help="needed with --method or when running all methods; a submission declares its own",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="stop after the first N pieces, for a quick look",
    )
    parser.add_argument(
        "--plots", action="store_true", help="save a tracking plot per piece"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write the run (default: results/runs/valid/<name>)",
    )
    args = parser.parse_args()

    # Determine if user wants to run all methods for an input_type
    run_all = args.method == "all" or (args.submission is None and args.method is None and args.input_type is not None)

    if not run_all and (args.submission is None) == (args.method is None):
        parser.error("pass a submission directory, --method <name>, or --input-type <audio|midi> (to run all methods)")

    if run_all and not args.input_type:
        parser.error("pass --input-type <audio|midi> when running all methods")

    methods_to_run = available_methods(args.input_type) if run_all else [args.method]
    if run_all and args.input_type == "midi":
        methods_to_run = [m for m in methods_to_run if m not in ("OPTM", "SL_OLTW")]

    all_metrics = []
    if run_all:
        print(f"Running all {len(methods_to_run)} methods for input_type='{args.input_type}' on {VALID_FOLD} fold...")

    for m in methods_to_run:
        try:
            metrics = evaluate_submission(
                args.submission if not run_all else None,
                fold=VALID_FOLD,
                builtin=m if run_all else args.method,
                input_type=args.input_type,
                limit=args.limit,
                save_plots=args.plots,
                run_dir=args.output,
            )
            all_metrics.append(metrics)
            print(summarise(metrics))
        except SubmissionError as e:
            print(f"\n[{m}] Error: {e}", file=sys.stderr)
            if not run_all:
                return 1

    if run_all and len(all_metrics) > 1:
        # Compute common tracked subset across all evaluated methods
        tracked_sets = {}
        for met in all_metrics:
            m_name = met.get("method") or met.get("submission")
            tracked_sets[m_name] = set(
                p["index"] for p in met.get("pieces", []) if p.get("tracked")
            )

        common_indices = (
            set.intersection(*tracked_sets.values()) if tracked_sets else set()
        )

        print("\n" + "=" * 70)
        print(
            f"SUMMARY TABLE Across All {len(all_metrics)} Methods (Common Tracked: {len(common_indices)} pieces)"
        )
        print("=" * 70)

        rows = []
        for met in all_metrics:
            m_name = met.get("method") or met.get("submission")
            tr = met.get("summary_all", {}).get("tracking_rate", 0.0)
            tracked_sum = met.get("summary_tracked", {})
            beat_tr = tracked_sum.get("beat", {})

            # Filter for common tracked pieces
            pieces = met.get("pieces", [])
            common_pieces = [p for p in pieces if p.get("index") in common_indices]
            sparc_list = [p["sparc"] for p in common_pieces if "sparc" in p and p.get("sparc") is not None]
            common_sparc = float(np.mean(sparc_list)) if sparc_list else float("nan")
            
            beat_err_list = [p["beat_mean"] for p in common_pieces if "beat_mean" in p and p.get("beat_mean") is not None]
            common_beat_err = float(np.mean(beat_err_list)) if beat_err_list else float("nan")

            b05_list = [p["beat_0.5b"] for p in common_pieces if "beat_0.5b" in p and p.get("beat_0.5b") is not None]
            common_05b = float(np.mean(b05_list)) if b05_list else float("nan")

            def _fmt_val(val, fmt_str):
                return "-" if val is None or (isinstance(val, float) and np.isnan(val)) else f"{val:{fmt_str}}"

            def _fmt_pct(val):
                return "-" if val is None or (isinstance(val, float) and np.isnan(val)) else f"{val * 100:.1f}%"

            rows.append(
                {
                    "Method": m_name,
                    "TR (%)": f"{tr * 100:.1f}%",
                    "Tracked": met.get("n_tracked", 0),
                    "MeanAE": _fmt_val(beat_tr.get("mean"), ".3f"),
                    "0.5b": _fmt_pct(beat_tr.get("0.5b")),
                    "SPARC": _fmt_val(tracked_sum.get("sparc"), ".2f"),
                    "Common Count": len(common_pieces),
                    "Common MeanAE": _fmt_val(common_beat_err, ".3f"),
                    "Common 0.5b": _fmt_pct(common_05b),
                    "Common SPARC": _fmt_val(common_sparc, ".2f"),
                    "RTF": _fmt_val(tracked_sum.get("rtf"), ".4f"),
                }
            )
        print(tabulate(rows, headers="keys", tablefmt="fancy_grid"))

    print(
        "\nThis is the validation fold. The leaderboard is the eval fold, which is\n"
        "evaluated for you when your pull request is merged."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
