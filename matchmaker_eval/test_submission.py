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

from matchmaker_eval.submission import SubmissionError
from run_submission import evaluate_submission

VALID_FOLD = "valid"


def summarise(metrics: dict) -> str:
    """A readable verdict, rather than a path to a JSON file."""
    overall = metrics.get("summary_all") or {}
    tracked = metrics.get("summary_tracked") or {}
    beat_all = (overall.get("beat") or {})
    beat_tracked = (tracked.get("beat") or {})
    n, total = metrics["n_tracked"], metrics["n_pieces"]

    lines = [
        "",
        f"{metrics['submission']}  ({metrics['input_type']}, validation fold)",
        "-" * 58,
        f"  tracked            {n}/{total} pieces"
        f"   ({overall.get('tracking_rate', 0.0):.0%})",
    ]
    if beat_tracked.get("median") is not None:
        lines.append(
            f"  beat error         {beat_tracked['median']:.3f} median "
            f"(tracked pieces)"
        )
    if beat_all.get("median") is not None:
        lines.append(
            f"                     {beat_all['median']:.3f} median (all pieces)"
        )
    if overall.get("rtf") is not None:
        lines.append(f"  real-time factor   {overall['rtf']:.4f}")

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
        help="measure a built-in method instead of a submission "
        "(needs --input-type)",
    )
    parser.add_argument(
        "--input-type",
        choices=("audio", "midi"),
        default=None,
        help="only needed with --method; a submission declares its own",
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

    if (args.submission is None) == (args.method is None):
        parser.error("pass either a submission directory or --method, not both")

    try:
        metrics = evaluate_submission(
            args.submission,
            fold=VALID_FOLD,
            builtin=args.method,
            input_type=args.input_type,
            limit=args.limit,
            save_plots=args.plots,
            run_dir=args.output,
        )
    except SubmissionError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1

    print(summarise(metrics))
    print(
        "\nThis is the validation fold. The leaderboard is the eval fold, which is\n"
        "evaluated for you when your pull request is merged."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
