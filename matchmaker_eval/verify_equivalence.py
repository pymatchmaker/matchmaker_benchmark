"""Check that a registered method is scored exactly like a built-in one.

    python matchmaker_eval/verify_equivalence.py --method pthmm --limit 3

The benchmark has two entry points: ``test_audio.py`` / ``test_symbolic.py`` for
the built-in methods, and ``run_submission.py`` for community submissions. They
must not be two different benchmarks. Since submissions register themselves with
``matchmaker.register_method``, both paths end in the same ``Matchmaker`` call
and the same ``run_evaluation``.

This script proves it rather than asserting it: it takes one built-in method,
re-registers the very same follower under a new name through the public
registration API, runs both on the same pieces, and compares every metric. Run
it after touching the evaluation code or the registration hooks.
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
from matchmaker import Matchmaker, register_method
from matchmaker.matchmaker import DEFAULT_KWARGS, unregister_method
from matchmaker.prob import OuterProductHMM, PitchHMM, PitchIOIHMM
from matchmaker.utils.eval import resolve_gt

from eval import run_evaluation
from matchmaker_eval.folds import load_fold
from utils import TOLERANCES_IN_BEATS

#: The built-in symbolic HMMs. Rebuilding these through register_method
#: mirrors how Matchmaker builds them internally.
HMM_CLASSES = {"pthmm": PitchHMM, "hmm": PitchIOIHMM, "outerhmm": OuterProductHMM}


def register_clone(method: str) -> str:
    """Register the built-in ``method``'s follower under a new name."""
    cls = HMM_CLASSES[method]
    name = f"clone-{method}"

    def build_follower(mm):
        kwargs = {"reference_features": mm.reference_features, "queue": mm.stream.queue}
        if cls is not OuterProductHMM:
            kwargs["has_insertions"] = True
        follower = cls(**kwargs)
        # The built-ins let the follower define its own state space; the base
        # class needs score_positions to know when the score has ended.
        if follower.score_positions is None:
            follower.score_positions = mm.score_positions
        return follower

    register_method(
        name,
        input_type="midi",
        build_follower=build_follower,
        default_kwargs=dict(DEFAULT_KWARGS["midi"].get(method, {})),
        overwrite=True,
    )
    return name


def metrics_for(mm, piece):
    perf_sec, score_beat = resolve_gt(piece.match_path, mm.score_part.note_array())
    gt = np.column_stack([perf_sec, score_beat])
    return run_evaluation(
        mm,
        gt=gt,
        tolerances=TOLERANCES_IN_BEATS,
        domain="score",
        debug=False,
        plot_dist_matrix=False,
        make_plot=False,
    )


def run(method: str, piece):
    mm = Matchmaker(
        score_file=str(piece.score_path),
        performance_file=str(piece.performance_path("midi")),
        input_type="midi",
        method=method,
    )
    list(mm.run(verbose=False))
    return metrics_for(mm, piece)


def compare(method: str, fold: str, limit: int) -> int:
    pieces = load_fold(fold, input_type="midi")[:limit]
    clone = register_clone(method)
    mismatches = 0

    try:
        for piece in pieces:
            print(f"\n{piece.piece_id}")
            builtin = run(method, piece)  # what test_symbolic.py does
            registered = run(clone, piece)  # what a submission does

            for domain in ("beat", "ms"):
                for key in sorted(builtin.get(domain, {})):
                    x, y = builtin[domain][key], registered[domain].get(key)
                    same = x == y or (
                        isinstance(x, float)
                        and isinstance(y, float)
                        and np.isclose(x, y)
                    )
                    if not same:
                        mismatches += 1
                        print(f"  MISMATCH {domain}_{key}: builtin={x} registered={y}")
            print(
                f"  beat_median  builtin={builtin['beat']['median']:.4f} "
                f"registered={registered['beat']['median']:.4f}"
            )
            print(
                f"  ms_median    builtin={builtin['ms']['median']:.4f} "
                f"registered={registered['ms']['median']:.4f}"
            )
    finally:
        unregister_method(clone, "midi")

    print()
    if mismatches:
        print(f"{mismatches} metric(s) differ between the two paths.")
        return 1
    print(f"All metrics identical across {len(pieces)} piece(s) for method '{method}'.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--method", default="pthmm", choices=sorted(HMM_CLASSES), help="built-in method"
    )
    parser.add_argument("--fold", default="tuning", help="fold to sample pieces from")
    parser.add_argument("--limit", type=int, default=3, help="how many pieces")
    args = parser.parse_args()
    return compare(args.method, args.fold, args.limit)


if __name__ == "__main__":
    sys.exit(main())
