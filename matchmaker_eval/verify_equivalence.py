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

The clone is built from matchmaker's own spec for the method
(``matchmaker/methods.yaml``), so this covers every built-in method rather than
a hand-maintained list of them, and the two paths cannot drift apart when a
method's construction changes.

A method the spec marks ``deterministic: false`` (a particle filter, say) cannot
be checked this way: the two runs draw from the same random stream, so their
metrics differ no matter how the follower was built. Those are reported as
inconclusive rather than as failures.
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
from matchmaker.matchmaker import unregister_method
from matchmaker.registry import REGISTRY
from matchmaker.utils.eval import resolve_gt

from eval import run_evaluation
from matchmaker_eval.folds import load_fold
from methods import builtin_methods, default_kwargs, processor_for
from utils import TOLERANCES_IN_BEATS


def register_clone(method: str, input_type: str) -> str:
    """Register the built-in ``method``'s follower under a new name.

    The spec object is itself the builder, so the clone is constructed by the
    very code path ``Matchmaker`` uses for the built-in — the point being that
    no second, hand-written description of the method can go stale.
    """
    spec = REGISTRY.method(input_type, method)
    name = f"clone-{method}"
    processor_type = processor_for(input_type, method)

    def build_follower(mm):
        follower = spec.build_follower(mm)
        # The built-ins let some followers define their own state space; the
        # base class needs score_positions to know when the score has ended.
        if follower.score_positions is None:
            follower.score_positions = mm.score_positions
        return follower

    register_method(
        name,
        input_type=input_type,
        build_follower=build_follower,
        # Under a new name the method's own processor overrides would be lost,
        # so the processor is built as the original method's.
        build_processor=lambda mm: REGISTRY.build_processor(
            mm, processor_type, method=method
        ),
        build_reference=spec.build_reference,
        default_kwargs=default_kwargs(input_type, method),
        overwrite=True,
    )
    return name


def is_builtin_for(method: str, input_type: str) -> bool:
    return method in builtin_methods(input_type)


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


def run(method: str, piece, input_type: str):
    mm = Matchmaker(
        score_file=str(piece.score_path),
        performance_file=str(piece.performance_path(input_type)),
        input_type=input_type,
        method=method,
    )
    list(mm.run(verbose=False))
    return metrics_for(mm, piece)


def compare(method: str, fold: str, limit: int, input_type: str = "midi") -> int:
    pieces = load_fold(fold, input_type=input_type)[:limit]
    deterministic = REGISTRY.method(input_type, method).deterministic
    if not deterministic:
        print(
            f"note: '{method}' is marked deterministic: false in matchmaker's "
            "spec. Repeated runs differ on their own, so any mismatch below "
            "says nothing about the two code paths.\n"
        )
    clone = register_clone(method, input_type)
    mismatches = 0

    try:
        for piece in pieces:
            print(f"\n{piece.piece_id}")
            builtin = run(method, piece, input_type)  # what the runners do
            registered = run(clone, piece, input_type)  # what a submission does

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
        unregister_method(clone, input_type)

    print()
    if mismatches and not deterministic:
        print(
            f"{mismatches} metric(s) differ, as expected for a non-deterministic "
            f"method. INCONCLUSIVE for '{method}' — compare a deterministic "
            "method to check the two paths."
        )
        return 0
    if mismatches:
        print(f"{mismatches} metric(s) differ between the two paths.")
        return 1
    print(f"All metrics identical across {len(pieces)} piece(s) for method '{method}'.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--method",
        default="pthmm",
        choices=sorted(set(builtin_methods("midi")) | set(builtin_methods("audio"))),
        help="built-in method to clone and compare",
    )
    parser.add_argument(
        "--input-type",
        default="midi",
        choices=("audio", "midi"),
        help="which stream the method consumes",
    )
    parser.add_argument("--fold", default="tuning", help="fold to sample pieces from")
    parser.add_argument("--limit", type=int, default=3, help="how many pieces")
    args = parser.parse_args()
    if not is_builtin_for(args.method, args.input_type):
        parser.error(
            f"'{args.method}' is not a built-in {args.input_type} method. "
            f"Available: {builtin_methods(args.input_type)}"
        )
    return compare(args.method, args.fold, args.limit, args.input_type)


if __name__ == "__main__":
    sys.exit(main())
