"""Evaluate matchmaker's built-in methods and refresh the leaderboard.

    python matchmaker_eval/run_references.py --input-type midi
    python matchmaker_eval/run_references.py --input-type audio
    python matchmaker_eval/run_references.py --input-type both
    python matchmaker_eval/run_references.py --input-type midi --exclude pf

These are the reference rows on the leaderboard — what a submission is trying to
beat. Which methods exist comes from the installed matchmaker's spec; which of
them this repo can label is ``data/builtin_methods.yaml``. A method matchmaker
has gained but nobody has described yet is reported at startup, so the gap is
visible rather than silent.

Each method runs in its own process so they go in parallel, and each is pinned
to a single BLAS thread: this workload is many small matrix operations, where
BLAS threading costs about twice what it saves and the processes would otherwise
fight over the same cores.

The leaderboard is rebuilt at the end, so a complete run lands on the site with
no further step. Methods that fail, or that cover fewer than all the fold's
pieces, are reported and left off it.
"""

# Entry-point path setup — see the note in run_submission.py.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "matchmaker_eval"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from matchmaker_eval.folds import LEADERBOARD_FOLD, REPO_ROOT
from matchmaker_eval.submission import (
    BUILTIN_METHODS_PATH,
    read_descriptions,
    undescribed_methods,
)

RESULTS_DIR = REPO_ROOT / "results" / "submissions"


def metrics_path_for(method: str, input_type: str, fold: str) -> Path:
    """Where run_submission.py puts this run's metrics.

    Only an eval-fold run lands in ``results/submissions/``; anything else goes
    under ``results/runs/<fold>/``. Reading the wrong one would report a
    published record as though it were the run that just finished.
    """
    name = f"{method}-{input_type}"
    if str(fold) == LEADERBOARD_FOLD:
        return RESULTS_DIR / name / "metrics.json"
    return REPO_ROOT / "results" / "runs" / str(fold) / name / "metrics.json"

#: One BLAS thread per process. Measured 2x faster than the default even for a
#: single method, before counting the gain from running several at once.
SINGLE_THREADED = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def described_methods(input_type: str) -> list:
    """The built-in methods this repo has leaderboard prose for."""
    return sorted(read_descriptions().get(input_type) or {})


def report_undescribed() -> None:
    """Name any matchmaker method that has no leaderboard entry here."""
    missing = undescribed_methods()
    for input_type, methods in missing.items():
        print(
            f"note: matchmaker has {input_type} method(s) {methods} with no entry "
            f"in {BUILTIN_METHODS_PATH.name}; they are not run as references. "
            "Describe them there to put them on the leaderboard."
        )


def run_one(method: str, input_type: str, fold: str, extra: list) -> dict:
    """Run one method in its own process. Never raises: failures are data."""
    command = [
        sys.executable,
        str(REPO_ROOT / "matchmaker_eval" / "run_submission.py"),
        "--method",
        method,
        "--input-type",
        input_type,
        "--fold",
        fold,
        *extra,
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env={**os.environ, **SINGLE_THREADED},
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started

    record = {
        "method": method,
        "input_type": input_type,
        "seconds": elapsed,
        "returncode": completed.returncode,
        "tail": completed.stderr.strip().splitlines()[-3:]
        or completed.stdout.strip().splitlines()[-3:],
    }
    metrics_path = metrics_path_for(method, input_type, fold)
    if completed.returncode == 0 and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        record.update(
            tracked=metrics["n_tracked"],
            pieces=metrics["n_pieces"],
            fold_size=metrics["fold_size"],
            partial=metrics["partial"],
        )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input-type",
        choices=("midi", "audio", "both"),
        default="both",
        help="which reference methods to run (default: both)",
    )
    parser.add_argument("--fold", default="eval", help="fold to evaluate on")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="METHOD",
        help="methods to skip, e.g. --exclude pf",
    )
    parser.add_argument(
        "--only", nargs="*", default=None, metavar="METHOD", help="run only these"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=6,
        help="how many methods to run at once (default: 6)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="pass --skip-missing through: run the pieces whose files are "
        "present. Those runs stay partial and off the leaderboard.",
    )
    parser.add_argument(
        "--no-leaderboard",
        action="store_true",
        help="do not rebuild the leaderboard afterwards",
    )
    args = parser.parse_args()

    report_undescribed()
    input_types = ("midi", "audio") if args.input_type == "both" else (args.input_type,)
    jobs = []
    for input_type in input_types:
        methods = args.only or described_methods(input_type)
        jobs += [(m, input_type) for m in methods if m not in args.exclude]

    if not jobs:
        print("Nothing to run.")
        return 1

    extra = ["--skip-missing"] if args.skip_missing else []
    print(
        f"Evaluating {len(jobs)} reference method(s) on the {args.fold} fold, "
        f"{args.jobs} at a time:\n  "
        + "\n  ".join(f"{it}/{m}" for m, it in jobs)
        + "\n"
    )

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda j: run_one(j[0], j[1], args.fold, extra), jobs))

    print(f"\n{'method':22} {'tracked':>12} {'time':>8}   status")
    print("-" * 62)
    publishable = 0
    for r in sorted(results, key=lambda r: (r["input_type"], r["method"])):
        label = f"{r['input_type']}/{r['method']}"
        if r["returncode"] != 0:
            status = "FAILED"
            counts = "-"
        elif r.get("partial"):
            status = f"partial ({r['pieces']}/{r['fold_size']}) - not published"
            counts = f"{r['tracked']}/{r['pieces']}"
        else:
            status = "published"
            counts = f"{r['tracked']}/{r['pieces']}"
            publishable += 1
        print(f"{label:22} {counts:>12} {r['seconds']:>7.0f}s   {status}")
        if r["returncode"] != 0:
            for line in r["tail"]:
                print(f"    {line}")

    if not args.no_leaderboard:
        print()
        for script in ("export_details.py", "leaderboard.py"):
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "matchmaker_eval" / script)],
                cwd=REPO_ROOT,
            )
    print(f"\n{publishable}/{len(results)} run(s) complete enough for the leaderboard.")
    return 0 if publishable else 1


if __name__ == "__main__":
    sys.exit(main())
