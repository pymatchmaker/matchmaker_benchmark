"""Evaluate one submission on one fold and write ``metrics.json``.

    python matchmaker_eval/run_submission.py submissions/alice
    python matchmaker_eval/run_submission.py submissions/alice --fold example
    python matchmaker_eval/run_submission.py submissions/alice --fold eval --plots

Importing a submission's ``solution.py`` registers its follower with matchmaker,
so from here on it is an ordinary method name: this module builds
``Matchmaker(method=...)`` and scores it with ``run_evaluation`` — the same call
and the same function ``test_audio.py`` / ``test_symbolic.py`` use for a
built-in method. A submission therefore cannot get a different (or more
favourable) evaluation than a baseline; there is no separate path for it.

Output lands in ``results/submissions/<name>/`` by default:

    metrics.json          — leaderboard record: summary + per-piece rows
    wp_<i>.tsv            — alignment path per piece (perf_sec, score_beat)
    gt_<i>.tsv            — ground truth per piece (perf_sec, score_beat)
    <i>.json              — per-piece metrics
"""

# Entry-point path setup. The repository root goes on sys.path so the
# `matchmaker_eval.*` modules can import each other by package name; the package
# directory goes on too, because the older eval modules import each other by
# bare module name. Submissions need neither — they import only matchmaker.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "matchmaker_eval"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import hashlib
import json
import os
import signal
import traceback
import warnings
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from matchmaker.utils.eval import resolve_gt

from eval import run_evaluation
from utils import TOLERANCES_IN_BEATS, compute_event_pooled_summary
from verify_tracking import check_tracking, plot_tracking

from matchmaker import Matchmaker
from matchmaker.base import OnlineAlignment

from matchmaker_eval.fetch_data import FetchError, fetch_for_pieces, load_config
from matchmaker_eval.folds import (
    LEADERBOARD_FOLD,
    REPO_ROOT,
    fold_path,
    load_fold,
    missing_files,
    require_files,
    shard_of,
)
from matchmaker_eval.submission import (
    ESTIMATED_BPM_KEY,
    SubmissionError,
    builtin_metadata,
    load_submission,
)

sys.setrecursionlimit(10000)

# Partitura narrates every score it parses — invisible-object notes, unparsable
# direction strings, composite durations it declines to convert. They are
# properties of the corpus, not of the run: the same handful repeat for all 146
# pieces and no submitter can act on them, so they bury the per-piece result
# lines this script exists to print. Scoped to partitura by module, so a
# warning raised by a submission's own code still comes through.
#
# Set PYTHONWARNINGS to anything to keep them — PYTHONWARNINGS=default when a
# score looks like it parsed wrongly.
if not os.environ.get("PYTHONWARNINGS"):
    warnings.filterwarnings("ignore", category=UserWarning, module=r"partitura.*")

#: Tracking verdict thresholds, per input type. A piece counts as "tracked"
#: unless some 30-second onset-wise window's median beat error exceeds the
#: threshold. MIDI is held to a tighter bound than audio.
TRACKING_THRESHOLD = {"audio": 1.0, "midi": 0.5}
SEGMENT_DURATION = 30.0

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "submissions"
DEFAULT_PIECE_TIMEOUT = 900.0


class PieceTimeout(Exception):
    """Raised when a single piece exceeds the per-piece time budget."""


@contextmanager
def piece_time_budget(seconds: Optional[float]):
    """Interrupt the current piece after ``seconds`` (POSIX only, best effort).

    Keeps one pathological piece — or a follower stuck in a loop — from
    stalling a whole benchmark run. The stream is closed by ``mm.run()``'s
    context manager as the exception unwinds.
    """
    if not seconds or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise PieceTimeout(f"exceeded the {seconds:.0f}s per-piece budget")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matchmaker_source() -> dict:
    """The git ref and commit matchmaker was installed from, if it was.

    ``pip install git+...@<ref>`` leaves a direct_url.json beside the package
    naming the ref asked for and the commit it resolved to. A release install
    has none, and then there is nothing to add.
    """
    try:
        import json
        from importlib.metadata import distribution

        raw = distribution("pymatchmaker").read_text("direct_url.json")
        if not raw:
            return {}
        vcs = (json.loads(raw).get("vcs_info") or {})
        found = {}
        if vcs.get("commit_id"):
            found["matchmaker_commit"] = vcs["commit_id"]
        if vcs.get("requested_revision"):
            found["matchmaker_ref"] = vcs["requested_revision"]
        return found
    except Exception:
        return {}


def environment_info() -> dict:
    """Record what produced these numbers, so a row can be reproduced."""
    import platform
    import subprocess

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        from importlib.metadata import version

        info["matchmaker"] = version("pymatchmaker")
    except Exception:
        info["matchmaker"] = "unknown"
    # The version string is the same on every branch, so on its own it cannot
    # say which matchmaker produced a row. When it was pip-installed from git,
    # pip records the resolved commit; keep that too.
    info.update(_matchmaker_source())
    try:
        info["benchmark_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        info["benchmark_commit"] = "unknown"
    return info


def flatten_metrics(nested: dict) -> dict:
    """``{"beat": {...}, "ms": {...}, ...}`` -> ``{"beat_...", "ms_...", ...}``."""
    flat = {}
    for key, value in nested.get("beat", {}).items():
        flat[f"beat_{key}"] = value
    for key, value in nested.get("ms", {}).items():
        flat[f"ms_{key}"] = value
    for key, value in nested.items():
        if key not in ("beat", "ms"):
            flat[key] = value
    return flat


def check_follower(follower) -> None:
    """Catch the two wiring mistakes early, with a message worth reading.

    Both surface deep inside ``OnlineAlignment.run()`` as an AttributeError or
    a TypeError on None, which tells a submitter nothing about what to fix.
    """
    if not isinstance(follower, OnlineAlignment):
        raise SubmissionError(
            "build_follower(mm) must return a matchmaker.base.OnlineAlignment "
            f"subclass instance, got {type(follower).__name__}."
        )
    if follower.queue is None:
        raise SubmissionError(
            "build_follower(mm) returned a follower without a queue. Pass "
            "queue=mm.stream.queue so the stream can feed it."
        )
    if follower.score_positions is None:
        raise SubmissionError(
            "build_follower(mm) returned a follower without score_positions. "
            "Pass score_positions=mm.score_positions — the base class needs "
            "them to know when the score is over."
        )


def run_piece(
    method, piece, input_type, index, run_dir, save_plots, estimated_bpm=False
):
    """Run and score one piece. Returns (flat_metrics, nested_metrics).

    ``method`` is the name the submission registered with matchmaker, so this
    is an ordinary ``Matchmaker`` call — the same one ``test_symbolic.py`` and
    ``test_audio.py`` make for a built-in method.

    ``estimated_bpm`` gives the follower the performance's tempo instead of the
    score's notated one. Off unless the submission declared it, and recorded so
    the leaderboard can mark the row.

    A piece whose fold row carries no tempo is run anyway, on ``tempo=None`` —
    which is matchmaker's ordinary default of the score's notated marking, or
    120 BPM where the score has none. The flat metrics report it in
    ``used_estimated_bpm`` so the count reaches the run summary: the entry is
    still marked as using the tempo, and how many pieces actually had one is
    then visible rather than assumed.
    """
    tempo = piece.performance_tempo if estimated_bpm else None

    mm = Matchmaker(
        score_file=str(piece.score_path),
        performance_file=str(piece.performance_path(input_type)),
        input_type=input_type,
        method=method,
        wait=False,
        unfold_score=True,
        tempo=tempo,
    )
    check_follower(mm.score_follower)
    list(mm.run(verbose=False))

    wp = mm.score_follower.alignment_path
    if wp is None or wp.size == 0:
        raise SubmissionError("the follower produced an empty alignment path")

    perf_sec, score_beat = resolve_gt(piece.match_path, mm.score_part.note_array())
    gt = np.column_stack([perf_sec, score_beat])

    nested = run_evaluation(
        mm,
        gt=gt,
        tolerances=TOLERANCES_IN_BEATS,
        musical_beat=piece.musical_beat,
        domain="score",
        debug=True,
        save_dir=run_dir,
        run_name=str(index),
        plot_dist_matrix=False,
        make_plot=save_plots,
    )

    # alignment_path row 0 is already absolute performance seconds.
    wp_sec = np.stack([wp[0].astype(float), wp[1].astype(float)]).T
    threshold = TRACKING_THRESHOLD[input_type]
    tracking = check_tracking(
        wp_sec,
        gt,
        segment_duration=SEGMENT_DURATION,
        threshold=threshold,
    )
    nested["tracked"] = bool(tracking["tracked"])
    nested["max_deviation"] = float(tracking["max_deviation"])
    nested["n_failed_segments"] = int(tracking["n_failed"])

    if save_plots:
        plot_tracking(
            wp_sec,
            gt,
            title=f"{piece.piece_id} #{index}",
            save_path=run_dir / f"tracking_{index}.png",
            segment_duration=SEGMENT_DURATION,
            threshold=threshold,
        )

    # Whether this piece actually got a measured tempo. False both when the
    # entry never asked and when it asked and the fold had none — the run
    # summary separates the two, since only the second is worth reporting.
    nested["used_estimated_bpm"] = tempo is not None

    with open(run_dir / f"{index}.json", "w") as f:
        json.dump(nested, f, indent=4, default=float)

    return flatten_metrics(nested), nested


def fold_file_label(fold) -> str:
    """The fold CSV's path for the record: repo-relative when it is in the repo.

    ``--fold`` also takes a path to a CSV anywhere on disk, which is how a
    one-off subset is run; those have no repo-relative form, so they are
    recorded absolute rather than crashing the run at the last step.
    """
    path = fold_path(fold)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def evaluate_submission(
    directory: Optional[Path] = None,
    fold: str = "eval",
    input_type: Optional[str] = None,
    run_dir: Optional[Path] = None,
    save_plots: bool = False,
    piece_timeout: Optional[float] = DEFAULT_PIECE_TIMEOUT,
    limit: Optional[int] = None,
    shard: Optional[int] = None,
    num_shards: int = 1,
    builtin: Optional[str] = None,
    skip_missing: bool = False,
    fetch: bool = True,
) -> dict:
    """Run a submission — or a built-in method — over a fold or one shard of it.

    Pass ``directory`` for a submission, or ``builtin`` for one of matchmaker's
    own methods. Both take the identical path from here on: the only difference
    is that a submission has to be imported first to register itself, while a
    built-in is already in matchmaker's registry.
    """
    if (directory is None) == (builtin is None):
        raise SubmissionError("Pass exactly one of a submission directory or --method.")

    if builtin is not None:
        if input_type is None:
            raise SubmissionError("--method also needs --input-type.")
        method = builtin
        metadata = builtin_metadata(builtin, input_type)
        name = f"{builtin}-{input_type}"
    else:
        directory = Path(directory)
        # Importing solution.py registers the follower with matchmaker; from
        # here on it is an ordinary method name, exactly like a built-in.
        method, registered_input_type, metadata = load_submission(directory)
        input_type = input_type or registered_input_type
        name = directory.name

    # Declared by the entry, not chosen here: a follower is given the tempo
    # only if it said it uses one, and the run is labelled accordingly.
    estimated_bpm = bool(metadata.get(ESTIMATED_BPM_KEY, False))

    all_pieces = load_fold(fold, input_type=input_type)
    fold_size = len(all_pieces)
    if limit:
        all_pieces = all_pieces[:limit]

    # Keep each piece's index within the whole fold. Shards write their
    # alignment paths as wp_<global index>.tsv, so merging them is a matter of
    # putting the files in one directory — no renaming, no collisions.
    indexed = list(enumerate(all_pieces, 1))
    if num_shards > 1:
        if shard is None:
            raise SubmissionError("--num-shards requires --shard")
        indexed = list(
            zip(
                shard_of([i for i, _ in indexed], shard, num_shards),
                shard_of([p for _, p in indexed], shard, num_shards),
            )
        )
    pieces = [p for _, p in indexed]

    if skip_missing:
        # A local dataset copy can be incomplete. Dropping those pieces keeps a
        # run usable for development, and the result stays marked partial so it
        # can never reach the leaderboard with a piece silently absent.
        usable = [
            (i, piece) for i, piece in indexed if not missing_files([piece], input_type)
        ]
        skipped = [piece.piece_id for i, piece in indexed if (i, piece) not in usable]
        if skipped:
            print(
                f"skipping {len(skipped)} piece(s) with missing files: "
                f"{', '.join(skipped[:5])}"
                f"{' ...' if len(skipped) > 5 else ''}\n"
            )
        indexed = usable
        pieces = [p for _, p in indexed]
        if not pieces:
            raise SubmissionError("No pieces of this fold have their files present.")
    else:
        # Fetch anything missing before failing. The datasets live in a separate
        # repository (data/data_sources.yaml); a first run on a clean machine
        # should just work rather than telling the user to go and run another
        # command.
        if fetch and missing_files(pieces, input_type):
            if load_config().get("enabled"):
                try:
                    fetch_for_pieces(
                        pieces, input_type, label=str(fold), fold=str(fold)
                    )
                except FetchError as e:
                    raise SubmissionError(
                        f"could not fetch the data for this run:\n{e}"
                    ) from e
        require_files(pieces, input_type)

    if run_dir is None:
        # Only an eval-fold run belongs in results/submissions/, the directory
        # the leaderboard reads. A validation or example run goes elsewhere, so it
        # cannot silently replace a published record with a one-piece smoke run.
        run_dir = (
            DEFAULT_RESULTS_DIR / name
            if str(fold) == LEADERBOARD_FOLD
            else REPO_ROOT / "results" / "runs" / str(fold) / name
        )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    shard_label = f" shard {shard + 1}/{num_shards}" if num_shards > 1 else ""
    print(
        f"{'reference ' if builtin else 'submission'} : {metadata['name']} ({name})\n"
        f"method     : {method}\n"
        f"input type : {input_type}\n"
        f"fold       : {fold} ({len(pieces)}/{fold_size} pieces{shard_label})\n"
        f"output     : {run_dir}\n"
    )

    results = defaultdict(list)
    piece_records, failures = [], []

    for position, (index, piece) in enumerate(indexed, 1):
        print(f"[{position}/{len(indexed)}] {piece.piece_id}", flush=True)
        record = {
            "index": index,
            "piece_id": piece.piece_id,
            "dataset": piece.dataset,
            "title": piece.title,
        }
        try:
            with piece_time_budget(piece_timeout):
                flat, _ = run_piece(
                    method,
                    piece,
                    input_type,
                    index,
                    run_dir,
                    save_plots,
                    estimated_bpm=estimated_bpm,
                )
        except Exception as e:
            # One broken piece must not void a whole run: record it as an
            # untracked failure (which is what it is, for the leaderboard) and
            # carry on.
            reason = f"{type(e).__name__}: {e}"
            print(f"  FAILED  {reason}")
            traceback.print_exc(limit=3)
            record.update({"tracked": False, "error": reason})
            piece_records.append(record)
            failures.append({"piece_id": piece.piece_id, "error": reason})
            results["Index"].append(index)
            results["tracked"].append(False)
            continue

        record.update(flat)
        piece_records.append(record)
        results["Index"].append(index)
        for key, value in flat.items():
            results[key].append(value)

        no_tempo = estimated_bpm and not flat.get("used_estimated_bpm")
        print(
            f"  {'TRACKED' if flat.get('tracked') else 'FAILED '}"
            f"  beat_median={flat.get('beat_median', float('nan')):.3f}"
            f"  max_dev={flat.get('max_deviation', float('nan')):.3f}b"
            + ("  (no estimated_bpm in the fold: used the score's tempo)"
               if no_tempo else "")
        )

    n_tracked = sum(1 for r in piece_records if r.get("tracked"))
    print(f"\ntracked {n_tracked}/{len(pieces)} pieces")

    # Ran, but not on the tempo it declared. Not a failure — matchmaker falls
    # back to the score's marking, which is what every other entry uses — but
    # the run is then a blend of two conditions, so it must not go unsaid.
    #
    # Counted off the fold rather than off the run records: a piece that
    # crashed has no verdict either way, and reporting it as "no tempo" would
    # blame the fold for an unrelated failure.
    n_with_tempo = sum(1 for r in piece_records if r.get("used_estimated_bpm"))
    missing_tempo = (
        [p for _, p in indexed if p.performance_tempo is None]
        if estimated_bpm
        else []
    )
    if missing_tempo:
        print(
            f"note: {len(missing_tempo)}/{len(indexed)} piece(s) carry no "
            f"estimated_bpm in {fold_file_label(fold)}, so the follower was "
            f"given the score's notated tempo for them instead.\n"
            f"      first: {', '.join(p.piece_id for p in missing_tempo[:3])}\n"
            f"      Fill the column in with:  "
            f"python matchmaker_eval/make_folds.py --from-repo"
        )

    summary_all = compute_event_pooled_summary(results, run_dir, tracked_only=False)
    summary_tracked = compute_event_pooled_summary(results, run_dir, tracked_only=True)

    metrics = {
        "submission": name,
        "method": method,
        # Whether the follower was given the performance's tempo. Carried into
        # the leaderboard so a row that had it is never compared silently with
        # rows that did not.
        "estimated_bpm": estimated_bpm,
        # Of the pieces run, how many actually had a published tempo. Equal to
        # n_pieces on a complete fold; lower means the rest fell back to the
        # score's marking, and the difference belongs in the record rather than
        # only in the log.
        "n_estimated_bpm": n_with_tempo,
        "kind": metadata.get("kind", "submission"),
        "metadata": metadata,
        "fold": str(fold),
        "fold_file": fold_file_label(fold),
        "fold_sha256": sha256_of(fold_path(fold)),
        "input_type": input_type,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment_info(),
        "n_pieces": len(pieces),
        "fold_size": fold_size,
        # A partial run — one shard, --limit, or a fold with unusable pieces —
        # is never a leaderboard entry on its own. Shards are recombined by
        # merge_shards.py; leaderboard.py drops anything still marked partial.
        "partial": len(pieces) < fold_size,
        "shard": ({"index": shard, "count": num_shards} if num_shards > 1 else None),
        "n_tracked": n_tracked,
        "n_failed": len(failures),
        "failures": failures,
        "summary_all": summary_all,
        "summary_tracked": summary_tracked,
        "pieces": piece_records,
    }

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"metrics written to {metrics_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "submission",
        type=Path,
        nargs="?",
        default=None,
        help="path to a submission directory (omit when using --method)",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="evaluate one of matchmaker's built-in methods instead of a "
        "submission, as a leaderboard reference point (needs --input-type). "
        "The names are matchmaker's own — see AVAILABLE_METHODS.",
    )
    parser.add_argument(
        "--fold",
        default="eval",
        help="fold name under data/folds/ or a path to a fold CSV (default: eval)",
    )
    parser.add_argument(
        "--input-type",
        choices=("audio", "midi"),
        default=None,
        help="override the submission's declared input type",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="output directory for this run"
    )
    parser.add_argument(
        "--plots", action="store_true", help="save per-piece alignment plots"
    )
    parser.add_argument(
        "--piece-timeout",
        type=float,
        default=DEFAULT_PIECE_TIMEOUT,
        help="seconds allowed per piece, 0 disables "
        f"(default: {DEFAULT_PIECE_TIMEOUT:.0f})",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only run the first N pieces"
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="do not download missing data; fail instead",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="skip pieces whose files are not on disk instead of failing "
        "(the run is then partial and never reaches the leaderboard)",
    )
    parser.add_argument(
        "--shard", type=int, default=None, help="run only this shard (0-based)"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="split the fold into this many shards (merge them with merge_shards.py)",
    )
    args = parser.parse_args()

    try:
        evaluate_submission(
            args.submission,
            builtin=args.method,
            fold=args.fold,
            input_type=args.input_type,
            run_dir=args.output,
            save_plots=args.plots,
            piece_timeout=args.piece_timeout,
            limit=args.limit,
            skip_missing=args.skip_missing,
            fetch=not args.no_fetch,
            shard=args.shard,
            num_shards=args.num_shards,
        )
    except SubmissionError as e:
        print(f"\nSubmission error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
