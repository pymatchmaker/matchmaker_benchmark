"""Export the detailed per-dataset and per-piece record behind each leaderboard row.

    python matchmaker_eval/export_details.py

Writes one file per evaluated entry to ``results/details/<entry>.json``:

* the run's pooled summaries and the same per dataset, copied from its
  ``metrics.json`` — every metric, not just the columns the main table shows;
* one row per piece, with its metrics and its tracking verdict;
* the alignment path and the ground truth for every piece, decimated to a fixed
  budget of points.

The leaderboard page loads a detail file only when someone opens that row, and
draws the alignment plots in the browser from these coordinates. No images are
stored: the raw path is the artifact, and a picture of it is a rendering
decision the page makes.

Decimation is why this is affordable. A full audio run holds about 745,000 path
points, which is roughly 15 MB of JSON per entry. At a few hundred points per
piece the shape of an alignment — the diagonal, where it stalls, where it jumps
— survives at any size a browser will draw it, and an entry costs about 1 MB.
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

import numpy as np

from matchmaker_eval.folds import LEADERBOARD_FOLD, REPO_ROOT
from matchmaker_eval.leaderboard import ACCURACY_COLUMNS, without_nan

SUBMISSION_RESULTS = REPO_ROOT / "results" / "submissions"
DETAILS_DIR = REPO_ROOT / "results" / "details"

#: Points kept per polyline. A leaderboard plot is a few hundred pixels wide;
#: beyond roughly this many points the extra coordinates are invisible.
PATH_POINTS = 300

#: Decimal places kept. Score positions are in beats and performance times in
#: seconds, so hundredths are already finer than a plot can show.
PRECISION = 2

#: Fields that identify a piece rather than measure it; everything else in a
#: piece record is a metric and is carried through verbatim. Listing the
#: exclusions rather than the inclusions means a metric added to the evaluation
#: reaches the website without a change here.
#: ``used_estimated_bpm`` is a condition the piece was run under, not a
#: measurement of the follower, so it is excluded here too — otherwise it lands
#: in the site's metric table as a boolean among the error statistics. The run's
#: count of it is published once, alongside n_tracked.
#: ``data_sha256`` names the bytes a piece was scored with (see provenance.py):
#: provenance, not a measurement. It is published beside the identity fields.
PIECE_IDENTITY = (
    "index",
    "piece_id",
    "dataset",
    "title",
    "tracked",
    "error",
    "used_estimated_bpm",
    "performance_bpm",
    "estimated_bpm",
    "data_sha256",
)

#: Order metrics are presented in. Anything not named here still appears, after
#: these, so nothing is ever silently dropped.
METRIC_ORDER = (
    "mean_b", "median_b", "std_b", *ACCURACY_COLUMNS, "skewness", "kurtosis",
    "smoothness", "max_deviation", "n_failed_segments", "rtf",
    "feature_latency_ms", "inference_latency_ms",
)


def ordered_metrics(piece: dict) -> list:
    """Metric names present on this piece, in presentation order."""
    present = [k for k in piece if k not in PIECE_IDENTITY]
    known = [k for k in METRIC_ORDER if k in present]
    return known + sorted(k for k in present if k not in METRIC_ORDER)


def decimate(points: np.ndarray, budget: int = PATH_POINTS) -> list:
    """Reduce a polyline to at most ``budget`` points, keeping both ends.

    A uniform stride, not a shape-simplifying algorithm: an alignment path is
    monotonic-ish and densely sampled, so evenly spaced points preserve the
    stalls and jumps that matter while staying trivial to reason about.
    """
    if len(points) == 0:
        return []
    if len(points) > budget:
        keep = np.linspace(0, len(points) - 1, budget).round().astype(int)
        keep = np.unique(keep)
        points = points[keep]
    return np.round(points.astype(float), PRECISION).tolist()


def read_path(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 2))
    try:
        data = np.loadtxt(path, delimiter="\t", skiprows=1, ndmin=2)
    except (ValueError, OSError):
        return np.empty((0, 2))
    return data if data.shape[1] >= 2 else np.empty((0, 2))


def summary_block(record: dict) -> dict:
    """Piece counts and both pooled summaries, as one block.

    The run's record and each entry of its ``datasets`` share this shape, so
    the page can put the whole run and every dataset in one table.
    """
    return {
        "n_pieces": record.get("n_pieces"),
        "n_tracked": record.get("n_tracked"),
        "n_failed": record.get("n_failed"),
        "summary_all": record.get("summary_all", {}),
        "summary_tracked": record.get("summary_tracked", {}),
    }


def export(run_dir: Path, include_paths: bool = True) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    pieces = metrics.get("pieces", [])

    detail_pieces = []
    for piece in pieces:
        index = piece["index"]
        row = {
            "index": index,
            "piece_id": piece.get("piece_id"),
            "dataset": piece.get("dataset"),
            "title": piece.get("title"),
            "tracked": bool(piece.get("tracked")),
            "performance_bpm": piece.get("performance_bpm"),
            "estimated_bpm": piece.get("estimated_bpm"),
            "data_sha256": piece.get("data_sha256"),
        }
        for key in ordered_metrics(piece):
            row[key] = piece[key]
        if piece.get("error"):
            row["error"] = piece["error"]

        if include_paths:
            # Columns are (perf_sec, score_beat) in both files.
            row["wp"] = decimate(read_path(run_dir / f"wp_{index}.tsv"))
            row["gt"] = decimate(read_path(run_dir / f"gt_{index}.tsv"))
        detail_pieces.append(row)

    return {
        "submission": metrics["submission"],
        "name": metrics.get("metadata", {}).get("name", metrics["submission"]),
        "kind": metrics.get("kind", "submission"),
        "method": metrics.get("method", ""),
        "input_type": metrics["input_type"],
        "fold": metrics.get("fold"),
        "fold_sha256": metrics.get("fold_sha256"),
        "evaluation": metrics.get("evaluation"),
        "n_pieces": metrics.get("n_pieces"),
        "n_tracked": metrics.get("n_tracked"),
        # How many pieces the follower was actually given a measured tempo for.
        # Only meaningful when the entry declared it; 0 otherwise.
        "n_estimated_bpm": metrics.get("n_estimated_bpm", 0),
        "path_points": PATH_POINTS if include_paths else 0,
        "axes": {"x": "performance time (s)", "y": "score position (beats)"},
        # Every metric the evaluation produced, in presentation order, so the
        # website can render the full table without knowing the metric set.
        "metrics": list(dict.fromkeys(key for piece in pieces for key in ordered_metrics(piece))),
        # Pooled over the whole run, and per dataset in the same shape. Both
        # come from the run's own record, computed there by the one function
        # that also produces the headline row.
        "overall": summary_block(metrics),
        "datasets": {
            name: summary_block(block)
            for name, block in sorted(metrics.get("datasets", {}).items())
        },
        "pieces": detail_pieces,
    }


#: Per-piece fields built from the run's wp_/gt_ files rather than its metrics.
PATH_FIELDS = ("wp", "gt")


def has_path_files(run_dir: Path) -> bool:
    """Whether the run's per-piece alignment paths are on this machine.

    Only ``metrics.json`` is committed. The ``wp_``/``gt_`` files exist only
    where the run itself was evaluated or merged, so a workflow that evaluates
    one entry finds every other entry here without them.
    """
    return any(run_dir.glob("wp_*.tsv")) or any(run_dir.glob("gt_*.tsv"))


def carry_over_paths(detail: dict, previous: Path) -> int:
    """Copy the alignment paths of ``previous`` into ``detail``. Returns how many pieces got them.

    Used when the run's path files are not here: the tables are rebuilt from
    ``metrics.json`` as usual, and the paths come from the detail file the run
    itself exported. A piece takes them only when its index and piece id match,
    so a re-indexed fold never gets another piece's path.
    """
    try:
        pieces = json.loads(previous.read_text()).get("pieces", [])
    except (OSError, ValueError):
        return 0
    by_index = {piece.get("index"): piece for piece in pieces}
    carried = 0
    for row in detail["pieces"]:
        old = by_index.get(row["index"])
        if old is None or old.get("piece_id") != row.get("piece_id"):
            continue
        found = False
        for field in PATH_FIELDS:
            if old.get(field):
                row[field] = old[field]
                found = True
        carried += found
    return carried


def remove_orphans(current: set, dry_run: bool = False) -> list:
    """Delete detail files with no corresponding run, and report what went.

    ``current`` is the set of file names just written. Anything else in
    ``results/details/`` belongs to a run that has been withdrawn or renamed.
    """
    if not DETAILS_DIR.exists():
        return []
    gone = []
    for path in sorted(DETAILS_DIR.glob("*.json")):
        if path.name in current:
            continue
        gone.append(path.name)
        if not dry_run:
            path.unlink()
    return gone


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--fold",
        default=LEADERBOARD_FOLD,
        help=f"only export runs on this fold (default: {LEADERBOARD_FOLD})",
    )
    parser.add_argument(
        "--no-paths",
        action="store_true",
        help="omit the alignment paths, leaving only the tables",
    )
    parser.add_argument(
        "--keep-orphans",
        action="store_true",
        help="report detail files whose run is gone instead of deleting them",
    )
    args = parser.parse_args()

    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    current = set()
    for metrics_path in sorted(SUBMISSION_RESULTS.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("fold") != args.fold or metrics.get("partial"):
            continue
        detail = export(metrics_path.parent, include_paths=not args.no_paths)
        out = DETAILS_DIR / f"{detail['submission']}.json"
        # An entry evaluated by another workflow run has no path files here;
        # rebuilt from metrics.json alone, its detail would lose every plot.
        note = ""
        if not args.no_paths and not has_path_files(metrics_path.parent) and out.exists():
            note = f", paths kept for {carry_over_paths(detail, out)} piece(s)"
        out.write_text(json.dumps(without_nan(detail), separators=(",", ":")) + "\n")
        current.add(out.name)
        size = out.stat().st_size / 1e6
        print(
            f"{detail['submission']:24} {len(detail['pieces']):>3} pieces, "
            f"{len(detail['datasets'])} dataset(s), {size:.1f} MB{note}"
        )
        written += 1

    # A detail file outlives its run otherwise, and the page fetches details by
    # name — so a withdrawn result would stay readable after its row is gone.
    removed = remove_orphans(current, dry_run=args.keep_orphans)
    for name in removed:
        verb = "would remove" if args.keep_orphans else "removed"
        print(f"{verb} orphaned detail {name}")

    if not written:
        # Not a failure: a fresh repository has no runs yet, and a dispatch on
        # a fold other than the default has nothing to export by definition.
        # Returning non-zero here would fail the workflow step after the
        # evaluation it follows had already succeeded.
        print(f"No complete {args.fold}-fold runs to export.")
        return 0
    print(f"\nwrote {written} file(s) to {DETAILS_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
