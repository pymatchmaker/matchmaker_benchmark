"""Build the leaderboard from the evaluated submissions.

    python matchmaker_eval/leaderboard.py            # rebuild results/leaderboard.*
    python matchmaker_eval/leaderboard.py --check    # fail if it is out of date

Reads every ``results/submissions/*/metrics.json`` produced by
``run_submission.py``, keeps the runs made on the **eval** fold, and writes

    results/leaderboard.json   full records, including per-fold provenance
    results/leaderboard.csv    the headline columns, for spreadsheets and papers

Ranking
-------
Rows are ordered by *tracking rate* first and median beat error second.
Accuracy is only reported over the pieces a submission actually tracked, so
ranking on accuracy alone would reward a follower that bails out of every hard
piece. Tracking rate is therefore the primary key, and the accuracy columns
describe how well a submission does on the pieces it stayed with.
"""

# Entry-point path setup — see the note in run_submission.py.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "matchmaker_eval"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import csv
import json
from datetime import datetime, timezone

from matchmaker_eval.folds import LEADERBOARD_FOLD, REPO_ROOT
from matchmaker_eval.retract import read_retractions

RESULTS_DIR = REPO_ROOT / "results"
SUBMISSION_RESULTS = RESULTS_DIR / "submissions"
LEADERBOARD_JSON = RESULTS_DIR / "leaderboard.json"
LEADERBOARD_CSV = RESULTS_DIR / "leaderboard.csv"

#: (column, path into the metrics record). Kept flat and explicit so the CSV
#: header is stable across benchmark versions.
CSV_COLUMNS = [
    "rank",
    "submission",
    "kind",
    "name",
    "authors",
    "input_type",
    "tracking_rate",
    "beat_median",
    "beat_mean",
    "beat_0.3b",
    "beat_0.5b",
    "beat_1.0b",
    "ms_median",
    "ms_300ms",
    "beat_median_all",
    "ms_median_all",
    "sparc",
    "rtf",
    "n_pieces",
    "n_tracked",
    "estimated_bpm",
    "evaluated",
]


def dataset_counts(metrics: dict) -> dict:
    """Pieces and tracking rate per dataset, small enough for the index file.

    The pooled per-dataset accuracy lives in the detail file; this is only what
    the main table needs to show a breakdown without loading a megabyte.
    """
    counts = {}
    for piece in metrics.get("pieces", []):
        dataset = piece.get("dataset", "unknown")
        entry = counts.setdefault(dataset, {"n_pieces": 0, "n_tracked": 0})
        entry["n_pieces"] += 1
        entry["n_tracked"] += int(bool(piece.get("tracked")))
    for entry in counts.values():
        entry["tracking_rate"] = round(entry["n_tracked"] / entry["n_pieces"], 4)
    return dict(sorted(counts.items()))


def row_from_metrics(metrics: dict) -> dict:
    """Flatten one metrics.json into a leaderboard row."""
    tracked = metrics.get("summary_tracked", {})
    beat = tracked.get("beat", {})
    ms = tracked.get("ms", {})
    # Accuracy over *every* piece, lost ones included. Without it a submission
    # that tracked nothing would be a row of blanks, and a submission that
    # tracked two easy pieces would look flawless.
    overall = metrics.get("summary_all", {})
    beat_all = overall.get("beat", {})
    ms_all = overall.get("ms", {})
    metadata = metrics.get("metadata", {})
    authors = metadata.get("authors") or []

    return {
        "submission": metrics["submission"],
        # "submission" (a community entry) or "reference" (a method that ships
        # with matchmaker). Both are ranked in one table — the point of a
        # reference row is to show what a submission has to beat — but the
        # website labels them so they are not confused.
        "kind": metrics.get("kind", "submission"),
        "method": metrics.get("method", ""),
        # The follower was given the performance's tempo rather than the
        # score's notated one. Shown with an asterisk: it is a different task
        # from following a performance blind, and the two must not be read as
        # one ranking without the difference being visible.
        "estimated_bpm": bool(metrics.get("estimated_bpm", False)),
        "name": metadata.get("name", metrics["submission"]),
        "authors": ", ".join(str(a) for a in authors),
        "affiliation": metadata.get("affiliation", ""),
        "url": metadata.get("url", ""),
        "description": metadata.get("description", "").strip(),
        "input_type": metrics["input_type"],
        "tracking_rate": metrics.get("summary_all", {}).get("tracking_rate", 0.0),
        "beat_median": beat.get("median"),
        "beat_mean": beat.get("mean"),
        "beat_0.3b": beat.get("0.3b"),
        "beat_0.5b": beat.get("0.5b"),
        "beat_1.0b": beat.get("1.0b"),
        "ms_median": ms.get("median"),
        "ms_300ms": ms.get("300ms"),
        "beat_median_all": beat_all.get("median"),
        "ms_median_all": ms_all.get("median"),
        "sparc": tracked.get("sparc"),
        "rtf": tracked.get("rtf"),
        "n_pieces": metrics.get("n_pieces", 0),
        "n_tracked": metrics.get("n_tracked", 0),
        "n_failed": metrics.get("n_failed", 0),
        "evaluated": metrics.get("timestamp", ""),
        "fold": metrics.get("fold", ""),
        "fold_sha256": metrics.get("fold_sha256", ""),
        "matchmaker": metrics.get("environment", {}).get("matchmaker", ""),
        "benchmark_commit": metrics.get("environment", {}).get("benchmark_commit", ""),
        "datasets": dataset_counts(metrics),
        # Per-piece rows and alignment paths, written by export_details.py and
        # loaded by the page only when someone opens this row.
        "details": f"details/{metrics['submission']}.json",
    }


def sort_key(row: dict):
    """Tracking rate first (descending), then median beat error (ascending)."""
    beat_median = row.get("beat_median")
    return (
        -(row.get("tracking_rate") or 0.0),
        beat_median if beat_median is not None else float("inf"),
        row["submission"],
    )


def collect(results_dir: Path = SUBMISSION_RESULTS, fold: str = LEADERBOARD_FOLD):
    """Read every metrics.json for ``fold`` and return ranked rows."""
    rows, skipped = [], []
    if not results_dir.exists():
        return rows, skipped

    for path in sorted(results_dir.glob("*/metrics.json")):
        try:
            metrics = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            skipped.append(f"{path}: not valid JSON ({e})")
            continue
        if metrics.get("fold") != fold:
            skipped.append(f"{path}: fold '{metrics.get('fold')}', not '{fold}'")
            continue
        if metrics.get("partial"):
            skipped.append(
                f"{path}: partial run ({metrics['n_pieces']}/"
                f"{metrics.get('fold_size')} pieces) — rerun the full fold"
            )
            continue
        rows.append(row_from_metrics(metrics))

    rows.sort(key=sort_key)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows, skipped


def build(fold: str = LEADERBOARD_FOLD) -> dict:
    rows, skipped = collect(fold=fold)
    # Published as part of the leaderboard: a result that was withdrawn is a
    # fact about the ranking, and hiding it would make the history unreadable.
    retractions = [
        {
            "submission": r["submission"],
            "reason": r.get("reason", ""),
            "retracted_at": r.get("retracted_at", ""),
            "input_type": (r.get("published") or {}).get("input_type", ""),
            "kind": (r.get("published") or {}).get("kind", ""),
        }
        for r in read_retractions()
    ]
    return {
        "fold": fold,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ranking": "tracking_rate desc, then beat_median asc",
        "metrics": {
            "tracking_rate": "share of pieces followed to the end within tolerance",
            "beat_median": "median |beat error| (perf->score), pooled over "
            "tracked pieces",
            "ms_median": "median |ms error| (score->perf), pooled over tracked pieces",
            "beat_median_all": "median |beat error| over every piece, lost "
            "ones included",
            "rtf": "real-time factor: processing time / performance duration",
            "estimated_bpm": "the follower was given the performance's tempo "
            "instead of the score's notated one — marked * in the table",
        },
        "entries": rows,
        "skipped": skipped,
        "retracted": retractions,
    }


def write(leaderboard: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_JSON.write_text(json.dumps(leaderboard, indent=2) + "\n")
    with open(LEADERBOARD_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in leaderboard["entries"]:
            writer.writerow(row)


def render_table(leaderboard: dict) -> str:
    entries = leaderboard["entries"]
    if not entries:
        return "(no submissions evaluated on the eval fold yet)"
    header = (
        f"{'#':>2}  {'entry':26} {'kind':10} {'in':5} {'track':>6} "
        f"{'beat_med':>9} {'ms_med':>8}  {'(all pieces)':>12}"
    )
    lines = [header, "-" * len(header)]
    for row in entries:

        def fmt(value, width, digits=3):
            return f"{'-':>{width}}" if value is None else f"{value:>{width}.{digits}f}"

        marked = row["submission"][:25] + "*" if row.get("estimated_bpm") else (
            row["submission"][:26]
        )
        lines.append(
            f"{row['rank']:>2}  {marked:26} {row['kind']:10} "
            f"{row['input_type']:5} "
            f"{row['tracking_rate']:>6.2f} "
            f"{fmt(row.get('beat_median'), 9)} {fmt(row.get('ms_median'), 8, 0)}  "
            f"{fmt(row.get('beat_median_all'), 12)}"
        )
    if any(row.get("estimated_bpm") for row in entries):
        lines.append("")
        lines.append(
            "* given the performance's tempo, not just the score's notated one"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fold", default=LEADERBOARD_FOLD, help="fold to rank on")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed leaderboard differs from a fresh build",
    )
    args = parser.parse_args()

    leaderboard = build(fold=args.fold)
    for message in leaderboard["skipped"]:
        print(f"skipped {message}")
    for record in leaderboard["retracted"]:
        print(f"withdrawn {record['submission']}: {record['reason']}")

    if args.check:
        if not LEADERBOARD_JSON.exists():
            print("results/leaderboard.json does not exist")
            return 1
        committed = json.loads(LEADERBOARD_JSON.read_text())
        for key in ("entries", "retracted"):
            if committed.get(key, [] if key == "retracted" else None) != leaderboard[
                key
            ]:
                print(f"results/leaderboard.json is out of date ({key}) — rebuild it.")
                return 1
        print("leaderboard is up to date")
        return 0

    write(leaderboard)
    print(render_table(leaderboard))
    print(
        f"\nwrote {LEADERBOARD_JSON.relative_to(REPO_ROOT)} and {LEADERBOARD_CSV.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
