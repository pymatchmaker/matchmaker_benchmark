"""Build the leaderboard from the evaluated submissions.

    python matchmaker_eval/leaderboard.py            # rebuild results/leaderboard.*
    python matchmaker_eval/leaderboard.py --check    # fail if it is out of date

Reads every ``results/submissions/*/metrics.json`` produced by
``run_submission.py``, keeps the runs made on the **eval** fold, and writes

    results/leaderboard.json            full records, with each fold's evaluation key
    results/leaderboard.csv             the headline columns, for papers
    results/leaderboard-<dataset>.json  the same entries ranked on one dataset
    results/leaderboard-<dataset>.csv

Each entry in ``leaderboard.json`` also carries a ``datasets`` block with the
headline columns per dataset, so the index answers "how does it do on asap"
without a second file. The per-dataset files are the other pivot — "who is best
on asap" — and are cut from the same rows, so the two can never disagree.

Ranking
-------
Rows are ordered by *tracking rate* first and median beat error second.
Accuracy is only reported over the pieces a submission actually tracked, so
ranking on accuracy alone would reward a follower that bails out of every hard
piece. Tracking rate is therefore the primary key, and the accuracy columns
describe how well a submission does on the pieces it stayed with.

The per-dataset files rank by the same key within the dataset. Read those ranks
next to ``n_pieces``: a dataset of thirty pieces separates methods far less
reliably than the whole fold does.
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
import math
from datetime import datetime, timezone

from matchmaker.utils.eval import TOLERANCES_IN_BEATS

from matchmaker_eval.load_folds import LEADERBOARD_FOLD, REPO_ROOT, RESULTS_DIR
from matchmaker_eval.evaluation_key import SHARED_COMPONENTS
from matchmaker_eval.retract import read_retractions
from matchmaker_eval.submission import read_descriptions

SUBMISSION_RESULTS = RESULTS_DIR / "submissions"
LEADERBOARD_JSON = RESULTS_DIR / "leaderboard.json"
LEADERBOARD_CSV = RESULTS_DIR / "leaderboard.csv"

#: A reference's citation links to its entry here, by the ``bib`` key that
#: data/builtin_methods.yaml gives each method.
BIBLIOGRAPHY_URL = "https://pymatchmaker.github.io/score-following-bibliography/"


def dataset_paths(dataset: str) -> tuple:
    """Where the leaderboard restricted to one dataset is written."""
    return (
        RESULTS_DIR / f"leaderboard-{dataset}.json",
        RESULTS_DIR / f"leaderboard-{dataset}.csv",
    )

#: (column, path into the metrics record). Kept flat and explicit so the CSV
#: header is stable across benchmark versions.
ACCURACY_COLUMNS = [f"accuracy_{tolerance:g}b" for tolerance in TOLERANCES_IN_BEATS]
CSV_COLUMNS = [
    "rank", "submission", "kind", "name", "authors", "input_type",
    "tracking_rate", "mean_b", "median_b", "std_b", *ACCURACY_COLUMNS,
    "skewness", "kurtosis", "smoothness", "median_b_all", "rtf",
    "feature_latency_ms", "inference_latency_ms", "n_pieces", "n_tracked",
    "estimated_bpm", "evaluated",
    # The conditions the row was scored under (see evaluation_key.py). Cite it
    # with any number taken from this file.
    "evaluation_key",
]

#: The per-dataset CSV adds where the entry stands on the whole fold, so a
#: table cut from one dataset never hides that the ranking there is partial.
DATASET_CSV_COLUMNS = CSV_COLUMNS[:1] + ["overall_rank"] + CSV_COLUMNS[1:]

#: Who the entry is: the same on every row that describes it.
IDENTITY_COLUMNS = (
    "submission",
    "kind",
    "method",
    "estimated_bpm",
    "name",
    "authors",
    "affiliation",
    "url",
    "citation_url",
    "description",
    "input_type",
)

#: Where the numbers came from: also the same on every row of an entry.
PROVENANCE_COLUMNS = (
    "evaluated",
    "fold",
    "fold_sha256",
    "evaluation_key",
    "matchmaker",
    "matchmaker_commit",
    "benchmark_commit",
    "details",
)

#: Where a leaderboard is kept once published under an evaluation key, inside
#: the results directory. A key's directory is rewritten while the key is
#: current and left as it is after, so the numbers a paper cites stay where it
#: says they are.
ARCHIVE_NAME = "archive"

def metric_columns(record: dict) -> dict:
    """The number columns of a row, from a record's pooled summaries.

    ``record`` is either the whole metrics.json or one entry of its
    ``datasets`` block — both carry piece counts plus ``summary_all`` and
    ``summary_tracked`` — so a per-dataset column is produced by the same code
    as the headline column and means exactly the same thing.
    """
    tracked = record.get("summary_tracked", {})
    overall = record.get("summary_all", {})
    return {
        "tracking_rate": overall.get("tracking_rate", 0.0),
        **{key: tracked.get(key) for key in (
            "mean_b", "median_b", "std_b", "skewness", "kurtosis", "smoothness",
            "rtf", "feature_latency_ms", "inference_latency_ms",
        )},
        **{key: tracked.get(key) for key in ACCURACY_COLUMNS},
        "median_b_all": overall.get("median_b"),
        "n_pieces": record.get("n_pieces", 0),
        "n_tracked": record.get("n_tracked", 0),
        "n_failed": record.get("n_failed", 0),
    }


def citation_url(metrics: dict) -> str:
    """The bibliography entry a reference's citation links to, or "".

    Read from data/builtin_methods.yaml when the board is built rather than
    from the run's metadata, so correcting a key needs no re-run and no edit
    to the stored results.
    """
    if metrics.get("kind") != "reference":
        return ""
    described = read_descriptions().get(metrics["input_type"]) or {}
    key = (described.get(metrics.get("method")) or {}).get("bib")
    return f"{BIBLIOGRAPHY_URL}#{key}" if key else ""


def row_from_metrics(metrics: dict) -> dict:
    """Flatten one metrics.json into a leaderboard row."""
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
        "citation_url": citation_url(metrics),
        "description": metadata.get("description", "").strip(),
        "input_type": metrics["input_type"],
        **metric_columns(metrics),
        "evaluated": metrics.get("timestamp", ""),
        "fold": metrics.get("fold", ""),
        "fold_sha256": metrics.get("fold_sha256", ""),
        "evaluation_key": (metrics.get("evaluation") or {}).get("key"),
        "evaluation": (metrics.get("evaluation") or {}).get("components") or {},
        "matchmaker": metrics.get("environment", {}).get("matchmaker", ""),
        "matchmaker_commit": metrics.get("environment", {}).get("matchmaker_commit", ""),
        "benchmark_commit": metrics.get("environment", {}).get("benchmark_commit", ""),
        # Per-piece rows and alignment paths, written by export_details.py and
        # loaded by the page only when someone opens this row.
        "details": f"details/{metrics['submission']}.json",
        # The headline columns again, once per dataset. A few hundred bytes
        # each, so the index answers "how does it do on asap" by itself.
        "datasets": {
            name: metric_columns(block)
            for name, block in sorted(metrics.get("datasets", {}).items())
        },
    }


def sort_key(row: dict):
    """Tracking rate first (descending), then median beat error (ascending)."""
    median_b = row.get("median_b")
    return (
        -(row.get("tracking_rate") or 0.0),
        median_b if median_b is not None else float("inf"),
        row["submission"],
    )


def collect(results_dir: Path = None, fold: str = LEADERBOARD_FOLD):
    """Read every metrics.json for ``fold`` and return ranked rows."""
    # Resolved at call time, like the paths write() uses, so a test can point
    # the whole module at a scratch results/ tree.
    results_dir = SUBMISSION_RESULTS if results_dir is None else results_dir
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
        summary = metrics.get("summary_all", {})
        if "smoothness" not in summary or "accuracy_b" in summary:
            skipped.append(f"{path}: previous metric format — re-evaluate this entry")
            continue
        rows.append(row_from_metrics(metrics))

    rows.sort(key=sort_key)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows, skipped


def dataset_boards(leaderboard: dict) -> dict:
    """One leaderboard per dataset, cut from the rows of the main one.

    An entry appears on a dataset's board when its run covered that dataset,
    with the dataset's columns in place of the headline ones and a fresh rank
    within the dataset. ``overall_rank`` keeps its place on the whole fold
    visible, because a rank over thirty pieces is not the same claim as one
    over the whole fold.
    """
    entries = {}
    for row in leaderboard["entries"]:
        for dataset, columns in row.get("datasets", {}).items():
            entries.setdefault(dataset, []).append(
                {
                    **{key: row[key] for key in IDENTITY_COLUMNS},
                    **columns,
                    **{key: row[key] for key in PROVENANCE_COLUMNS},
                    "overall_rank": row["rank"],
                }
            )

    boards = {}
    for dataset, rows in sorted(entries.items()):
        rows.sort(key=sort_key)
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        boards[dataset] = {
            "fold": leaderboard["fold"],
            "dataset": dataset,
            "generated": leaderboard["generated"],
            "ranking": leaderboard["ranking"] + ", within this dataset",
            "metrics": leaderboard["metrics"],
            "entries": rows,
        }
    return boards


def split_by_conditions(rows: list) -> tuple:
    """(conditions per input type, rows scored under them, rows scored otherwise).

    The current conditions are those of the most recently evaluated row: every
    run is scored by the code, data and fold of its moment, so the newest run
    is the one that reflects them. MIDI and audio stream different files, so
    their keys always differ in ``data_sha256``: each input type has its own
    current key, the newest among its rows that share the newest run's code
    components (fold, scoring, matchmaker, partitura). Every ranked row,
    whichever its input type, therefore shares those.

    A row with any other key — or none, from before keys existed — was scored
    under conditions that have since changed and is not ranked against the
    rest. It is kept, unranked, with the components that differ, so the page
    can say what it is waiting for.

    When no row carries a key at all, nothing can be told apart and every row
    is treated as current: the board then reads as it did before keys.
    """
    if not rows:
        return {}, [], []

    def evaluated(row):
        return row.get("evaluated") or ""

    def code(row):
        theirs = row.get("evaluation") or {}
        return {name: theirs.get(name) for name in SHARED_COMPONENTS}

    newest_code = code(max(rows, key=evaluated))
    conditions = {}
    for input_type in sorted({row["input_type"] for row in rows}):
        same_code = [
            row for row in rows
            if row["input_type"] == input_type and code(row) == newest_code
        ]
        if same_code:
            latest = max(same_code, key=evaluated)
            conditions[input_type] = {
                "key": latest.get("evaluation_key"),
                "components": latest.get("evaluation") or {},
            }

    current, earlier = [], []
    for row in rows:
        reference = conditions.get(row["input_type"])
        if reference and row.get("evaluation_key") == reference["key"]:
            current.append(row)
            continue
        earlier.append(row)
        theirs = row.get("evaluation") or {}
        # Without a current key of its own input type, only the code components
        # can be compared: another input type's data hash says nothing here.
        ours = reference["components"] if reference else newest_code
        names = set(ours) | set(theirs) if reference else SHARED_COMPONENTS
        row["evaluation_changed"] = sorted(
            name for name in names if ours.get(name) != theirs.get(name)
        ) if row.get("evaluation_key") else []
    return conditions, current, earlier


def rank(rows: list) -> list:
    rows.sort(key=sort_key)
    for position, row in enumerate(rows, 1):
        row["rank"] = position
    return rows


def build(fold: str = LEADERBOARD_FOLD) -> dict:
    rows, skipped = collect(fold=fold)
    conditions, rows, earlier = split_by_conditions(rows)
    rank(rows)
    earlier.sort(key=sort_key)
    for row in earlier:
        row["rank"] = None
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
        # Every ranked entry was scored under the conditions of its input
        # type. Cite its key with the numbers; results/archive/<key>/ keeps
        # them.
        "evaluation_keys": {
            input_type: condition["key"] for input_type, condition in conditions.items()
        },
        "evaluation_conditions": {
            input_type: condition["components"]
            for input_type, condition in conditions.items()
        },
        "ranking": "tracking_rate desc, then median_b asc",
        "metrics": {
            "tracking_rate": "share of pieces followed to the end within tolerance",
            "median_b": "median absolute beat error, pooled over tracked pieces",
            "mean_b": "mean absolute beat error, pooled over tracked pieces",
            **{key: "share of onsets within " + key.removeprefix("accuracy_") for key in ACCURACY_COLUMNS},
            "smoothness": "mean SPARC smoothness across tracked pieces; higher is smoother",
            "median_b_all": "median absolute beat error over every piece",
            "rtf": "real-time factor: processing time / performance duration",
            "estimated_bpm": "given the performance tempo rather than only the score tempo",
            "evaluation_key": "hash of the conditions a row was scored under "
            "(fold, data files, scoring code, matchmaker sources, partitura); "
            "rows are only ranked under the current key of their input type",
        },
        "entries": rows,
        # Scored under other conditions than the entries above, so not ranked
        # against them. Each carries its own key and the components that
        # differ; re-running it moves it back into the ranking.
        "earlier_conditions": earlier,
        "skipped": skipped,
        "retracted": retractions,
    }


def without_nan(value):
    """NaN and infinity as null, recursively.

    JSON has no spelling for them. Python writes ``NaN`` anyway and reads it
    back, but a browser's ``response.json()`` rejects the whole file — one
    undefined skewness would take the page's every number down with it.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: without_nan(item) for key, item in value.items()}
    if isinstance(value, list):
        return [without_nan(item) for item in value]
    return value


def write_pair(board: dict, json_path: Path, csv_path: Path, columns: list) -> None:
    json_path.write_text(json.dumps(without_nan(board), indent=2) + "\n")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in board["entries"]:
            writer.writerow(row)


def write(leaderboard: dict) -> list:
    """Write the main leaderboard and one per dataset. Returns the file names.

    A per-dataset file whose dataset no longer appears in any run is removed:
    it would otherwise keep publishing a ranking nothing feeds any more.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_pair(leaderboard, LEADERBOARD_JSON, LEADERBOARD_CSV, CSV_COLUMNS)
    written = [LEADERBOARD_JSON.name, LEADERBOARD_CSV.name]

    for dataset, board in dataset_boards(leaderboard).items():
        json_path, csv_path = dataset_paths(dataset)
        write_pair(board, json_path, csv_path, DATASET_CSV_COLUMNS)
        written += [json_path.name, csv_path.name]

    for stale in sorted(RESULTS_DIR.glob("leaderboard-*.*")):
        if stale.name not in written and stale.suffix in (".json", ".csv"):
            stale.unlink()

    for input_type, key in leaderboard.get("evaluation_keys", {}).items():
        if key:
            written += archive(leaderboard, input_type)
    return written


def archive(leaderboard: dict, input_type: str) -> list:
    """Keep one input type's table under its evaluation key. Returns what was written.

    ``results/archive/<key>/`` holds the entries of that key, ranked among
    themselves: rewritten
    while the key is current, untouched once a newer key takes over. A paper
    that cites a key can therefore always point at the exact table it took
    its numbers from, and a reader can check them against the conditions
    recorded beside it. ``results/archive/index.json`` lists every key with
    when it was first and last published.
    """
    key = leaderboard["evaluation_keys"][input_type]
    conditions = leaderboard["evaluation_conditions"][input_type]
    archive_dir = RESULTS_DIR / ARCHIVE_NAME
    index_path = archive_dir / "index.json"
    target = archive_dir / key
    target.mkdir(parents=True, exist_ok=True)
    board = {
        **{k: leaderboard[k] for k in ("fold", "generated")},
        "input_type": input_type,
        "evaluation_key": key,
        "evaluation_conditions": conditions,
        **{k: leaderboard[k] for k in ("ranking", "metrics")},
        "entries": rank([
            dict(row) for row in leaderboard["entries"] if row["input_type"] == input_type
        ]),
    }
    write_pair(board, target / "leaderboard.json", target / "leaderboard.csv", CSV_COLUMNS)

    index = []
    if index_path.exists():
        index = json.loads(index_path.read_text()).get("archives", [])
    entry = next((e for e in index if e["key"] == key), None)
    if entry is None:
        entry = {
            "key": key,
            "input_type": input_type,
            "first_published": leaderboard["generated"],
        }
        index.append(entry)
    entry.update({
        "last_updated": leaderboard["generated"],
        "n_entries": len(board["entries"]),
        "conditions": conditions,
    })
    index.sort(key=lambda e: e["first_published"])
    index_path.write_text(json.dumps({"archives": index}, indent=2) + "\n")
    return [f"archive/{key}/leaderboard.json", f"archive/{key}/leaderboard.csv",
            "archive/index.json"]


def render_table(leaderboard: dict) -> str:
    entries = leaderboard["entries"]
    if not entries:
        return "(no submissions evaluated on the eval fold yet)"
    header = (
        f"{'#':>2}  {'entry':26} {'kind':10} {'in':5} {'track':>6} "
        f"{'beat_med':>9} {'smooth':>8}  {'(all pieces)':>12}"
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
            f"{fmt(row.get('median_b'), 9)} {fmt(row.get('smoothness'), 8, 4)}  "
            f"{fmt(row.get('median_b_all'), 12)}"
        )
    if any(row.get("estimated_bpm") for row in entries):
        lines.append("")
        lines.append(
            "* given the performance's tempo, not just the score's notated one"
        )
    keys = leaderboard.get("evaluation_keys") or {}
    if any(keys.values()):
        lines.append("")
        lines.append("evaluation key " + ", ".join(
            f"{input_type} {key}" for input_type, key in keys.items() if key
        ))
    for row in leaderboard.get("earlier_conditions", []):
        changed = ", ".join(row.get("evaluation_changed") or []) or "no key"
        lines.append(
            f"  not ranked: {row['submission']} (key {row.get('evaluation_key')}, "
            f"differs in {changed}) — re-run it to rank it"
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
        for key in ("entries", "earlier_conditions", "retracted"):
            if committed.get(key, [] if key != "entries" else None) != leaderboard[
                key
            ]:
                print(f"results/leaderboard.json is out of date ({key}) — rebuild it.")
                return 1
        print("leaderboard is up to date")
        return 0

    written = write(leaderboard)
    print(render_table(leaderboard))
    print(f"\nwrote {', '.join(written)} to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
