"""Withdraw a published result from the leaderboard.

    # see what is published, and what has been withdrawn before
    python matchmaker_eval/retract.py --list

    # withdraw one entry; the reason is required and is published
    python matchmaker_eval/retract.py pfkorz-midi \
        --reason "tempo model bug, see matchmaker#71"

    # then rebuild and commit, exactly as a normal evaluation does
    python matchmaker_eval/export_details.py
    python matchmaker_eval/leaderboard.py

A leaderboard row is not a record in a database — it is derived from
``results/submissions/<entry>/metrics.json`` every time the leaderboard is
rebuilt. So withdrawing a result is deleting that directory and rebuilding, and
this script is that operation done carefully:

  * the per-piece detail file goes too, so a withdrawn result is not still
    readable at ``results/details/<entry>.json`` after its row disappears;
  * the withdrawal is recorded in ``results/retracted.json`` with a reason and
    the numbers that were published, so the leaderboard's history stays
    auditable rather than being silently rewritten;
  * the archived record can be restored with ``--undo``.

To *replace* a result rather than withdraw it — a rerun after a bug fix — do
not use this script. Re-run the evaluation; ``run_submission.py`` overwrites the
same directory and the next rebuild picks up the new numbers.
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
import shutil
from datetime import datetime, timezone

from matchmaker_eval.folds import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "results"
SUBMISSION_RESULTS = RESULTS_DIR / "submissions"
DETAILS_DIR = RESULTS_DIR / "details"
RETRACTED_PATH = RESULTS_DIR / "retracted.json"
ARCHIVE_DIR = RESULTS_DIR / "retracted"


class RetractionError(Exception):
    """Raised when a result cannot be withdrawn."""


def read_retractions() -> list:
    if not RETRACTED_PATH.exists():
        return []
    try:
        data = json.loads(RETRACTED_PATH.read_text())
    except json.JSONDecodeError as e:
        raise RetractionError(f"{RETRACTED_PATH} is not valid JSON: {e}") from e
    return data.get("retractions", []) if isinstance(data, dict) else data


def write_retractions(retractions: list) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RETRACTED_PATH.write_text(
        json.dumps(
            {
                "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "retractions": retractions,
            },
            indent=2,
        )
        + "\n"
    )


def published_entries() -> dict:
    """``{entry name: its metrics record}`` for everything currently published."""
    entries = {}
    if not SUBMISSION_RESULTS.exists():
        return entries
    for path in sorted(SUBMISSION_RESULTS.glob("*/metrics.json")):
        try:
            entries[path.parent.name] = json.loads(path.read_text())
        except json.JSONDecodeError:
            entries[path.parent.name] = {}
    return entries


def summarise(metrics: dict) -> dict:
    """The numbers that were published, kept so a retraction can be audited."""
    tracked = metrics.get("summary_tracked") or {}
    return {
        "method": metrics.get("method"),
        "kind": metrics.get("kind"),
        "input_type": metrics.get("input_type"),
        "fold": metrics.get("fold"),
        "fold_sha256": metrics.get("fold_sha256"),
        "evaluated": metrics.get("timestamp"),
        "n_pieces": metrics.get("n_pieces"),
        "n_tracked": metrics.get("n_tracked"),
        "tracking_rate": (metrics.get("summary_all") or {}).get("tracking_rate"),
        "beat_median": (tracked.get("beat") or {}).get("median"),
        "environment": metrics.get("environment"),
    }


def retract(entry: str, reason: str, archive: bool = True) -> dict:
    """Withdraw ``entry``. Returns the record written to retracted.json."""
    run_dir = SUBMISSION_RESULTS / entry
    if not (run_dir / "metrics.json").exists():
        published = sorted(published_entries())
        raise RetractionError(
            f"'{entry}' is not a published result. Published: {published}"
        )
    metrics = json.loads((run_dir / "metrics.json").read_text())

    record = {
        "submission": entry,
        "reason": reason,
        "retracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "published": summarise(metrics),
        "archived": None,
    }

    if archive:
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        destination = ARCHIVE_DIR / entry
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(run_dir), str(destination))
        record["archived"] = str(destination.relative_to(REPO_ROOT))
    else:
        shutil.rmtree(run_dir)

    detail = DETAILS_DIR / f"{entry}.json"
    if detail.exists():
        detail.unlink()

    retractions = [r for r in read_retractions() if r.get("submission") != entry]
    retractions.append(record)
    write_retractions(retractions)
    return record


def undo(entry: str) -> None:
    """Put an archived result back, and drop its retraction record."""
    source = ARCHIVE_DIR / entry
    if not (source / "metrics.json").exists():
        archived = (
            sorted(p.name for p in ARCHIVE_DIR.glob("*"))
            if ARCHIVE_DIR.exists()
            else []
        )
        raise RetractionError(
            f"'{entry}' has no archived result to restore. Archived: {archived}"
        )
    destination = SUBMISSION_RESULTS / entry
    if destination.exists():
        raise RetractionError(
            f"{destination.relative_to(REPO_ROOT)} already exists — "
            "remove or rename it before restoring."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    write_retractions(
        [r for r in read_retractions() if r.get("submission") != entry]
    )


def render_list() -> str:
    lines = ["published:"]
    entries = published_entries()
    if not entries:
        lines.append("  (none)")
    for name, metrics in entries.items():
        rate = (metrics.get("summary_all") or {}).get("tracking_rate")
        rate = "-" if rate is None else f"{rate:.3f}"
        lines.append(
            f"  {name:30} {metrics.get('kind', '?'):11} "
            f"{metrics.get('fold', '?'):6} track={rate}"
        )
    retractions = read_retractions()
    lines.append("")
    lines.append("withdrawn:")
    if not retractions:
        lines.append("  (none)")
    for record in retractions:
        lines.append(
            f"  {record['submission']:30} {record['retracted_at'][:10]}  "
            f"{record['reason']}"
        )
        if record.get("archived"):
            lines.append(f"  {'':30} archived at {record['archived']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "entry",
        nargs="?",
        help="the published result to withdraw, e.g. 'pfkorz-midi' — the "
        "directory name under results/submissions/",
    )
    parser.add_argument(
        "--reason",
        help="why it is being withdrawn. Published in results/retracted.json, "
        "so write it for whoever reads the leaderboard, not for yourself.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="show what is published and what has been withdrawn",
    )
    parser.add_argument(
        "--undo",
        action="store_true",
        help="restore a previously withdrawn result from the archive",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="delete the run outright instead of archiving it under "
        "results/retracted/ (cannot be undone)",
    )
    args = parser.parse_args()

    try:
        if args.list:
            print(render_list())
            return 0
        if not args.entry:
            parser.error("name the result to withdraw, or pass --list")
        if args.undo:
            undo(args.entry)
            print(f"restored {args.entry}")
        else:
            if not args.reason:
                parser.error(
                    "--reason is required: a withdrawn result is published as "
                    "withdrawn, and the reason is what readers see."
                )
            record = retract(args.entry, args.reason, archive=not args.purge)
            print(f"withdrew {args.entry}: {record['reason']}")
            if record["archived"]:
                print(f"archived to {record['archived']} (restore with --undo)")
        print(
            "\nNow rebuild and commit:\n"
            "  python matchmaker_eval/export_details.py\n"
            "  python matchmaker_eval/leaderboard.py"
        )
    except RetractionError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
