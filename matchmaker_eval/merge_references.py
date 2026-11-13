"""Merge a sharded run of several reference methods at once.

    python matchmaker_eval/merge_references.py results/refshards

The references workflow evaluates every built-in method as a matrix of
(method x shard), so one run produces shards belonging to many different
entries. GitHub flattens them all into one download directory:

    results/refshards/refshard-arzt-audio-3/metrics.json
    results/refshards/refshard-pthmm-midi-0/metrics.json

``merge_shards.py`` merges the shards of *one* entry, so this regroups them by
entry first and then merges each. Entries whose shards did not all arrive are
reported and skipped rather than merged into a partial record.
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
import re
import shutil
from collections import defaultdict

from matchmaker_eval.load_folds import REPO_ROOT
from merge_shards import MergeError, merge

#: refshard-<method>-<input type>-<shard index>
SHARD_DIR = re.compile(r"^refshard-(?P<entry>.+)-(?P<shard>\d+)$")

DEFAULT_OUTPUT = REPO_ROOT / "results" / "submissions"


def group(download_dir: Path) -> dict:
    """``{entry: [shard directories]}`` from a flat artifact download."""
    download_dir = Path(download_dir)
    if not download_dir.is_dir():
        raise MergeError(f"{download_dir} is not a directory.")

    grouped = defaultdict(list)
    for path in sorted(download_dir.iterdir()):
        if not path.is_dir():
            continue
        match = SHARD_DIR.match(path.name)
        if not match:
            continue
        # An artifact may unpack one level deeper than expected.
        holder = path if (path / "metrics.json").exists() else None
        if holder is None:
            nested = [p for p in path.iterdir() if (p / "metrics.json").exists()]
            holder = nested[0] if len(nested) == 1 else None
        if holder is None:
            continue
        grouped[match.group("entry")].append(holder)
    return dict(grouped)


def merge_all(download_dir: Path, output_root: Path, staging: Path) -> dict:
    """Merge every complete entry found. Returns ``{entry: summary or error}``."""
    grouped = group(download_dir)
    if not grouped:
        raise MergeError(f"no refshard-* directories found under {download_dir}")

    report = {}
    for entry, shard_dirs in sorted(grouped.items()):
        target = Path(staging) / entry
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for path in shard_dirs:
            shutil.copytree(path, target / path.name)
        try:
            merged = merge(target, Path(output_root) / entry)
        except MergeError as e:
            report[entry] = {"error": str(e), "shards": len(shard_dirs)}
            continue
        report[entry] = {
            "shards": len(shard_dirs),
            "n_pieces": merged["n_pieces"],
            "n_tracked": merged["n_tracked"],
            "partial": merged["partial"],
        }
    return report


def output_for_fold(fold: str) -> Path:
    """Keep auxiliary folds from replacing the published piano records."""
    return DEFAULT_OUTPUT if fold == "eval" else REPO_ROOT / "results" / "runs" / fold


def check_fold(download_dir: Path, expected: str) -> None:
    """Validate the entire batch before any existing result is overwritten."""
    for entry, shards in group(download_dir).items():
        for shard in shards:
            actual = json.loads((shard / "metrics.json").read_text()).get("fold")
            if actual != expected:
                raise MergeError(f"{entry}: expected fold {expected!r}, found {actual!r}")


def check_expected_entries(download_dir: Path, expected: list) -> None:
    """An entirely missing method is a failed batch, not a smaller benchmark."""
    found = set(group(download_dir))
    if found != set(expected):
        raise MergeError(
            f"reference entries differ from plan: missing={sorted(set(expected)-found)}, "
            f"unexpected={sorted(found-set(expected))}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "download_dir", type=Path, help="where the shard artifacts were downloaded"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where merged records go (default depends on --fold)",
    )
    parser.add_argument("--fold", default="eval", help="fold to merge; non-eval folds are isolated")
    parser.add_argument("--expected-entries-json", help="JSON array of every entry planned by CI")
    parser.add_argument(
        "--staging",
        type=Path,
        default=None,
        help="scratch directory for regrouped shards",
    )
    args = parser.parse_args()
    staging = args.staging or (args.download_dir.parent / "refgrouped")

    try:
        if args.expected_entries_json is not None:
            check_expected_entries(args.download_dir, json.loads(args.expected_entries_json))
        check_fold(args.download_dir, args.fold)
        report = merge_all(args.download_dir, args.output or output_for_fold(args.fold), staging)
    except MergeError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1

    print(f"{'entry':30} {'shards':>7} {'pieces':>7} {'tracked':>8}  status")
    print("-" * 70)
    incomplete = 0
    for entry, info in report.items():
        if "error" in info:
            print(f"{entry:30} {info['shards']:>7} {'-':>7} {'-':>8}  FAILED")
            print(f"    {info['error']}")
            incomplete += 1
            continue
        status = "partial - not published" if info["partial"] else "complete"
        incomplete += int(info["partial"])
        print(
            f"{entry:30} {info['shards']:>7} {info['n_pieces']:>7} "
            f"{info['n_tracked']:>8}  {status}"
        )
    print(f"\n{len(report) - incomplete}/{len(report)} entry(ies) complete.")
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
