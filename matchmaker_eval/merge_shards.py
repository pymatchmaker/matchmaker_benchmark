"""Combine sharded benchmark runs back into one metrics record.

    python matchmaker_eval/merge_shards.py results/shards/alice \
        --output results/submissions/alice

The evaluation workflow splits a fold across parallel jobs, each producing its
own run directory. Because shards name their outputs by the piece's index in the
whole fold (``wp_37.tsv``, not ``wp_2.tsv``), merging is just collecting the
files in one directory and recomputing the pooled summary over all of them.

The merged record is marked complete only if the shards together cover the fold.
A leaderboard entry is never built from a partial run.
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
from collections import defaultdict

from utils import compute_event_pooled_summary

from matchmaker_eval.folds import REPO_ROOT


class MergeError(Exception):
    """Raised when shards cannot be combined into a coherent record."""


def find_shards(root: Path):
    """Every shard run directory under ``root`` (each holding a metrics.json)."""
    paths = sorted(root.glob("*/metrics.json"))
    if not paths:
        paths = sorted(root.glob("*/*/metrics.json"))  # artifact download nesting
    if not paths:
        raise MergeError(f"no shard metrics.json found under {root}")
    return paths


def merge(root: Path, output: Path) -> dict:
    shard_paths = find_shards(Path(root))
    records = [json.loads(p.read_text()) for p in shard_paths]

    # Every shard must describe the same experiment, or the merged numbers
    # would silently mix two different things.
    for key in (
        "submission",
        "method",
        "kind",
        "fold",
        "fold_sha256",
        "input_type",
        "estimated_bpm",
    ):
        values = {r.get(key) for r in records}
        if len(values) > 1:
            raise MergeError(f"shards disagree on '{key}': {sorted(map(str, values))}")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    pieces, failures = [], []
    results = defaultdict(list)
    seen_indices = set()

    for path, record in zip(shard_paths, records):
        shard_dir = path.parent
        for piece in record.get("pieces", []):
            index = piece["index"]
            if index in seen_indices:
                raise MergeError(
                    f"piece index {index} appears in more than one shard — "
                    "the shards were not produced from the same fold split"
                )
            seen_indices.add(index)
            pieces.append(piece)

            results["Index"].append(index)
            results["tracked"].append(bool(piece.get("tracked")))
            for key, value in piece.items():
                if key not in (
                    "index",
                    "piece_id",
                    "dataset",
                    "title",
                    "tracked",
                    "error",
                ):
                    results[key].append(value)

            # Alignment paths are what the pooled summary is computed from.
            for prefix in ("wp", "gt"):
                source = shard_dir / f"{prefix}_{index}.tsv"
                if source.exists():
                    shutil.copy2(source, output / source.name)
        failures.extend(record.get("failures", []))

    pieces.sort(key=lambda p: p["index"])
    first = records[0]
    fold_size = first.get("fold_size", len(pieces))
    complete = len(pieces) == fold_size

    merged = {
        **{
            k: first.get(k)
            for k in (
                "submission",
                # What was run and how it is labelled. Without these a
                # reference method merged from shards would come out of
                # leaderboard.py as an ordinary submission with no method name.
                "method",
                "kind",
                # Whether the entry was given the performance's tempo. Dropping
                # it here published a sharded pfkorz run as though it had run
                # on the notated tempo, unmarked and next to entries that did.
                "estimated_bpm",
                "metadata",
                "fold",
                "fold_file",
                "fold_sha256",
                "input_type",
                "environment",
            )
        },
        "timestamp": max(r.get("timestamp", "") for r in records),
        "n_pieces": len(pieces),
        "fold_size": fold_size,
        "partial": not complete,
        "shard": None,
        "merged_from": [str(p.parent.name) for p in shard_paths],
        "n_tracked": sum(1 for p in pieces if p.get("tracked")),
        # Recounted from the merged pieces rather than summed across shards, so
        # it stays right whatever subset of shards was merged.
        "n_estimated_bpm": sum(1 for p in pieces if p.get("used_estimated_bpm")),
        "n_failed": len(failures),
        "failures": failures,
        "summary_all": compute_event_pooled_summary(
            results, output, tracked_only=False
        ),
        "summary_tracked": compute_event_pooled_summary(
            results, output, tracked_only=True
        ),
        "pieces": pieces,
    }

    (output / "metrics.json").write_text(
        json.dumps(merged, indent=2, default=float) + "\n"
    )
    status = "complete" if complete else f"PARTIAL ({len(pieces)}/{fold_size})"
    print(
        f"merged {len(shard_paths)} shard(s) -> {output/'metrics.json'}\n"
        f"  {status}, tracked {merged['n_tracked']}/{len(pieces)}"
    )
    if not complete:
        print("  partial runs are excluded from the leaderboard")
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path, help="directory holding the shard run dirs")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="merged run directory (default: results/submissions/<submission>)",
    )
    args = parser.parse_args()

    try:
        if args.output is None:
            first = json.loads(find_shards(args.root)[0].read_text())
            args.output = REPO_ROOT / "results" / "submissions" / first["submission"]
        merge(args.root, args.output)
    except MergeError as e:
        print(f"\nMerge error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
