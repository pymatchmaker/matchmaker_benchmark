"""Precision on the pieces every compared method tracks.

Each method's own tracked subset differs, so its tracked-only precision is
measured on different pieces from the next method's. This report fixes one
subset — the pieces tracked by *every* method in the comparison set — and pools
each method's events over that subset alone, overall and per dataset.

Reads finished ``run_submission.py`` outputs (``metrics.json`` with its
per-piece records, plus ``wp_<i>.tsv`` / ``gt_<i>.tsv``), so it works the same
on a local run and on downloaded leaderboard results::

    python matchmaker_eval/common_tracked.py \\
        --runs results/submissions/{dixon,arzt,outerhmm,skf}-audio \\
        --report results/submissions/pfkorz-audio \\
        --output results/common_tracked/audio.json

``--runs`` define the common subset. ``--report`` entries are evaluated on that
subset without shaping it — for a method that received side information (a
performance tempo, say), whose successes should not narrow the comparison.
The pooling is ``utils.compute_event_pooled_summary``, the function behind every
leaderboard number, so a common-subset value means what the leaderboard's
tracked-subset value means, on fewer pieces.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import TIMING_KEYS, compute_event_pooled_summary  # noqa: E402


class CommonTrackedError(ValueError):
    pass


def load_run(run_dir: Path) -> dict:
    """A finished, complete run: its ``metrics.json`` and where its paths live."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise CommonTrackedError(f"no metrics.json in {run_dir}")
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("partial"):
        raise CommonTrackedError(
            f"{run_dir} is a partial run ({metrics.get('n_pieces')}/"
            f"{metrics.get('fold_size')} pieces); merge its shards first"
        )
    return {
        "name": metrics.get("submission") or run_dir.name,
        "run_dir": run_dir,
        "fold_sha256": metrics.get("fold_sha256"),
        "input_type": metrics.get("input_type"),
        "pieces": metrics["pieces"],
    }


def check_same_fold(runs: Sequence[dict]) -> None:
    """Every run must cover the same pieces under the same indices."""
    reference = runs[0]
    keys = [(p["index"], p["piece_id"]) for p in reference["pieces"]]
    for run in runs[1:]:
        if run["fold_sha256"] != reference["fold_sha256"]:
            raise CommonTrackedError(
                f"{run['name']} ran on a different fold from {reference['name']}"
            )
        if [(p["index"], p["piece_id"]) for p in run["pieces"]] != keys:
            raise CommonTrackedError(
                f"{run['name']} and {reference['name']} index their pieces differently"
            )


def common_indices(runs: Sequence[dict]) -> set:
    """Indices of the pieces that every run tracked."""
    return set.intersection(
        *({p["index"] for p in run["pieces"] if p.get("tracked")} for run in runs)
    )


def _results(pieces: List[dict]) -> dict:
    """The column layout ``compute_event_pooled_summary`` reads."""
    results = defaultdict(list)
    timing = [key for key in TIMING_KEYS if any(key in p for p in pieces)]
    for piece in pieces:
        results["Index"].append(piece["index"])
        results["tracked"].append(bool(piece.get("tracked")))
        for key in timing:
            results[key].append(piece.get(key, float("nan")))
    return results


def _summaries(pieces: List[dict], run_dir: Path, common: set) -> dict:
    results = _results(pieces)
    in_common = [p for p in pieces if p["index"] in common]
    return {
        "n_pieces": len(pieces),
        "n_tracked": sum(1 for p in pieces if p.get("tracked")),
        "n_common": len(in_common),
        "tracked": compute_event_pooled_summary(results, run_dir, tracked_only=True),
        # On the common subset every piece counts, whether or not this run
        # tracked it: for a --report entry the subset was fixed by others.
        "common": (
            compute_event_pooled_summary(
                results, run_dir, tracked_only=False, common_indices=common
            )
            if in_common
            else None
        ),
    }


def evaluate_run(run: dict, common: set, defines_subset: bool) -> dict:
    pieces = run["pieces"]
    by_dataset = defaultdict(list)
    for piece in pieces:
        by_dataset[piece.get("dataset") or "unknown"].append(piece)
    return {
        "run_dir": str(run["run_dir"]),
        "defines_subset": defines_subset,
        "overall": _summaries(pieces, run["run_dir"], common),
        "datasets": {
            dataset: _summaries(rows, run["run_dir"], common)
            for dataset, rows in sorted(by_dataset.items())
        },
    }


def build_report(run_dirs: Sequence[Path], report_dirs: Sequence[Path] = ()) -> dict:
    runs = [load_run(d) for d in run_dirs]
    extra = [load_run(d) for d in report_dirs]
    if not runs:
        raise CommonTrackedError("at least one --runs entry defines the subset")
    check_same_fold(runs + extra)
    if len({r["input_type"] for r in runs + extra}) > 1:
        raise CommonTrackedError("compare runs of one input type at a time")

    common = common_indices(runs)
    pieces = runs[0]["pieces"]
    datasets = sorted({p.get("dataset") or "unknown" for p in pieces})
    return {
        "input_type": runs[0]["input_type"],
        "fold_sha256": runs[0]["fold_sha256"],
        "defined_by": [r["name"] for r in runs],
        "reported_only": [r["name"] for r in extra],
        "n_common": len(common),
        "n_common_by_dataset": {
            ds: sum(1 for p in pieces if p["index"] in common and (p.get("dataset") or "unknown") == ds)
            for ds in datasets
        },
        "common_pieces": sorted(p["piece_id"] for p in pieces if p["index"] in common),
        "methods": {
            **{r["name"]: evaluate_run(r, common, True) for r in runs},
            **{r["name"]: evaluate_run(r, common, False) for r in extra},
        },
    }


def _fmt(value: Optional[float], digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def format_table(report: dict) -> str:
    """One row per (dataset, method): TR, then tracked-subset and common-subset precision."""
    header = (
        "| Dataset | Method | TR (%) | n | MeanAE | MedAE | <=0.5b (%) | <=1.0b (%) | SPARC "
        "| n_com | MeanAE | MedAE | <=0.5b (%) | <=1.0b (%) | SPARC |"
    )
    lines = [header, "|" + "---|" * 15]

    def cells(block: Optional[dict]) -> List[str]:
        if not block or "beat" not in block:
            return ["--"] * 5
        beat = block["beat"]
        return [
            _fmt(beat.get("mean")),
            _fmt(beat.get("median")),
            _fmt(100 * beat["0.5b"], 1) if "0.5b" in beat else "--",
            _fmt(100 * beat["1.0b"], 1) if "1.0b" in beat else "--",
            _fmt(block.get("sparc")),
        ]

    datasets = sorted(report["n_common_by_dataset"]) + ["overall"]
    for dataset in datasets:
        for name, entry in report["methods"].items():
            block = entry["overall"] if dataset == "overall" else entry["datasets"].get(dataset)
            if block is None:
                continue
            tr = 100 * block["n_tracked"] / block["n_pieces"] if block["n_pieces"] else None
            label = name if entry["defines_subset"] else f"{name}*"
            lines.append(
                "| "
                + " | ".join(
                    [dataset, label, _fmt(tr, 1), str(block["n_tracked"])]
                    + cells(block["tracked"])
                    + [str(block["n_common"])]
                    + cells(block["common"])
                )
                + " |"
            )
    lines.append("")
    lines.append(
        "* evaluated on the common subset without defining it. "
        "Common precision pools every common piece, tracked by that entry or not."
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", type=Path, required=True,
                        help="run directories whose tracked pieces define the common subset")
    parser.add_argument("--report", nargs="*", type=Path, default=[],
                        help="run directories evaluated on that subset without shaping it")
    parser.add_argument("--output", type=Path, default=None,
                        help="write the report here as JSON (and a .md table beside it)")
    args = parser.parse_args(argv)

    report = build_report(args.runs, args.report)
    table = format_table(report)
    print(table)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=float))
        args.output.with_suffix(".md").write_text(table + "\n")
        print(f"\nreport written to {args.output}")
    return report


if __name__ == "__main__":
    main()
