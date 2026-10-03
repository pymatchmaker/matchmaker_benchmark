"""Score a finished run again from its saved alignment paths.

The follower's output does not depend on the ground truth, so when only the
evaluation changes — how GT is built, how a piece is judged — a run need not be
repeated: its ``wp_<i>.tsv`` paths are scored again by the same code
``run_submission.py`` uses (``resolve_gt``, ``evaluate_alignment``,
``check_tracking``, the pooled summaries) and written to a new directory::

    python matchmaker_eval/rescore.py output/old/arzt-midi --output output/new/arzt-midi

Timing columns (RTF, latency) cannot be measured again from a path and are
carried over from the original records. The score is loaded by constructing
``Matchmaker`` exactly as a run does, so the GT is joined to the same unfolded
note array.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import json
import shutil
from datetime import datetime, timezone

import numpy as np
from matchmaker import Matchmaker
from matchmaker.utils.eval import evaluate_alignment, resolve_gt

from run_submission import (
    SEGMENT_DURATION,
    TRACKING_THRESHOLD,
    environment_info,
    flatten_metrics,
)
from utils import (
    TIMING_KEYS,
    TOLERANCES_IN_BEATS,
    TOLERANCES_IN_MS,
    compute_sparc,
    dataset_summaries,
    pooled_summaries,
    save_nparray_to_csv,
)
from verify_tracking import check_tracking

from matchmaker_eval.folds import load_fold

#: Any MIDI method will do: only the score loading is used, and it does not
#: depend on the method.
SCORE_LOADER = {"midi": "pthmm", "audio": "arzt"}


def score_notes(piece, input_type: str):
    mm = Matchmaker(
        score_file=str(piece.score_path),
        performance_file=str(piece.performance_path(input_type)),
        input_type=input_type,
        method=SCORE_LOADER[input_type],
        wait=False,
        unfold_score=True,
    )
    return mm.score_part.note_array()


def rescore_piece(record: dict, piece, input_type: str, src: Path, dst: Path) -> dict:
    index = record["index"]
    wp = np.loadtxt(src / f"wp_{index}.tsv", delimiter="\t", skiprows=1, ndmin=2)
    shutil.copyfile(src / f"wp_{index}.tsv", dst / f"wp_{index}.tsv")
    perf_sec, score_beat = wp[:, 0], wp[:, 1]

    gt_perf, gt_beat = resolve_gt(piece.match_path, score_notes(piece, input_type))
    save_nparray_to_csv(
        np.column_stack([gt_perf, gt_beat]),
        (dst / f"gt_{index}.tsv").as_posix(),
        header="perf_sec\tscore_beat",
    )

    nested = evaluate_alignment(
        score_beat,
        perf_sec,
        gt_beat,
        gt_perf,
        beat_tolerances=TOLERANCES_IN_BEATS,
        ms_tolerances=TOLERANCES_IN_MS,
    )
    old = json.loads((src / f"{index}.json").read_text())
    for key in TIMING_KEYS:
        if key in old:
            nested[key] = old[key]
    nested["sparc"] = compute_sparc(score_beat, perf_sec)
    tracking = check_tracking(
        wp,
        np.column_stack([gt_perf, gt_beat]),
        segment_duration=SEGMENT_DURATION,
        threshold=TRACKING_THRESHOLD[input_type],
    )
    nested["tracked"] = bool(tracking["tracked"])
    nested["max_deviation"] = float(tracking["max_deviation"])
    nested["n_failed_segments"] = int(tracking["n_failed"])
    nested["used_estimated_bpm"] = old.get("used_estimated_bpm", False)
    with open(dst / f"{index}.json", "w") as f:
        json.dump(nested, f, indent=4, default=float)

    keep = {k: record[k] for k in ("index", "piece_id", "dataset", "title")}
    return {**keep, **flatten_metrics(nested)}


def rescore_run(src: Path, dst: Path) -> dict:
    src, dst = Path(src), Path(dst)
    metrics = json.loads((src / "metrics.json").read_text())
    if metrics.get("partial"):
        raise ValueError(f"{src} is a partial run; merge its shards first")
    input_type = metrics["input_type"]
    pieces = dict(enumerate(load_fold(metrics["fold"], input_type=input_type), 1))
    dst.mkdir(parents=True, exist_ok=True)

    records = []
    for record in metrics["pieces"]:
        if "error" in record:
            # Never produced a path: stays the untracked failure it was.
            records.append(record)
            continue
        piece = pieces[record["index"]]
        assert piece.piece_id == record["piece_id"], (piece.piece_id, record)
        records.append(rescore_piece(record, piece, input_type, src, dst))
        print(
            f"[{record['index']}/{len(metrics['pieces'])}] {record['piece_id']}  "
            f"tracked {record.get('tracked')} -> {records[-1]['tracked']}",
            flush=True,
        )

    metrics.update(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rescored_from": {
                "run_dir": str(src),
                "timestamp": metrics.get("timestamp"),
                "environment": metrics.get("environment"),
            },
            "environment": environment_info(),
            "n_tracked": sum(1 for r in records if r.get("tracked")),
            **pooled_summaries(records, dst),
            "datasets": dataset_summaries(records, dst),
            "pieces": records,
        }
    )
    with open(dst / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\ntracked {metrics['n_tracked']}/{len(records)}; written to {dst}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="a finished run_submission.py output")
    parser.add_argument("--output", type=Path, required=True,
                        help="directory for the rescored run (never the source)")
    args = parser.parse_args()
    if args.output.resolve() == args.run_dir.resolve():
        parser.error("--output must differ from the run being rescored")
    rescore_run(args.run_dir, args.output)


if __name__ == "__main__":
    main()
