"""MIDI (symbolic) score following benchmark runner.

Mirrors test_audio.py structure:
  - Loops over dataset metadata
  - Runs alignment per piece (HMM via Matchmaker, OLTW via event-level)
  - Saves WP/GT as TSV
  - Computes event-pooled summary (tracked-only)
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import partitura as pt

from eval import run_evaluation
from matchmaker import Matchmaker
from matchmaker.matchmaker import DEFAULT_KWARGS
from matchmaker.utils.eval import resolve_gt
from utils import (
    TOLERANCES_IN_BEATS,
    SymbolicEvalConfig,
    compute_event_pooled_summary,
    save_config,
    save_results_to_csv,
)
from verify_tracking import check_tracking, plot_tracking

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.setrecursionlimit(10000)

TRACKING_THRESHOLD = 0.5  # beats
TRACKING_MIN_FAILS = 2

WORKING_DIR = Path(__file__).parent.parent
DATASET_DIR = {
    "asap": Path("~/data/asap-dataset-matchmaker").expanduser(),
    "batik": Path("~/data/batik_plays_mozart").expanduser(),
    "vienna": Path("~/data/vienna4x22").expanduser(),
}
METADATA_PATH = {
    "valid": WORKING_DIR / "data/metadata-validation.csv",
    "asap": WORKING_DIR / "data/reduced/metadata-asap.csv",
    "batik": WORKING_DIR / "data/reduced/metadata-batik.csv",
    "vienna": WORKING_DIR / "data/reduced/metadata-vienna.csv",
    "example": WORKING_DIR / "data/metadata-example.csv",
}
OUTPUT_DIR = WORKING_DIR / "output"

TEMPO_DEPENDENT_METHODS = ["pfkorz"]

TEMPO_METADATA_PATH = WORKING_DIR / "data/perf_tempo_estimate"


def run_tests_and_eval_by_dataset(
    dataset_type, method, run_dir=None, save_plots=True
):
    """Run symbolic alignment for all pieces in a dataset."""
    metadata = pd.read_csv(METADATA_PATH[dataset_type])
    is_valid = dataset_type in ("valid", "example")

    if not is_valid:
        tempo_metadata_fn = TEMPO_METADATA_PATH / f"{dataset_type}_tempo_estimates.csv"
        tempo_metadata = pd.read_csv(tempo_metadata_fn)

    results = defaultdict(list)

    for i, row in enumerate(metadata.itertuples(), 1):
        if is_valid:
            dataset_dir = DATASET_DIR[row.dataset]
        else:
            dataset_dir = DATASET_DIR[dataset_type]

        match_path = dataset_dir / row.match
        score_xml = dataset_dir / row.xml_score
        perf_midi = dataset_dir / row.midi_performance
        print(f"[{i}/{len(metadata)}] {row.title}")

        if method in TEMPO_DEPENDENT_METHODS and not is_valid:
            tempo_estimate = tempo_metadata.loc[tempo_metadata["midi_performance_file"] == row.midi_performance, "estimated_bpm"].values[0]
        else:
            tempo_estimate = None

        try:
            # Run alignment via Matchmaker (HMM or event-level OLTW)
            mm_kwargs = DEFAULT_KWARGS["midi"].get(method, {}).copy()
            mm = Matchmaker(
                score_file=str(score_xml),
                performance_file=str(perf_midi),
                input_type="midi",
                method=method,
                tempo=tempo_estimate,
                kwargs=mm_kwargs if mm_kwargs else None,
            )
            list(mm.run(verbose=False))
            wp = mm.score_follower.alignment_path  # (2, T): perf, score

            # Convert WP perf axis to absolute seconds (HMM WPs are IOI-
            # accumulated from 0; OLTW event WPs are already absolute).
            wp_perf_sec = mm._wp_perf_to_seconds(wp[0].astype(float))
            wp = np.stack([wp_perf_sec, wp[1].astype(float)])

            ps, sb = resolve_gt(match_path, mm.score_part.note_array())
            gt = np.column_stack([ps, sb])

            wp_T = wp.T if wp.shape[0] == 2 else wp
            tracking = check_tracking(
                wp_T,
                gt,
                frame_rate=1,
                segment_duration=30,
                threshold=TRACKING_THRESHOLD,
                mode="beat",
                min_fails=TRACKING_MIN_FAILS,
            )

            nested = run_evaluation(
                mm,
                gt=gt,
                tolerances=TOLERANCES_IN_BEATS,
                domain="score",
                debug=run_dir is not None,
                save_dir=run_dir,
                run_name=str(i),
                make_plot=save_plots,
            )
            nested["tracked"] = tracking["tracked"]
            nested["max_deviation"] = float(tracking["max_deviation"])
            nested["n_failed_segments"] = int(tracking["n_failed"])

            # Flatten nested {"beat": {...}, "ms": {...}} for the per-piece results
            # table (matches test_audio columns); the per-piece JSON stays nested.
            piece_result = {}
            for k, v in nested.get("beat", {}).items():
                piece_result[f"beat_{k}"] = v
            for k, v in nested.get("ms", {}).items():
                piece_result[f"ms_{k}"] = v
            for k, v in nested.items():
                if k not in ("beat", "ms"):
                    piece_result[k] = v

            # Save WP/GT — column order: perf_sec, score_beat
            if run_dir is not None:
                np.savetxt(run_dir / f"wp_{i}.tsv", wp_T, delimiter="\t", fmt="%.6f", header="perf_sec\tscore_beat", comments="")
                np.savetxt(run_dir / f"gt_{i}.tsv", gt, delimiter="\t", fmt="%.6f", header="perf_sec\tscore_beat", comments="")
                with open(run_dir / f"{i}.json", "w") as f:
                    json.dump(nested, f, indent=4, default=float)
                if save_plots:
                    plot_tracking(
                        wp_T,
                        gt,
                        frame_rate=1,
                        title=f"{method} #{i}",
                        save_path=run_dir / f"tracking_{i}.png",
                        mode="beat",
                        threshold=TRACKING_THRESHOLD,
                        min_fails=TRACKING_MIN_FAILS,
                    )

            status = "TRACKED" if tracking["tracked"] else "FAILED"
            print(f"  {status} (max_dev={tracking['max_deviation']:.3f}b)")

            results["Index"].append(i)
            results["Piece"].append(row.title)
            for k, v in piece_result.items():
                results[k].append(v)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(
        description="MIDI score following benchmark (mirrors test_audio.py)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="asap",
        help="Dataset (valid, example, asap, batik, vienna)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="hmm",
        help="Method (hmm, pthmm, outerhmm, arzt, dixon)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving per-piece tracking plots (faster evaluation)",
        default=False,
    )
    args = parser.parse_args()

    method = args.method
    dataset = args.dataset
    processor = DEFAULT_KWARGS.get("midi", {}).get(method, {}).get("processor")
    config = SymbolicEvalConfig(method=method, dataset=dataset, processor=processor)

    ts = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    run_dir = OUTPUT_DIR / f"test_{ts}_sym_{method}_{dataset}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method: {method}, Dataset: {dataset}, Output: {run_dir}")

    results = run_tests_and_eval_by_dataset(
        dataset, method, run_dir=run_dir,
        save_plots=not args.no_plots,
    )

    n_total = len(results["Index"])
    n_tracked = sum(results["tracked"])
    print(f"\nTracked: {n_tracked}/{n_total}")

    save_results_to_csv(results, (run_dir / "test_results.tsv").as_posix())

    summary_all = compute_event_pooled_summary(results, run_dir, tracked_only=False)
    summary_tracked = compute_event_pooled_summary(results, run_dir, tracked_only=True)
    for label, s in [("all", summary_all), ("tracked", summary_tracked)]:
        path = run_dir / f"summary_{label}.json"
        with open(path, "w") as f:
            json.dump(s, f, indent=4)
        print(f"Results saved to: {path}")

    save_config(config, run_dir)


if __name__ == "__main__":
    main()
