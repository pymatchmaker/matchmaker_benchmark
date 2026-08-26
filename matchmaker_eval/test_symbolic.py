"""MIDI (symbolic) score following benchmark runner.

Mirrors test_audio.py structure:
  - Loops over dataset metadata
  - Runs alignment per piece (HMM via Matchmaker, OLTW via event-level)
  - Saves WP/GT as TSV
  - Computes event-pooled summary (tracked-only)
"""

import argparse
import copy
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import partitura as pt
import wandb

from eval import run_evaluation
from matchmaker import Matchmaker
from matchmaker.matchmaker import DEFAULT_KWARGS
from matchmaker.utils.eval import resolve_gt
from utils import (
    DATASET_DIR,
    OUTPUT_DIR,
    SYMBOLIC_METADATA_PATH as METADATA_PATH,
    TEMPO_DEPENDENT_METHODS,
    TEMPO_METADATA_PATH,
    TOLERANCES_IN_BEATS,
    WORKING_DIR,
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


def resolve_asset_path(path_value: str, raw_base_dir: Path) -> Path:
    """Resolve raw-dataset paths and benchmark-generated assets uniformly."""
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    if path.parts[:2] == ("data", "preprocessed"):
        return WORKING_DIR / path
    return raw_base_dir / path


def run_tests_and_eval_by_dataset(
    dataset_type,
    method,
    run_dir=None,
    save_plots=True,
    matchmaker_kwargs=None,
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

        match_path = resolve_asset_path(row.match, dataset_dir)
        score_xml = resolve_asset_path(row.xml_score, dataset_dir)
        perf_midi = resolve_asset_path(row.midi_performance, dataset_dir)
        print(f"[{i}/{len(metadata)}] {row.title}")

        if method in TEMPO_DEPENDENT_METHODS and not is_valid:
            tempo_estimate = tempo_metadata.loc[tempo_metadata["midi_performance_file"] == row.midi_performance, "estimated_bpm"].values[0]
        else:
            tempo_estimate = None

        try:
            # Run alignment via Matchmaker (HMM or event-level OLTW)
            base_kwargs = (
                matchmaker_kwargs
                if matchmaker_kwargs is not None
                else DEFAULT_KWARGS["midi"].get(method, {})
            )
            mm_kwargs = copy.deepcopy(base_kwargs)
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
                segment_duration=30,
                threshold=TRACKING_THRESHOLD,
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
                        title=f"{method} #{i}",
                        save_path=run_dir / f"tracking_{i}.png",
                        threshold=TRACKING_THRESHOLD,
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


def _resolve_sweep_value(key, value):
    """Resolve class-valued W&B parameters used by Matchmaker."""
    if key != "tempo_model" or not isinstance(value, str):
        return value
    from matchmaker.utils import tempo_models

    try:
        return getattr(tempo_models, value)
    except AttributeError as exc:
        raise ValueError(f"Unknown tempo model in sweep config: {value}") from exc


def build_sweep_kwargs(method: str, wconfig) -> dict:
    """Merge W&B sweep parameters into the current MIDI method defaults."""
    method_kwargs = copy.deepcopy(DEFAULT_KWARGS.get("midi", {}).get(method, {}))
    for key, value in wconfig.items():
        if key in ("dataset", "input_type", "method"):
            continue
        method_kwargs[key] = _resolve_sweep_value(key, value)
    return method_kwargs


def main():
    parser = argparse.ArgumentParser(
        description="MIDI score following benchmark (mirrors test_audio.py)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(METADATA_PATH.keys()),
        help="Dataset to evaluate",
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
    parser.add_argument(
        "--sweep", action="store_true", help="Run as a W&B sweep agent"
    )
    args = parser.parse_args()
    if args.sweep:
        wandb.init(entity="matchmaker", project=f"symbolic-{args.method}-sweep")

    method = args.method
    dataset = args.dataset
    matchmaker_kwargs = None
    if args.sweep:
        method = wandb.config.get("method", method)
        dataset = wandb.config.get("dataset", dataset)
        if dataset is None:
            parser.error("a sweep requires dataset in W&B config or --dataset")
        matchmaker_kwargs = build_sweep_kwargs(method, wandb.config)
    elif dataset is None:
        dataset = "asap"
    effective_kwargs = (
        matchmaker_kwargs
        if matchmaker_kwargs is not None
        else DEFAULT_KWARGS.get("midi", {}).get(method, {})
    )
    processor = effective_kwargs.get("processor")
    config = SymbolicEvalConfig(method=method, dataset=dataset, processor=processor)

    ts = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if args.sweep:
        run_dir = OUTPUT_DIR / f"sweep_sym_{method}_{ts}_{wandb.run.id}"
    else:
        run_dir = OUTPUT_DIR / f"test_{ts}_sym_{method}_{dataset}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method: {method}, Dataset: {dataset}, Output: {run_dir}")

    results = run_tests_and_eval_by_dataset(
        dataset, method, run_dir=run_dir,
        save_plots=not args.sweep and not args.no_plots,
        matchmaker_kwargs=matchmaker_kwargs,
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
    if args.sweep:
        wandb.log(
            {
                "average": summary_all,
                "tracking_rate": summary_all["tracking_rate"],
            }
        )
        if summary_tracked.get("tracked_count", 0) > 0:
            wandb.log({"tracked_average": summary_tracked})
        wandb.finish()


if __name__ == "__main__":
    main()
