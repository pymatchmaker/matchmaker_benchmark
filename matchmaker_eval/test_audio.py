import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from eval import run_offline_alignment, run_score_following
from methods import audio_rates, available_methods, default_kwargs
from tabulate import tabulate
from utils import (
    AudioEvalConfig,
    compute_event_pooled_summary,
    save_config,
    save_results_to_csv,
)

import wandb

sys.setrecursionlimit(10000)

WORKING_DIR = Path(__file__).parent.parent
DATASET_DIR = {
    "asap": Path("~/data/asap-dataset-matchmaker").expanduser(),
    "batik": Path("~/data/batik_plays_mozart").expanduser(),
    "vienna": Path("~/data/vienna4x22").expanduser(),
    "kraisler": Path("~/data/KRAISLER").expanduser(),
    "chorale": Path("~/data/chorale-bricks").expanduser(),
    "urmp": Path("~/data/URMP").expanduser(),
    "winterreise": Path("~/data/winterreise").expanduser(),
    "zeilinger": Path("~/data/Zeilinger_data").expanduser(),
}
METADATA_PATH = {
    "valid": WORKING_DIR / "data/metadata-validation.csv",
    "example": WORKING_DIR / "data/metadata-example.csv",
    "asap": WORKING_DIR / "data/reduced/metadata-asap.csv",
    "batik": WORKING_DIR / "data/reduced/metadata-batik.csv",
    "vienna": WORKING_DIR / "data/reduced/metadata-vienna.csv",
    "kraisler": WORKING_DIR / "data/metadata-kraisler.csv",
    "chorale": WORKING_DIR / "data/metadata-chorale.csv",
    "urmp": WORKING_DIR / "data/metadata-urmp.csv",
    "winterreise": WORKING_DIR / "data/metadata-winterreise.csv",
    "zeilinger": WORKING_DIR / "data/metadata-zeilinger-note.csv",
}
OUTPUT_DIR = WORKING_DIR / "output"
DISPLAY_COLUMNS = [
    "Index",
    "Piece",
    "beat_mean",
    "beat_median",
    "beat_0.3b",
    "beat_0.5b",
    "beat_1.0b",
    "ms_mean",
    "ms_median",
    "ms_300ms",
    "ms_1000ms",
    "tracked",
]


def report_results_to_wandb(averaged_result: dict, config: AudioEvalConfig):
    wandb.init(
        entity="matchmaker",
        project="matchmaker",
        group=config.method,
        config=config.model_dump(include=config.attr_exp),
        name=f"{config.method}-{config.dataset}-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}",
    )
    wandb.log(averaged_result, step=None)
    wandb.finish()


def print_summary_table(results: dict):
    display = {k: results[k] for k in DISPLAY_COLUMNS if k in results}
    print(tabulate(display, headers="keys", tablefmt="fancy_grid", showindex=True))
    return results


def run_tests_and_eval_by_dataset(
    dataset_type: str,
    config: AudioEvalConfig,
    run_dir: Optional[Path] = None,
    dry_run: bool = False,
    granularity: str = "note",
    matchmaker_kwargs: Optional[dict] = None,
    save_plots: bool = True,
    indices: Optional[list] = None,
    workers: int = 1,
):
    if run_dir is None and not dry_run:
        raise ValueError("run_dir must be provided if not dry_run")

    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH[dataset_type], skipinitialspace=True)
    metadata.columns = metadata.columns.str.strip()
    str_cols = metadata.select_dtypes(include=["object"]).columns
    metadata[str_cols] = metadata[str_cols].apply(lambda x: x.str.strip())
    if workers < 1:
        raise ValueError("workers must be positive")
    if indices is not None and (
        len(set(indices)) != len(indices)
        or any(i < 1 or i > len(metadata) for i in indices)
    ):
        raise ValueError("indices must be distinct one-based metadata rows")
    if workers > 1:
        selected = indices if indices is not None else list(range(1, len(metadata) + 1))
        run_piece = partial(
            run_tests_and_eval_by_dataset, dataset_type, config, run_dir,
            dry_run, granularity, matchmaker_kwargs, save_plots,
        )
        results = defaultdict(list)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for piece in pool.map(run_piece, ([i] for i in selected)):
                for key, values in piece.items():
                    results[key].extend(values)
        return results
    is_valid_dataset = dataset_type in ("valid", "example")
    results = defaultdict(list)
    for i, row in enumerate(metadata.itertuples(), 1):
        if indices is not None and i not in indices:
            continue
        print(row)
        # Handle validation dataset with mixed sources
        if is_valid_dataset:
            # folder_dir = DATASET_DIR.get(row.dataset, dataset_dir)
            current_dataset = row.dataset
            dataset_dir = DATASET_DIR[current_dataset]
        else:
            current_dataset = dataset_type
            dataset_dir = DATASET_DIR[dataset_type]

        use_musical_beat = current_dataset in ["asap", "kraisler"]

        # kraisler annotations are temporarily beat-level TODO: fix to note-level
        piece_granularity = "beat" if current_dataset == "kraisler" else granularity

        # Determine base directory: some datasets (e.g. chorale) have paths
        # relative to a folder column, while others include the full path.
        has_folder = hasattr(row, "folder")
        if has_folder and not row.xml_score.startswith(row.folder):
            base_dir = dataset_dir / row.folder
        else:
            base_dir = dataset_dir

        if current_dataset == "winterreise":
            score_xml = WORKING_DIR / row.xml_score
        else:
            score_xml = base_dir / row.xml_score
        perf_audio = base_dir / row.audio_performance

        # Unified GT: match-file datasets pass the .match file directly (score
        # beats read from the match file); no-match datasets use a precomputed
        # data/gt/<dataset>/<i>.tsv of (perf_sec, score_beat).
        has_match = (
            hasattr(row, "match") and pd.notna(row.match) and str(row.match).strip()
        )
        if has_match:
            match_file = dataset_dir / row.match
            perf_annotations = None
            piece_gt = match_file
        else:
            match_file = None
            if dataset_type in ("urmp", "kraisler", "winterreise", "chorale"):
                gt_path = WORKING_DIR / row.performance_annotations
            else:
                gt_path = WORKING_DIR / "data" / "gt" / current_dataset / f"{i}.tsv"
            if not gt_path.exists():
                print(f"GT file not found: {gt_path}, skipping")
                continue
            gt_arr = np.loadtxt(gt_path, delimiter="\t", skiprows=1, ndmin=2)
            perf_annotations = gt_arr[:, 0]  # perf_sec column
            piece_gt = gt_path

        try:
            if config.method == "offline":
                result = run_offline_alignment(
                    score_xml,
                    perf_audio,
                    match_file,
                    config,
                    use_musical_beat,
                    perf_annotations=perf_annotations,
                    granularity=piece_granularity,
                )
            else:
                result = run_score_following(
                    score_xml,
                    perf_audio,
                    config,
                    use_musical_beat,
                    dry_run=dry_run,
                    save_dir=run_dir,
                    run_name=f"{i}",
                    match_file=match_file,
                    granularity=piece_granularity,
                    matchmaker_kwargs=matchmaker_kwargs,
                    save_plots=save_plots,
                    gt=piece_gt,
                )
        except Exception as e:
            print(f"Error: {e}")
            continue

        # add metadata to results
        results["Index"].append(i)
        results["Piece"].append(row.title)
        results["Name"].append(perf_audio.stem)
        # results["Difficulty"].append(row.difficulty)

        # add config to results
        for k, v in config.model_dump(include=config.attr_exp).items():
            results[k].append(v)

        # add evaluation results to results
        for k, v in result.items():
            results[k].append(v)

        print_summary_table(results)

    print_summary_table(results)
    return results


def build_sweep_kwargs(method: str, wconfig) -> dict:
    """The method's defaults from matchmaker's spec, overridden by the sweep."""
    method_kwargs = default_kwargs("audio", method)
    for key, value in wconfig.items():
        if key not in ("dataset", "method"):
            method_kwargs[key] = value
    return method_kwargs


def main(args):
    dataset_type = args.dataset
    method = args.method
    dry_run = args.dry_run
    use_wandb = args.wandb
    granularity = args.granularity or "note"

    matchmaker_kwargs = json.loads(args.kwargs) if args.kwargs else None

    if args.sweep:
        method = wandb.config.get("method", method)
        dataset_type = wandb.config.get("dataset", dataset_type)
        matchmaker_kwargs = build_sweep_kwargs(method, wandb.config)

    # Report the rates the run will actually use: the method's defaults from
    # matchmaker's spec, overridden by the sweep config when there is one.
    kw = {**default_kwargs("audio", method), **(matchmaker_kwargs or {})}
    config = AudioEvalConfig(
        method=method,
        dataset=dataset_type,
        **audio_rates(kw),
    )

    print(f"Config: {config.model_dump(include=config.attr_exp)}")

    # save results
    ts = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if not dry_run:
        if args.sweep:
            save_dir = OUTPUT_DIR / f"sweep_{config.method}_{ts}_{wandb.run.id}"
        else:
            save_dir = OUTPUT_DIR / f"test_{ts}_aud_{config.method}_{config.dataset}"
        if args.output_dir:
            save_dir = Path(args.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "method_kwargs.json").open("w") as f:
            json.dump({**default_kwargs("audio", method), **(matchmaker_kwargs or {})}, f, indent=2)

    run_dir = save_dir if not dry_run else None
    results = run_tests_and_eval_by_dataset(
        config.dataset,
        config,
        run_dir,
        dry_run,
        granularity,
        matchmaker_kwargs=matchmaker_kwargs,
        save_plots=not args.sweep and not args.no_plots,
        indices=args.indices,
        workers=args.workers,
    )

    if not dry_run:
        save_config(config, run_dir)

        # save individual results
        save_results_to_csv(
            results, save_path=(run_dir / f"test_results.tsv").as_posix()
        )

        expected = len(args.indices) if args.indices is not None else len(
            pd.read_csv(METADATA_PATH[config.dataset])
        )
        completed = len(results.get("Index", []))
        with (run_dir / "completion.json").open("w") as f:
            json.dump({"expected": expected, "completed": completed,
                       "complete": completed == expected}, f, indent=2)
        if completed != expected:
            raise RuntimeError(
                f"Incomplete evaluation: {completed}/{expected}; inspect the log"
            )

        # Compute event-wise pooled summary (both all and tracked-only)
        summary_all = compute_event_pooled_summary(results, run_dir, tracked_only=False)
        summary_tracked = compute_event_pooled_summary(
            results, run_dir, tracked_only=True
        )
        for label, s in [("all", summary_all), ("tracked", summary_tracked)]:
            path = run_dir / f"summary_{label}.json"
            with open(path, "w") as f:
                json.dump(s, f, indent=4)
            print(f"Results saved to: {path}")

    if not dry_run and args.sweep:
        # Sweep mode: log to current wandb run
        wandb.log(
            {
                "average": summary_all,
                "tracking_rate": summary_all.get("tracking_rate", 0),
            }
        )
        if summary_tracked.get("tracked_count", 0) > 0:
            wandb.log({"tracked_average": summary_tracked})
    elif not dry_run and use_wandb:
        report_results_to_wandb(summary_tracked, config)


def build_parser() -> argparse.ArgumentParser:
    """The command line. Method choices come from matchmaker's registry."""
    parser = argparse.ArgumentParser(
        description="Testing Matchmaker with different methods and datasets. "
        "Per-method configuration is read from matchmaker's spec."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(METADATA_PATH.keys()),
        default="asap",
        help="Dataset to use (asap, vienna, batik, kraisler, or valid)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="arzt",
        choices=available_methods("audio") + ["offline"],
        help="Audio method from matchmaker's registry, or 'offline' for "
        "offline DTW",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="dry run (without saving or reporting results)",
        default=False,
    )
    parser.add_argument(
        "--wandb", action="store_true", help="report results to wandb", default=False
    )
    parser.add_argument(
        "--sweep", action="store_true", help="run as wandb sweep agent", default=False
    )
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["beat", "note"],
        default="note",
        help="Granularity to use (beat or note)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip saving per-piece plots (faster evaluation)",
        default=False,
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel piece evaluations")
    parser.add_argument("--indices", type=int, nargs="+", help="One-based metadata rows")
    parser.add_argument("--kwargs", help="JSON overrides for method defaults")
    parser.add_argument("--output-dir", help="Explicit result directory")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.sweep:
        with wandb.init(
            entity="matchmaker",
            project=f"audio-{args.method}-sweep",
        ):
            main(args)
    else:
        main(args)
