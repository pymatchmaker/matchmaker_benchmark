import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from eval import METRICS, run_offline_alignment, run_score_following
from tabulate import tabulate
from tqdm import tqdm
from utils import (
    MatchmakerEvalConfig,
    get_list_of_exp_config,
    save_config,
    save_results_to_csv,
)

import wandb

WORKING_DIR = Path(__file__).parent.parent
DATASET_DIR = {
    "asap": Path("/Users/carlos/repos/ismir2025_matchmaker_eval/datasets/asap-dataset-matchmaker"),
    "batik": Path("/Users/carlos/repos/ismir2025_matchmaker_eval/datasets/Batik_Audio"),
    "vienna": Path("/Users/carlos/repos/ismir2025_matchmaker_eval/datasets/vienna4x22"),
}
METADATA_PATH = {
    "asap": WORKING_DIR / "data/metadata-asap.csv",
    "batik": WORKING_DIR / "data/metadata-batik.csv",
    "vienna": WORKING_DIR / "data/metadata-vienna.csv",
}
OUTPUT_DIR = WORKING_DIR / "output"


def report_results_to_wandb(averaged_result: dict, config: MatchmakerEvalConfig):
    wandb.init(
        entity="matchmaker",
        project="matchmaker",
        group=config.method,
        config=config.model_dump(include=config.attr_exp),
        name=f"{config.method}-{config.dataset}-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}",
    )
    wandb.log(averaged_result, step=None)
    wandb.finish()


def run_tests_and_eval_by_dataset(
    dataset_type: str,
    config: MatchmakerEvalConfig,
    run_dir: Optional[Path] = None,
    dry_run: bool = False,
):
    if run_dir is None and not dry_run:
        raise ValueError("run_dir must be provided if not dry_run")

    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = DATASET_DIR[dataset_type]
    metadata = pd.read_csv(METADATA_PATH[dataset_type])
    results = defaultdict(list)
    use_musical_beat = dataset_type == "asap"
    for i, row in enumerate(metadata.itertuples(), 1):
        print(row)
        score_xml = dataset_dir / row.xml_score
        score_midi = dataset_dir / row.midi_score
        perf_audio = dataset_dir / row.audio_performance
        perf_beat_ann = dataset_dir / row.performance_annotations

        try:
            if config.method == "offline":
                result = run_offline_alignment(
                    score_xml,
                    perf_audio,
                    perf_beat_ann,
                    config,
                    use_musical_beat,
                )
            else:
                result = run_score_following(
                    score_xml,
                    perf_audio,
                    perf_beat_ann,
                    config,
                    use_musical_beat,
                    dry_run=dry_run,
                    save_dir=run_dir,
                    run_name=f"{i}",
                )
        except Exception as e:
            print(f"Error: {e}")
            continue

        # add metadata to results
        results["Index"].append(i)
        results["Piece"].append(row.title)
        results["Name"].append(perf_audio.stem)
        results["Difficulty"].append(row.difficulty)

        # add config to results
        for k, v in config.model_dump(include=config.attr_exp).items():
            results[k].append(v)

        # add evaluation results to results
        for k, v in result.items():
            results[k].append(v)

        print("Results")
        print(tabulate(results, headers="keys", tablefmt="fancy_grid", showindex=True))

    print(tabulate(results, headers="keys", tablefmt="fancy_grid", showindex=True))
    return results


def main(args):
    dataset_type = args.dataset
    method = args.method
    dry_run = args.dry_run
    wandb = args.wandb

    # save results
    if not dry_run:
        save_dir = (
            OUTPUT_DIR / f"test_results_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

    configs = get_list_of_exp_config()
    run_dir = None
    for i, config in enumerate(tqdm(configs), 1):
        if dataset_type:
            config.dataset = dataset_type
        if method:
            config.method = method
        print(f"Config: {config.model_dump(include=config.attr_exp)}")

        run_dir = save_dir / f"{i}" if not dry_run else None
        results = run_tests_and_eval_by_dataset(
            config.dataset, config, run_dir, dry_run
        )

        if not dry_run:
            save_config(config, run_dir)

            # save individual results
            save_results_to_csv(
                results, save_path=(run_dir / f"test_results.tsv").as_posix()
            )

            # save averaged results
            averaged_result = {
                k: f"{np.mean(v):.4f}" for k, v in results.items() if k in METRICS
            }
            averaged_result["piece_count"] = len(results["Piece"])
            averaged_result["count"] = sum([c for c in results["count"]])
            results_file = run_dir / "summary_results.json"
            with open(results_file, "w") as f:
                json.dump(averaged_result, f, indent=4)
            print(f"Results saved to: {results_file}")

        if not dry_run and wandb:
            # report averaged results to wandb
            report_results_to_wandb(averaged_result, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Testing Matchmaker with different methods and datasets. For further configs, use config/experiment.yaml"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["asap", "vienna", "batik"],
        default="asap",
        help="Dataset to use (asap, vienna, or batik)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["hmm", "pthmm", "dixon", "arzt", "offline"],
        default="arzt",
        help="Method to use (hmm, dixon, arzt, or offline)",
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
    args = parser.parse_args()

    main(args)
