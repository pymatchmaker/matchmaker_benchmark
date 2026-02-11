import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from eval import (
    METRICS,
    run_offline_alignment,
    run_score_following,
    parse_match_file_for_note_onsets,
)
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
    "asap": Path("~/data/asap-dataset-matchmaker").expanduser(),
    "batik": Path("~/data/Batik_Audio").expanduser(),
    "vienna": Path("~/workspace/vienna4x22").expanduser(),
    "pfvn": Path("~/data/KRAISLER").expanduser(),
    "chorale": Path("~/data/chorale-bricks").expanduser(),
    "winterreise": Path("~/data/winterreise").expanduser(),
    "zeilinger": Path("~/data/Zeilinger_data").expanduser(),
}
METADATA_PATH = {
    "valid": WORKING_DIR / "data/validation_data.csv",
    "asap": WORKING_DIR / "data/reduced/metadata-asap.csv",
    "batik": WORKING_DIR / "data/reduced/metadata-batik.csv",
    "vienna": WORKING_DIR / "data/reduced/metadata-vienna.csv",
    "pfvn": WORKING_DIR / "data/metadata-pfvn.csv",
    "chorale": WORKING_DIR / "data/metadata-chorale.csv",
    "winterreise": WORKING_DIR / "data/metadata-winterreise.csv",
    "zeilinger": WORKING_DIR / "data/metadata-zeilinger-note.csv",
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
    granularity: str = "note",
):
    if run_dir is None and not dry_run:
        raise ValueError("run_dir must be provided if not dry_run")

    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH[dataset_type], skipinitialspace=True)
    metadata.columns = metadata.columns.str.strip()
    str_cols = metadata.select_dtypes(include=["object"]).columns
    metadata[str_cols] = metadata[str_cols].apply(lambda x: x.str.strip())
    is_valid_dataset = dataset_type == "valid"
    results = defaultdict(list)
    for i, row in enumerate(metadata.itertuples(), 1):
        print(row)
        # Handle validation dataset with mixed sources
        if is_valid_dataset:
            # folder_dir = DATASET_DIR.get(row.dataset, dataset_dir)
            current_dataset = row.dataset
            dataset_dir = DATASET_DIR[current_dataset]
        else:
            current_dataset = dataset_type
            dataset_dir = DATASET_DIR[dataset_type]

        use_musical_beat = current_dataset in ["asap", "pfvn"]
        score_xml = dataset_dir / row.xml_score
        # score_midi = dataset_dir / row.midi_score
        perf_audio = dataset_dir / row.audio_performance

        # Use match file
        match_file = dataset_dir / row.match

        try:
            if config.method == "offline":
                result = run_offline_alignment(
                    score_xml,
                    perf_audio,
                    match_file,
                    config,
                    use_musical_beat,
                )
            else:
                result = run_score_following(
                    score_xml,
                    perf_audio,
                    match_file,
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
        # results["Difficulty"].append(row.difficulty)

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
    granularity = args.granularity or "note"
    adjust_tempo = args.adjust_tempo

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
        config.adjust_tempo = adjust_tempo
        print(f"Config: {config.model_dump(include=config.attr_exp)}")

        run_dir = save_dir / f"{i}" if not dry_run else None
        results = run_tests_and_eval_by_dataset(
            config.dataset, config, run_dir, dry_run, granularity
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
        choices=list(METADATA_PATH.keys()),
        default="asap",
        help="Dataset to use (asap, vienna, batik, pfvn, or valid)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["hmm", "dixon", "arzt", "offline", "audio_outerhmm"],
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
    parser.add_argument(
        "--granularity",
        type=str,
        choices=["beat", "note"],
        default="note",
        help="Granularity to use (beat or note)",
    )
    parser.add_argument(
        "--adjust-tempo",
        action="store_true",
        help="Adjust tempo based on performance audio length",
        default=False,
    )
    args = parser.parse_args()

    main(args)
