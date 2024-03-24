import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from eval import run_evaluation, run_offline_alignment, run_score_following
from tabulate import tabulate
from tqdm import tqdm
from utils import MatchmakerEvalConfig, get_list_of_exp_config

import wandb

ASAP_DIR = "/Users/jiyun/workspace/asap-dataset"
WORKING_DIR = Path(__file__).parent.parent
METADATA_ASAP = WORKING_DIR / "data/metadata-asap-test.csv"


def save_test_results(results, save_path: str):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(results.keys())
        writer.writerows(zip(*results.values()))


def report_results_to_wandb(averaged_result: dict, config: MatchmakerEvalConfig):
    wandb.init(
        entity="matchmaker",
        project="matchmaker",
        group="online-dp",
        config=config.model_dump(include=config.attr_exp),
        name=f"online-dp-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}",
    )
    wandb.log(averaged_result, step=None)
    wandb.finish()


def run_tests_and_eval(asap_dir, metadata_asap, config):
    results = defaultdict(list)
    for i, row in enumerate(metadata_asap.itertuples()):
        if i % 10 != 0:
            continue
        print(row)
        dir_path = asap_dir / row.folder
        target_audio = asap_dir / row.audio_performance
        score_audio = dir_path / "midi_score_adjusted.wav"

        try:
            # Run score following & evaluation
            wp = run_score_following(
                score_audio.as_posix(), target_audio.as_posix(), config
            )
            # wp = run_offline_alignment(score_audio, target_audio)

            # Run evaluation
            score_beat_ann = dir_path / f"{score_audio.stem}_annotations.txt"
            target_beat_ann = asap_dir / row.performance_annotations
            result = run_evaluation(
                wp, score_beat_ann, target_beat_ann, config.frame_rate
            )
        except Exception as e:
            print(f"Error: {e}")
            continue

        # if result["mean"] > 10000:  # remove outliers?
        #     continue

        # add metadata to results
        results["Piece"].append(row.folder)
        results["Name"].append(target_audio.stem)
        for k, v in config.model_dump(include=config.attr_exp).items():
            results[k].append(v)

        for k, v in result.items():
            results[k].append(v)

        print("Results")
        print(tabulate(results, headers="keys", tablefmt="fancy_grid", showindex=True))
    print(tabulate(results, headers="keys", tablefmt="fancy_grid", showindex=True))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asap-dir", type=str, help="Path to ASAP dataset", default=ASAP_DIR
    )
    parser.add_argument(
        "--metadata-asap", type=str, help="Path to metadata file", default=METADATA_ASAP
    )
    parser.add_argument(
        "--wandb", action="store_true", help="save result to wandb", default=False
    )
    parser.add_argument("--dry-run", action="store_true", help="dry run", default=False)
    args = parser.parse_args()

    asap_dir = Path(args.asap_dir)
    metadata_asap = pd.read_csv(args.metadata_asap)

    configs = get_list_of_exp_config()
    for config in tqdm(configs):
        results = run_tests_and_eval(asap_dir, metadata_asap, config)
        if not args.dry_run:
            now = datetime.now()
            save_test_results(
                results,
                save_path=f"{WORKING_DIR}/output/test_results_{now.strftime('%Y-%m-%d-%H:%M:%S')}.tsv",
            )
            if args.wandb:
                averaged_result = {
                    k: f"{np.mean(v):.2f}"
                    for k, v in results.items()
                    if k not in {"Piece", "Name"}
                }
                report_results_to_wandb(averaged_result)


if __name__ == "__main__":
    main()
