import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from eval import (
    run_evaluation,
    run_offline_alignment,
    run_score_following,
    FEATURES,
    DEFAULT_LOCAL_COST,
    WINDOW_SIZE,
    SAMPLE_RATE,
    HOP_LENGTH,
    FRAME_PER_SEG,
    FRAME_RATE,
)
from tabulate import tabulate
from datetime import datetime

ASAP_DIR = "/Users/jiyun/workspace/asap-dataset"
WORKING_DIR = Path(__file__).parent.parent
METADATA_ASAP = WORKING_DIR / "data/metadata-asap-test.csv"


def save_test_results(results, save_path: str):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(results.keys())
        writer.writerows(zip(*results.values()))


def report_results_to_wandb(averaged_result, config=None):
    wandb.init(
        entity="matchmaker",
        project="matchmaker",
        group="online-dp",
        config={
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "window_size(s)": WINDOW_SIZE / FRAME_RATE,
            "distance_func": DEFAULT_LOCAL_COST,
            "frame_per_seg": FRAME_PER_SEG,
            "feature": FEATURES,
            "dataset": "asap",
            "algorithm": "OLTWDixon",
        },
        name=f"{DEFAULT_LOCAL_COST}_{FEATURES}_chunk({FRAME_PER_SEG})_window({WINDOW_SIZE/FRAME_RATE})s",
    )
    wandb.log(averaged_result, step=None)
    wandb.finish()


def run_tests_and_eval(asap_dir, metadata_asap):
    results = defaultdict(list)
    for row in metadata_asap.itertuples():
        print(row)
        dir_path = asap_dir / row.folder
        target_audio = asap_dir / row.audio_performance
        score_audio = dir_path / "midi_score_adjusted.wav"

        try:
            # Run score following & evaluation
            wp = run_score_following(score_audio.as_posix(), target_audio.as_posix())
            # wp = run_offline_alignment(score_audio, target_audio)

            # Run evaluation
            score_beat_ann = dir_path / f"{score_audio.stem}_annotations.txt"
            target_beat_ann = asap_dir / row.performance_annotations
            result = run_evaluation(wp, score_beat_ann, target_beat_ann)
        except Exception as e:
            print(f"Error: {e}")
            continue

        # if result["mean"] > 10000:  # remove outliers?
        #     continue

        # add metadata to results
        results["Piece"].append(row.folder)
        results["Name"].append(target_audio.stem)

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
    args = parser.parse_args()

    asap_dir = Path(args.asap_dir)
    metadata_asap = pd.read_csv(args.metadata_asap)

    results = run_tests_and_eval(asap_dir, metadata_asap)
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
