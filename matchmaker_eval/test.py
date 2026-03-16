import argparse
import json
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import soundfile as sf
from eval import parse_annotation_csv, run_offline_alignment, run_score_following
from tabulate import tabulate
from tqdm import tqdm
from utils import (
    MatchmakerEvalConfig,
    compute_event_pooled_summary,
    get_list_of_exp_config,
    save_config,
    save_results_to_csv,
)

import wandb

sys.setrecursionlimit(10000)

WORKING_DIR = Path(__file__).parent.parent
DATASET_DIR = {
    "asap": Path("~/data/asap-dataset-matchmaker").expanduser(),
    "batik": Path("~/data/Batik_Audio").expanduser(),
    "vienna": Path("~/data/vienna4x22").expanduser(),
    "pfvn": Path("~/data/KRAISLER").expanduser(),
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
    "pfvn": WORKING_DIR / "data/metadata-pfvn.csv",
    "chorale": WORKING_DIR / "data/metadata-chorale.csv",
    "urmp": WORKING_DIR / "data/metadata-urmp.csv",
    "winterreise": WORKING_DIR / "data/metadata-winterreise.csv",
    "zeilinger": WORKING_DIR / "data/metadata-zeilinger-note.csv",
}
OUTPUT_DIR = WORKING_DIR / "output"
DISPLAY_COLUMNS = [
    "Index",
    "Piece",
    "mean",
    "median",
    "300ms",
    "1000ms",
    "mean_b",
    "median_b",
    "0.3b",
    "1.0b",
    "2.0b",
    "tracked",
]


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


def print_summary_table(results: dict):
    display = {k: results[k] for k in DISPLAY_COLUMNS if k in results}
    print(tabulate(display, headers="keys", tablefmt="fancy_grid", showindex=True))
    return results


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
    is_valid_dataset = dataset_type in ("valid", "example")
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

        # Determine base directory: some datasets (e.g. chorale) have paths
        # relative to a folder column, while others include the full path.
        has_folder = hasattr(row, "folder")
        if has_folder and not row.xml_score.startswith(row.folder):
            base_dir = dataset_dir / row.folder
        else:
            base_dir = dataset_dir

        score_xml = base_dir / row.xml_score
        # score_midi = base_dir / row.midi_score
        perf_audio = base_dir / row.audio_performance

        # Get performance annotations: from note annotation file, match file, or annotation CSV
        trimmed_audio_path = None
        has_match = hasattr(row, "match")
        if has_match:
            match_file = dataset_dir / row.match
            # Use match file parsing (via eval.py) to ensure annotation alignment
            # with build_score_annotations, especially when ignore_invisible_objects
            # changes the note count.
            perf_annotations = None
        else:
            match_file = None
            annotation_file = base_dir / row.performance_annotations
            if not annotation_file.exists():
                print(f"Annotation file not found: {annotation_file}, skipping")
                continue
            perf_annotations = parse_annotation_csv(annotation_file)

        # Trim leading silence (URMP only): shift audio and annotations by first onset
        trim_offset = 0.0
        trimmed_audio_path = None
        if (
            current_dataset == "urmp"
            and perf_annotations is not None
            and len(perf_annotations) > 0
        ):
            margin = 0.5  # keep 0.5s before first note
            trim_offset = max(0, perf_annotations[0] - margin)
            if trim_offset > 1.0:  # only trim if >1s of silence
                perf_annotations = perf_annotations - trim_offset
                audio_data, sr = sf.read(str(perf_audio))
                start_sample = int(trim_offset * sr)
                audio_trimmed = audio_data[start_sample:]
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp.name, audio_trimmed, sr)
                trimmed_audio_path = Path(tmp.name)
                perf_audio = trimmed_audio_path

        try:
            if config.method == "offline":
                result = run_offline_alignment(
                    score_xml,
                    perf_audio,
                    match_file,
                    config,
                    use_musical_beat,
                    perf_annotations=perf_annotations,
                    granularity=granularity,
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
                    perf_annotations=perf_annotations,
                    granularity=granularity,
                )
        except Exception as e:
            print(f"Error: {e}")
            continue
        finally:
            if trimmed_audio_path is not None:
                trimmed_audio_path.unlink(missing_ok=True)

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


def main(args):
    dataset_type = args.dataset
    method = args.method
    dry_run = args.dry_run
    wandb = args.wandb
    granularity = args.granularity or "note"

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
            config.dataset, config, run_dir, dry_run, granularity
        )

        if not dry_run:
            save_config(config, run_dir)

            # save individual results
            save_results_to_csv(
                results, save_path=(run_dir / f"test_results.tsv").as_posix()
            )

            # Compute event-wise pooled summary (both all and tracked-only)
            summary_all = compute_event_pooled_summary(
                results, run_dir, config, tracked_only=False
            )
            summary_tracked = compute_event_pooled_summary(
                results, run_dir, config, tracked_only=True
            )
            for label, s in [("all", summary_all), ("tracked", summary_tracked)]:
                path = run_dir / f"summary_{label}.json"
                with open(path, "w") as f:
                    json.dump(s, f, indent=4)
                print(f"Results saved to: {path}")

        if not dry_run and wandb:
            report_results_to_wandb(summary_tracked, config)


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
    args = parser.parse_args()

    main(args)
