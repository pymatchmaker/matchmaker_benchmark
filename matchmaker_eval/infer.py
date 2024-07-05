import argparse
import json
from datetime import datetime
from pathlib import Path

from eval import run_evaluation, run_score_following
from utils import (
    convert_score_to_audio,
    initialize_config,
    save_config,
    save_nparray_to_csv,
    save_score_following_result,
)

DEFAULT_SCORE_PATH = "./resources/ex_score.mid"
DEFAULT_PERF_PATH = "./resources/ex_VuV01M.wav"
WORKING_DIR = Path(__file__).parent.parent
OUTPUT_DIR = WORKING_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score",
        dest="score_path",
        help="path to score file (.musicxml or .mid)",
        default=DEFAULT_SCORE_PATH,
    )
    parser.add_argument(
        "--perf",
        dest="perf_path",
        help="path to performance file (.wav or .mid) or empty for live performance mode",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help=f"Run evaluation with annotation file ('_annotations.txt' file in the same dir)",
        default=False,
    )
    args = parser.parse_args()

    score_path = Path(args.score_path)
    print(f"Score path: {score_path}")
    assert score_path.exists()

    perf_path = Path(args.perf_path) if args.perf_path else ""
    print(
        f"Performance path: {perf_path if perf_path else 'live performance mode (mic input)'}"
    )

    config = initialize_config()

    # Convert score to audio
    score_audio_path = convert_score_to_audio(score_path, config.sample_rate)

    print(f"Score path: {score_path}")
    print(f"Score audio path: {score_audio_path}")

    # Run score following & save result
    model, wp = run_score_following(score_audio_path, perf_path, config)
    save_dir = (
        OUTPUT_DIR / f"infer_results_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, save_dir)

    # Save warping path
    save_path = save_dir / f"wp_results.tsv"
    save_nparray_to_csv(wp.T, save_path.as_posix())

    if not args.eval:
        return

    # Retrieve beat annotations
    dir_path = score_path.parent
    score_beat_ann = dir_path / f"{score_path.stem}_annotations.txt"
    perf_beat_ann = dir_path / f"{perf_path.stem}_annotations.txt"

    # Run evaluation
    result_stats = run_evaluation(
        model.warping_path, score_beat_ann, perf_beat_ann, config.frame_rate
    )
    results = {"Name": perf_path.stem}
    for k, v in config.model_dump(include=config.attr_exp).items():
        results[k] = v
    results |= result_stats
    print(f"Results for {perf_path.stem}")
    print(json.dumps(results, indent=4))

    # Save evaluation results to JSON file
    results_file = save_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {results_file}")

    save_score_following_result(
        model, save_dir, score_beat_ann, perf_beat_ann, config.frame_rate
    )

    score_audio_path.unlink()


if __name__ == "__main__":
    main()
