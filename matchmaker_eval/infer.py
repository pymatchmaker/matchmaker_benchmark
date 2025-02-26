import argparse
import json
from datetime import datetime
from pathlib import Path

from matchmaker import Matchmaker
from utils import (
    initialize_config,
    save_config,
    save_nparray_to_csv,
    save_score_following_result,
)

DEFAULT_SCORE_PATH = "./resources/ex_score.mid"
DEFAULT_PERF_PATH = "./resources/ex_VuV01M.wav"
DEFAULT_PERF_ANNOTS_PATH = "./resources/ex_VuV01M_annotations.txt"
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
        default=DEFAULT_PERF_PATH,
    )
    parser.add_argument(
        "--perf-annots",
        dest="perf_annots_path",
        help="path to performance annotation file (.txt)",
        default=DEFAULT_PERF_ANNOTS_PATH,
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
    perf_annots_path = Path(args.perf_annots_path) if args.perf_annots_path else ""
    print(f"Performance annotation path: {perf_annots_path}")

    config = initialize_config()
    print(f"Config: {config}")

    # Run score following & save result
    mm = Matchmaker(
        score_file=score_path,
        performance_file=perf_path,
        input_type="audio",
        method=config.method,
        distance_func=config.distance_func,
        frame_rate=config.frame_rate,
        sample_rate=config.sample_rate,
        feature_type=config.feature_type,
    )
    for current_position in mm.run():
        print(f"Current position: {current_position}")

    wp = mm.score_follower.warping_path
    result_stats = mm.run_evaluation(perf_annots_path)

    save_dir = (
        OUTPUT_DIR / f"infer_results_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, save_dir)

    # Save warping path
    save_path = save_dir / f"wp_results.tsv"
    save_nparray_to_csv(wp.T, save_path.as_posix())

    results = {"Name": perf_path.stem}
    for k, v in config.model_dump(include=config.attr_infer).items():
        results[k] = v
    results |= result_stats
    print(f"Results for {perf_path.stem}")
    print(json.dumps(results, indent=4))

    # Save evaluation results to JSON file
    results_file = save_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {results_file}")

    score_annots = mm.build_score_annotations()
    save_score_following_result(
        mm.score_follower, save_dir, score_annots, perf_annots_path, config.frame_rate
    )


if __name__ == "__main__":
    main()
