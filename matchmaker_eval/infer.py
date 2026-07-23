import argparse
import json
from datetime import datetime
from pathlib import Path

from eval import run_evaluation
from matchmaker import Matchmaker
from matchmaker.matchmaker import DEFAULT_KWARGS
from utils import (
    TOLERANCES_IN_BEATS,
    AudioEvalConfig,
    save_config,
    save_nparray_to_csv,
)

DEFAULT_SCORE_PATH = "./resources/ex_score.musicxml"
DEFAULT_PERF_PATH = "./resources/ex_VuV01M.wav"
DEFAULT_GT_PATH = "./resources/ex_VuV01M.match"
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
        "--gt",
        dest="gt_path",
        help="path to ground truth (.match file or .tsv with perf_sec/score_beat)",
        default=DEFAULT_GT_PATH,
    )
    parser.add_argument(
        "--method",
        dest="method",
        help="score following method (arzt, dixon, hmm, ...)",
        default="arzt",
    )
    args = parser.parse_args()

    score_path = Path(args.score_path)
    print(f"Score path: {score_path}")
    assert score_path.exists()

    perf_path = Path(args.perf_path) if args.perf_path else ""
    print(
        f"Performance path: {perf_path if perf_path else 'live performance mode (mic input)'}"
    )
    gt_path = Path(args.gt_path) if args.gt_path else ""
    print(f"Ground truth path: {gt_path}")

    method = args.method
    kw = DEFAULT_KWARGS.get("audio", {}).get(method, {})
    cfg_kwargs = {k: v for k, v in kw.items() if k in ("sample_rate", "frame_rate")}
    if "frame_rate" not in cfg_kwargs and "hop_length" in kw:
        sr = cfg_kwargs.get("sample_rate", 44100)
        cfg_kwargs["frame_rate"] = sr / kw["hop_length"]
    config = AudioEvalConfig(method=method, dataset="infer", **cfg_kwargs)
    print(f"Config: {config.model_dump(include=config.attr_exp)}")

    mm_kwargs = dict(kw)
    mm_kwargs["sample_rate"] = config.sample_rate
    mm_kwargs["frame_rate"] = config.frame_rate

    # Run score following & save result
    mm = Matchmaker(
        score_file=score_path,
        performance_file=perf_path,
        input_type="audio",
        method=method,
        kwargs=mm_kwargs,
    )
    for current_position in mm.run():
        print(f"Current position: {current_position}")

    wp = mm.score_follower.alignment_path

    save_dir = (
        OUTPUT_DIR / f"infer_results_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, save_dir)

    result_stats = run_evaluation(
        mm,
        gt=gt_path,
        tolerances=TOLERANCES_IN_BEATS,
        debug=True,
        save_dir=save_dir,
        run_name="infer",
    )

    # Save alignment path
    save_path = save_dir / f"wp_results.tsv"
    save_nparray_to_csv(wp.T, save_path.as_posix())

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


if __name__ == "__main__":
    main()
