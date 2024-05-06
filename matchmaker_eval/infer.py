import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import partitura
from eval import run_evaluation, run_score_following, run_offline_alignment
from utils import (
    initialize_config,
    save_config,
    convert_score_to_audio,
    save_score_following_result,
)

DEFAULT_AUDIO_PATH = "./resources/ex_VuV01M.wav"
DEFAULT_MIDI_PATH = "./resources/ex_midi_score.mid"
WORKING_DIR = Path(__file__).parent.parent
OUTPUT_DIR = WORKING_DIR / "output"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio", type=str, help="path to metadata file", default=DEFAULT_AUDIO_PATH
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation with annotation file and score following result",
        default=False,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--score", dest="score_path", help="path to score file")
    group.add_argument(
        "--midi", dest="midi_path", help="path to midi file", default=DEFAULT_MIDI_PATH
    )
    args = parser.parse_args()

    target_audio = Path(args.audio)
    dir_path = target_audio.parent
    midi_path = None
    if args.score_path:
        score_path = Path(args.score_path)
        print(f"Score path: {score_path}")
        file_extension = score_path.suffix
        if file_extension.lower() in {".xml", ".musicxml"}:
            score = partitura.load_score(score_path.as_posix())
            midi_path = score_path.parent / "tmp_midi_score.mid"
            print(f"Saving score as midi: {midi_path}")
            partitura.save_score_midi(score, midi_path.as_posix())
    else:
        midi_path = Path(args.midi_path)

    assert midi_path.exists()

    config = initialize_config()
    # score_audio_path = Path(
    #     "/Users/jiyun/workspace/ismir2024_matchmaker/matchmaker_eval/score_audio.wav"
    # )
    score_audio_path = midi_path.with_suffix(".wav")  # "ex_midi_score.wav"
    if not score_audio_path.exists():
        score_audio_path = convert_score_to_audio(
            midi_path, score_audio_path, config.sample_rate
        )

    print(f"Midi path: {midi_path}")
    print(f"Score audio path: {score_audio_path}")

    # Run score following & save result
    model, wp = run_score_following(
        score_audio_path.as_posix(), target_audio.as_posix(), config
    )
    # wp = run_offline_alignment(score_audio_path, target_audio, config)
    # save results
    save_dir = (
        OUTPUT_DIR / f"infer_results_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, save_dir)

    # score_beat_ann = "/Users/jiyun/workspace/ismir2024_matchmaker/matchmaker_eval/score_annotations.tsv"
    # perf_beat_ann = "/Users/jiyun/workspace/ismir2024_matchmaker/matchmaker_eval/perf_annotations.tsv"
    score_beat_ann = dir_path / f"{midi_path.stem}_annotations.txt"
    perf_beat_ann = dir_path / f"{target_audio.stem}_annotations.txt"

    # Run evaluation
    result_stats = run_evaluation(
        model.warping_path, score_beat_ann, perf_beat_ann, config.frame_rate
    )
    results = {"Name": target_audio.stem}
    for k, v in config.model_dump(include=config.attr_exp).items():
        results[k] = v
    results |= result_stats
    print(f"Results for {target_audio.stem}")
    print(json.dumps(results, indent=4))

    # Save results to JSON file
    results_file = save_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {results_file}")

    save_score_following_result(
        model, save_dir, score_beat_ann, perf_beat_ann, config.frame_rate
    )


if __name__ == "__main__":
    main()
