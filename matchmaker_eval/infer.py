import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import partitura
import scipy
from eval import run_evaluation, run_score_following, run_offline_alignment
from utils import (
    initialize_config,
    save_config,
    convert_score_to_audio,
    save_nparray_to_csv,
)

DEFAULT_AUDIO_PATH = "./resources/ex_VuV01M.wav"
DEFAULT_MIDI_PATH = "./resources/ex_midi_score.mid"
WORKING_DIR = Path(__file__).parent.parent
OUTPUT_DIR = WORKING_DIR / "output"


def save_score_following_result(model, save_dir, score_ann, target_ann, frame_rate):
    save_path = save_dir / "online_results.tsv"
    save_nparray_to_csv(model.warping_path.T, save_path.as_posix())

    dist = scipy.spatial.distance.cdist(
        model.reference_features,
        model.input_features[: model.warping_path[1][-1]],
        metric=model.local_cost_fun,
    )  # [d, wy]
    plt.figure(figsize=(20, 20))
    plt.imshow(dist, aspect="auto", origin="lower", interpolation="nearest")
    plt.title(
        f"[{save_dir.name}] \n Matchmaker alignment path with ground-truth labels",
        fontsize=25,
    )
    plt.xlabel("Performance Audio frame", fontsize=25)
    plt.ylabel("Score Audio frame", fontsize=25)

    # plot online DTW path
    ref_paths, target_paths = model.warping_path[0], model.warping_path[1]
    for n in range(len(ref_paths)):
        plt.plot(target_paths[n], ref_paths[n], ".", color="purple", alpha=0.5)

    # plot ground-truth labels
    ref_annots = pd.read_csv(filepath_or_buffer=score_ann, delimiter="\t", header=None)[
        0
    ]
    target_annots = pd.read_csv(
        filepath_or_buffer=target_ann, delimiter="\t", header=None
    )[0]
    for ref, target in zip(ref_annots, target_annots):
        plt.plot(target * frame_rate, ref * frame_rate, "o", color="r", alpha=0.2)
    plt.savefig(save_dir / "online_dtw_path.png")


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

    score_beat_ann = dir_path / f"{midi_path.stem}_annotations.txt"
    target_beat_ann = dir_path / f"{target_audio.stem}_annotations.txt"
    save_score_following_result(
        model, save_dir, score_beat_ann, target_beat_ann, config.frame_rate
    )

    # Run evaluation
    result_stats = run_evaluation(
        model.warping_path, score_beat_ann, target_beat_ann, config.frame_rate
    )
    results = {"Name": target_audio.stem}
    for k, v in config.model_dump(include=config.attr_exp).items():
        results[k] = v
    results |= result_stats
    print(f"Results for {target_audio.stem}")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
