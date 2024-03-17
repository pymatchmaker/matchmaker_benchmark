import argparse
import csv
from pathlib import Path

import partitura
from eval import convert_score_to_audio, run_evaluation, run_score_following
from tabulate import tabulate


def save_score_following_result(wp, save_path: str):
    """Save score following result to file.

    Parameters
    ----------
    wp : np.array with shape (n, 2)
        array of warping path.
    save_path : str
        path to save score following result.
    """
    with open(save_path, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerows(wp.T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio", type=str, help="path to metadata file", required=True
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation with annotation file and score following result",
        default=False,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--score", dest="score_path", help="path to score file")
    group.add_argument("--midi", dest="midi_path", help="path to midi file")
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

    score_audio_path = midi_path.with_suffix(".wav")  # "ex_midi_score.wav"
    if not score_audio_path.exists():
        score_audio_path = convert_score_to_audio(
            midi_path=midi_path,
            save_path=score_audio_path,
        )

    print(f"Midi path: {midi_path}")
    print(f"Score audio path: {score_audio_path}")

    # Run score following & save result
    wp = run_score_following(score_audio_path.as_posix(), target_audio.as_posix())
    save_score_following_result(wp=wp, save_path="./wp-result.tsv")

    # Run evaluation
    if args.eval:
        score_beat_ann = dir_path / "ex_midi_score_annotations.txt"
        target_beat_ann = dir_path / f"{target_audio.stem}_annotations.txt"
        results = {"Name": target_audio.stem}
        result_stats = run_evaluation(wp, score_beat_ann, target_beat_ann)

        results |= result_stats

        print(
            tabulate([results], headers="keys", tablefmt="fancy_grid", showindex=True)
        )


if __name__ == "__main__":
    main()
