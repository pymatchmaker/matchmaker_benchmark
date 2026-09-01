import numpy as np
import partitura as pt
from partitura.score import unfold_part_maximal, Part, remove_grace_notes
from partitura.performance import PerformedPart, PerformedNote
from partitura.utils.music import performance_from_part
from partitura.musicanalysis.performance_codec import (
    encode_performance,
    decode_performance,
    onsetwise_to_notewise,
)
from typing import List, Tuple

from matchmaker.utils.symbolic import save_wav_fluidsynth

import matplotlib.pyplot as plt

import warnings
import os

warnings.filterwarnings("ignore", module="partitura")

RNG = np.random.RandomState(1984)


def minmax_array(
    x: np.ndarray,
    max_val: float = 1,
    min_val: float = 0,
) -> np.ndarray:

    norm_x = (max_val - min_val) * (x - x.min()) / (x.max() - x.min()) + min_val

    return norm_x


def descending_minmax_array(
    x: np.ndarray,
    max_val: float = 1,
    min_val: float = 0,
) -> np.ndarray:

    norm_x = (max_val - min_val) * (1 - minmax_array(x)) + min_val

    return norm_x


def oscillating_step_func(
    x: np.ndarray,
    freq: float,
    normalize: bool = True,
    max_val: float = 168,
    min_val: float = 30,
) -> np.ndarray:

    step_fun = np.sign(np.sin(2 * np.pi * freq * x))

    step_fun[step_fun == 0] = 1

    if normalize:
        step_fun = minmax_array(
            step_fun,
            max_val=max_val,
            min_val=min_val,
        )

    return step_fun


def generate_oscillating_tempo_perf(
    spart: Part,
    parameters: np.ndarray,
    alignment: List[dict],
    snote_ids: List[str],
    unique_onset_idxs: List[np.ndarray],
    unique_onsets: np.ndarray,
    freq: float,
    max_tempo: float,
    min_tempo: float,
    ftype: str = "sine",
) -> Tuple[PerformedPart, List[dict]]:

    if ftype == "sine":
        oscillating_tempo_bpm = (max_tempo - min_tempo) * 0.5 * (
            np.sin(2 * np.pi * freq * unique_onsets) + 1
        ) + min_tempo

    elif ftype == "step":
        oscillating_tempo_bpm = oscillating_step_func(
            x=unique_onsets,
            freq=freq,
            max_val=max_tempo,
            min_val=min_tempo,
        )

    oscillating_beat_period = 60 / oscillating_tempo_bpm

    parameters_oscillating_tempo = parameters.copy()
    parameters_oscillating_tempo["beat_period"] = onsetwise_to_notewise(
        oscillating_beat_period,
        unique_onset_idxs=unique_onset_idxs,
    )

    oscillating_perf = decode_performance(
        score=spart,
        performance_array=parameters_oscillating_tempo,
        snote_ids=snote_ids,
    )

    out_alignment = alignment.copy()

    return oscillating_perf, out_alignment


def generate_speeding_tempo_perf(
    spart: Part,
    parameters: np.ndarray,
    alignment: List[dict],
    snote_ids: List[str],
    unique_onset_idxs: List[np.ndarray],
    unique_onsets: np.ndarray,
    max_tempo: float,
    min_tempo: float,
    ftype: str = "lin",
    ptype: str = "accel",
) -> PerformedPart:

    norm_func = minmax_array if ptype == "accel" else descending_minmax_array

    norm_tempo = norm_func(unique_onsets)

    def rescale(norm_tempo: np.ndarray) -> np.ndarray:
        return (max_tempo - min_tempo) * norm_tempo + min_tempo

    if ftype == "lin":
        tempo_bpm = rescale(norm_tempo=norm_tempo)
    elif ftype == "exp":
        tempo_bpm = rescale(norm_tempo=minmax_array(np.exp(norm_tempo)))
    elif ftype == "log":
        tempo_bpm = rescale(norm_tempo=minmax_array(np.log1p(norm_tempo)))
    elif isinstance(ftype, (float, int)):
        tempo_bpm = rescale(norm_tempo=norm_tempo**ftype)

    tempo_beat_period = 60 / tempo_bpm

    parameters_gen_tempo = parameters.copy()
    parameters_gen_tempo["beat_period"] = onsetwise_to_notewise(
        tempo_beat_period,
        unique_onset_idxs=unique_onset_idxs,
    )

    speeding_perf = decode_performance(
        score=spart,
        performance_array=parameters_gen_tempo,
        snote_ids=snote_ids,
    )

    out_alignment = alignment.copy()
    return speeding_perf, out_alignment


def generate_random_timing_perf(
    spart: Part,
    parameters: np.ndarray,
    alignment: List[dict],
    snote_ids: List[str],
    unique_onset_idxs: List[np.ndarray],
    unique_onsets: np.ndarray,
    bpm: float,
    timing_spread: float,
    rng: np.random.RandomState = RNG,
) -> PerformedPart:

    tempo_beat_period = 60 / (bpm * np.ones_like(unique_onsets))
    parameters_gen_timing = parameters.copy()
    parameters_gen_timing["beat_period"] = onsetwise_to_notewise(
        tempo_beat_period,
        unique_onset_idxs=unique_onset_idxs,
    )

    parameters_gen_timing["timing"] = rng.uniform(
        low=-timing_spread,
        high=timing_spread,
        size=len(parameters),
    )

    timing_perf = decode_performance(
        score=spart,
        performance_array=parameters_gen_timing,
        snote_ids=snote_ids,
    )

    out_alignment = alignment.copy()
    return timing_perf, out_alignment


def generate_insertions_deletions_perf(
    spart: Part,
    parameters: np.ndarray,
    alignment: List[dict],
    snote_ids: List[str],
    unique_onset_idxs: List[np.ndarray],
    unique_onsets: np.ndarray,
    bpm: float,
    insertion_ratio: float,
    deletion_ratio: float,
    rng: np.random.RandomState = RNG,
) -> PerformedPart:

    snote_array = spart.note_array()
    tempo_beat_period = 60 / (bpm * np.ones_like(unique_onsets))
    parameters_gen_timing = parameters.copy()
    parameters_gen_timing["beat_period"] = onsetwise_to_notewise(
        tempo_beat_period,
        unique_onset_idxs=unique_onset_idxs,
    )

    timing_perf = decode_performance(
        score=spart,
        performance_array=parameters_gen_timing,
        snote_ids=snote_ids,
    )

    out_alignment = alignment.copy()

    n_deletions = int(np.round(deletion_ratio * len(snote_array)))

    if n_deletions > 0:
        deleted_ids = rng.choice(
            snote_array["id"],
            size=n_deletions,
            replace=False,
        )

        timing_perf.notes = [n for n in timing_perf.notes if n["id"] not in deleted_ids]
        out_alignment = [al for al in alignment if al["score_id"] not in deleted_ids]
        out_alignment += [
            dict(
                label="deletion",
                score_id=delid,
            )
            for delid in deleted_ids
        ]

    n_insertions = int(np.round(insertion_ratio * len(snote_array)))

    if n_insertions > 0:

        pnote_array = timing_perf.note_array()
        first_ponset = pnote_array["onset_sec"].min()
        last_ponset = pnote_array["onset_sec"].max()
        inserted_pitch = rng.randint(
            low=21,
            high=108,
            size=n_insertions,
        ).astype(int)
        inserted_onsets = rng.uniform(
            low=first_ponset,
            high=last_ponset,
            size=n_insertions,
        )
        inserted_durations = rng.uniform(
            low=pnote_array["duration_sec"].min(),
            high=pnote_array["duration_sec"].max(),
            size=n_insertions,
        )

        inserted_velocities = rng.uniform(
            low=pnote_array["velocity"].min(),
            high=pnote_array["velocity"].max(),
            size=n_insertions,
        ).astype(int)

        inserted_notes = [
            dict(
                id=f"nin{i}",
                note_on=non,
                note_off=non + ndur,
                midi_pitch=pitch,
                velocity=vel,
            )
            for i, (non, ndur, pitch, vel) in enumerate(
                zip(
                    inserted_onsets,
                    inserted_durations,
                    inserted_pitch,
                    inserted_velocities,
                )
            )
        ]

        out_alignment += [
            dict(
                label="insertion",
                performance_id=pid["id"],
            )
            for pid in inserted_notes
        ]

        timing_perf.notes += inserted_notes

        timing_perf.notes.sort(key=lambda x: x["note_on"])

    return timing_perf, out_alignment


def generate_oscillating(
    out_dir_audio,
    out_dir_match,
    out_dir_midi,
    piece_name,
    spart,
    alignment,
    parameters,
    snote_ids,
    unique_onset_idxs,
    unique_onsets,
) -> None:
    for ftype in ["sine", "step"]:
        for freq in [0.0625, 0.125, 0.25, 0.5]:
            for min_tempo in [30, 60, 90]:
                for max_tempo in [100, 150, 200]:
                    out_name = f"{piece_name}-{ftype}-freq{freq:.4f}-min_tempo_{min_tempo}-max_tempo_{max_tempo}"

                    print(f"generating {out_name}")

                    if os.path.exists(os.path.join(out_dir_audio, f"{out_name}.wav")):
                        continue

                    oscillating_perf, gen_alignment = generate_oscillating_tempo_perf(
                        spart=spart,
                        parameters=parameters,
                        alignment=alignment,
                        snote_ids=snote_ids,
                        unique_onset_idxs=unique_onset_idxs,
                        unique_onsets=unique_onsets,
                        freq=freq,
                        max_tempo=max_tempo,
                        min_tempo=min_tempo,
                        ftype=ftype,
                    )

                    save_wav_fluidsynth(
                        input_data=oscillating_perf,
                        out=os.path.join(out_dir_audio, f"{out_name}.wav"),
                    )

                    pt.save_match(
                        alignment=gen_alignment,
                        performance_data=oscillating_perf,
                        score_data=spart,
                        out=os.path.join(out_dir_match, f"{out_name}.match"),
                        assume_unfolded=True,
                    )
                    pt.save_performance_midi(
                        performance_data=oscillating_perf,
                        out=os.path.join(out_dir_midi, f"{out_name}.mid"),
                    )


def generate_speeding(
    out_dir_audio,
    out_dir_match,
    out_dir_midi,
    piece_name,
    spart,
    alignment,
    snote_ids,
    parameters,
    unique_onset_idxs,
    unique_onsets,
) -> None:
    for ptype in ["accel", "rall"]:
        for ftype in ["lin", "exp", "log", 1 / 3, 0.5, 2, 3]:
            for min_tempo in [30, 60, 90]:
                for max_tempo in [100, 150, 200]:
                    ftype_str = ftype if isinstance(ftype, str) else f"poly{ftype:.3f}"
                    out_name = f"{piece_name}-{ptype}-ftype_{ftype_str}-min_tempo_{min_tempo}-max_tempo_{max_tempo}"

                    print(f"generating {out_name}")

                    if os.path.exists(os.path.join(out_dir_audio, f"{out_name}.wav")):
                        continue

                    speeding_perf, gen_alignment = generate_speeding_tempo_perf(
                        spart=spart,
                        parameters=parameters,
                        alignment=alignment,
                        snote_ids=snote_ids,
                        unique_onset_idxs=unique_onset_idxs,
                        unique_onsets=unique_onsets,
                        max_tempo=max_tempo,
                        min_tempo=min_tempo,
                        ftype=ftype,
                        ptype=ptype,
                    )
                    save_wav_fluidsynth(
                        input_data=speeding_perf,
                        out=os.path.join(
                            out_dir_audio,
                            f"{out_name}.wav",
                        ),
                    )
                    pt.save_match(
                        alignment=gen_alignment,
                        performance_data=speeding_perf,
                        score_data=spart,
                        out=os.path.join(
                            out_dir_match,
                            f"{out_name}.match",
                        ),
                        assume_unfolded=True,
                    )
                    pt.save_performance_midi(
                        performance_data=speeding_perf,
                        out=os.path.join(
                            out_dir_midi,
                            f"{out_name}.mid",
                        ),
                    )


def generate_timing(
    out_dir_audio,
    out_dir_match,
    out_dir_midi,
    piece_name,
    spart,
    alignment,
    parameters,
    snote_ids,
    unique_onset_idxs,
    unique_onsets,
) -> None:
    for bpm in [30, 60, 90, 120]:
        for timing_spread in [0.01, 0.03, 0.05, 0.1, 0.2]:
            out_name = (
                f"{piece_name}-timing-bpm_{bpm}-timing_spread_{timing_spread:.02f}"
            )

            print(f"generating {out_name}")

            if os.path.exists(os.path.join(out_dir_audio, f"{out_name}.wav")):
                continue

            timing_perf, gen_alignment = generate_random_timing_perf(
                spart=spart,
                parameters=parameters,
                alignment=alignment,
                snote_ids=snote_ids,
                unique_onset_idxs=unique_onset_idxs,
                unique_onsets=unique_onsets,
                bpm=bpm,
                timing_spread=timing_spread,
            )

            save_wav_fluidsynth(
                input_data=timing_perf,
                out=os.path.join(out_dir_audio, f"{out_name}.wav"),
            )
            pt.save_match(
                alignment=gen_alignment,
                performance_data=timing_perf,
                score_data=spart,
                out=os.path.join(out_dir_match, f"{out_name}.match"),
                assume_unfolded=True,
            )
            pt.save_performance_midi(
                performance_data=timing_perf,
                out=os.path.join(out_dir_midi, f"{out_name}.mid"),
            )


def generate_insdel(
    out_dir_audio,
    out_dir_match,
    out_dir_midi,
    piece_name,
    spart,
    alignment,
    parameters,
    snote_ids,
    unique_onset_idxs,
    unique_onsets,
) -> None:
    for bpm in [30, 60, 90, 120]:
        for insertion_ratio in [0.0, 0.05, 0.1, 0.15, 0.2]:
            for deletion_ratio in [0.0, 0.05, 0.1, 0.15, 0.2]:

                if insertion_ratio == 0 and deletion_ratio == 0:
                    continue

                out_name = f"{piece_name}-insdel-bpm_{bpm}-insertion_ratio_{insertion_ratio:.2f}-deletion_ratio_{deletion_ratio:.2f}"

                print(f"generating {out_name}")

                if os.path.exists(os.path.join(out_dir_audio, f"{out_name}.wav")):
                    continue

                insdel_perf, gen_alignment = generate_insertions_deletions_perf(
                    spart=spart,
                    parameters=parameters,
                    alignment=alignment,
                    snote_ids=snote_ids,
                    unique_onset_idxs=unique_onset_idxs,
                    unique_onsets=unique_onsets,
                    bpm=bpm,
                    insertion_ratio=insertion_ratio,
                    deletion_ratio=deletion_ratio,
                )

                save_wav_fluidsynth(
                    input_data=insdel_perf,
                    out=os.path.join(out_dir_audio, f"{out_name}.wav"),
                )

                pt.save_match(
                    alignment=gen_alignment,
                    performance_data=insdel_perf,
                    score_data=spart,
                    out=os.path.join(out_dir_match, f"{out_name}.match"),
                    assume_unfolded=True,
                )
                pt.save_performance_midi(
                    performance_data=insdel_perf,
                    out=os.path.join(out_dir_midi, f"{out_name}.mid"),
                )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("Generate synthetic performances")

    parser.add_argument(
        "--filename",
        "-f",
        default=None,
        help="Score in MusicXML format",
    )
    parser.add_argument(
        "--type",
        "-t",
        default=1,
        type=int,
        help="Generation type",
    )

    args = parser.parse_args()

    if args.filename is None:
        raise ValueError("No score given!")

    out_dir = "../synthetic_performances"

    out_dir_audio = os.path.join(out_dir, "audio")

    if not os.path.exists(out_dir_audio):
        os.mkdir(out_dir_audio)

    out_dir_match = os.path.join(out_dir, "match")

    if not os.path.exists(out_dir_match):
        os.mkdir(out_dir_match)

    out_dir_midi = os.path.join(out_dir, "midi")

    if not os.path.exists(out_dir_midi):
        os.mkdir(out_dir_midi)

    piece_name = os.path.basename(args.filename).replace(".musicxml", "")
    score = pt.load_score(args.filename)

    spart = unfold_part_maximal(score[0])

    # Remove grace notes
    remove_grace_notes(spart)

    snote_array = spart.note_array()

    ppart_deadpan = performance_from_part(part=spart, bpm=60, velocity=64)

    pnote_array = ppart_deadpan.note_array()

    alignment = [
        dict(
            label="match",
            score_id=sni,
            performance_id=sni,
        )
        for sni in snote_array["id"]
    ]

    parameters, snote_ids, unique_onset_idxs = encode_performance(
        score=spart,
        performance=ppart_deadpan,
        alignment=alignment,
        return_u_onset_idx=True,
    )

    unique_onsets = np.array(
        [snote_array[uix]["onset_beat"].mean() for uix in unique_onset_idxs]
    )

    if args.type == 1:
        generate_oscillating(
            out_dir_audio,
            out_dir_match,
            out_dir_midi,
            piece_name,
            spart,
            alignment,
            parameters,
            snote_ids,
            unique_onset_idxs,
            unique_onsets,
        )
    elif args.type == 2:

        generate_speeding(
            out_dir_audio,
            out_dir_match,
            out_dir_midi,
            piece_name,
            spart,
            alignment,
            snote_ids,
            parameters,
            unique_onset_idxs,
            unique_onsets,
        )

    elif args.type == 3:

        generate_timing(
            out_dir_audio,
            out_dir_match,
            out_dir_midi,
            piece_name,
            spart,
            alignment,
            parameters,
            snote_ids,
            unique_onset_idxs,
            unique_onsets,
        )

    elif args.type == 4:

        generate_insdel(
            out_dir_audio,
            out_dir_match,
            out_dir_midi,
            piece_name,
            spart,
            alignment,
            parameters,
            snote_ids,
            unique_onset_idxs,
            unique_onsets,
        )
