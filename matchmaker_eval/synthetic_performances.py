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

from partitura.io.exportaudio import save_wav_fluidsynth

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


def insert_neighbourhood_pitch_duration(onsets, note_array, rng, t=1):
    # get the pitches and durations of the neighboring notes in note_array within a time window of t seconds from each onset in onsets
    inserted_pitches = []
    inserted_durations = []
    for onset in onsets:
        neighbor_pitches = []
        neighbor_durations = []
        neighbors = note_array[
            (note_array["onset_sec"] >= onset - t) & (note_array["onset_sec"] <= onset + t)
        ]
        for neighbor in neighbors:
            neighbor_pitches.append(neighbor["pitch"])
            neighbor_durations.append(neighbor["duration_sec"])
        inserted_pitches.append(np.mean(neighbor_pitches).astype(int) if neighbor_pitches else rng.randint(21, 108))
        inserted_durations.append(np.mean(neighbor_durations) if neighbor_durations else rng.uniform(low=note_array["duration_sec"].min(), high=note_array["duration_sec"].max(), size=1)[0])
    return inserted_pitches, inserted_durations

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
        
        inserted_onsets = rng.uniform(
            low=first_ponset,
            high=last_ponset,
            size=n_insertions,
        )
        # inserted_pitch = rng.randint(
        #     low=21,
        #     high=108,
        #     size=n_insertions,
        # ).astype(int)

        # inserted_durations = rng.uniform(
        #     low=pnote_array["duration_sec"].min(),
        #     high=pnote_array["duration_sec"].max(),
        #     size=n_insertions,
        # )

        # insert pitches and durations based on the neighboring notes in pnote_array
        inserted_pitch, inserted_durations = insert_neighbourhood_pitch_duration(inserted_onsets, pnote_array, rng, t=1)

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

def generate_inserted_repetitions(
    spart: Part,
    parameters: np.ndarray,
    alignment: List[dict],
    snote_ids: List[str],
    unique_onset_idxs: List[np.ndarray],
    unique_onsets: np.ndarray,
    bpm: float,
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

    pnote_array = timing_perf.note_array()
    first_ponset = pnote_array["onset_sec"].min()
    last_ponset = pnote_array["onset_sec"].max()

    # choose a random note_id from pnote_array which will mark the start of the repetition
    repetition_start_note_id = rng.choice(pnote_array["id"])
    repetition_start_note_id_onset = pnote_array[pnote_array["id"] == repetition_start_note_id]["onset_sec"][0]

    # choose a random second between repetition_start_note_id_onset and the last_ponset
    repetition_start_time = rng.uniform(
        low=repetition_start_note_id_onset,
        high=last_ponset,
    )

    # make a subset of pnote_array which contains all notes from the first_ponset to repetition_start_time
    init_subset = pnote_array[
        (pnote_array["onset_sec"] >= first_ponset) & (pnote_array["onset_sec"] <= repetition_start_time)
    ]

    initial_onsets = init_subset["onset_sec"].tolist()
    initial_durations = init_subset["duration_sec"].tolist()
    initial_pitch = init_subset["pitch"].tolist()
    initial_velocities = init_subset["velocity"].tolist()
    initial_ids = init_subset["id"].tolist()

    initial_notes = [
        dict(
            id=initid,
            note_on=non,
            note_off=non + ndur,
            midi_pitch=pitch,
            velocity=vel,
        )
        for initid, non, ndur, pitch, vel in zip(
            initial_ids,
            initial_onsets,
            initial_durations,
            initial_pitch,
            initial_velocities,
        )
    ]

    out_alignment = out_alignment[:len(initial_notes)]
    
    # make a subset of pnote_array which contains all notes that start after the repetition_start_note_id_onset
    repetition_subset = pnote_array[pnote_array["onset_sec"] >= repetition_start_note_id_onset]

    inserted_onsets = []
    inserted_durations = []
    inserted_pitch = []
    inserted_velocities = []
    original_ids = []
    
    for i in range(len(repetition_subset)):
        note = repetition_subset[i]
        old_onset = note["onset_sec"]
        new_onset = old_onset - repetition_start_note_id_onset + repetition_start_time + 0.5  # add a small offset to avoid exact overlap
        repetition_subset[i]["onset_sec"] = new_onset
        inserted_onsets.append(new_onset)
        inserted_durations.append(note["duration_sec"])
        inserted_pitch.append(note["pitch"])
        inserted_velocities.append(note["velocity"])
        original_ids.append(note["id"])
        if repetition_subset[i]["id"] in initial_ids:
            last_digit = int(repetition_subset[i]["id"][-1])
            new_id = repetition_subset[i]["id"][:-1] + str(last_digit + 1)
            repetition_subset[i]["id"] = new_id
                
    
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
            label="match",
            score_id=original_ids[i],
            performance_id=repetition_subset[i]["id"],
        )
        for i in range(len(repetition_subset))
    ]

    final_note_array = np.concatenate([init_subset, repetition_subset])
    
    # sort final_note_array by onset_sec
    final_note_array = final_note_array[np.argsort(final_note_array["onset_sec"])]
    
    final_ppart = PerformedPart.from_note_array(final_note_array)
    
    return final_ppart, out_alignment

def generate_fumble(
    note_array: np.ndarray,
    fumble_id: str,
    fumble_extent: float,
    alignment: List[dict],
) -> np.ndarray:
    
    fumble_point = note_array[note_array["id"] == fumble_id]["onset_sec"][0]

    # make a subset of note_array which contains all notes that start before fumble_point + fumble_extent seconds
    pre_fumble_subset = note_array[
        note_array["onset_sec"] < fumble_point + fumble_extent
    ]
    
    pre_fumble_alignment = alignment[: len(pre_fumble_subset)]
    post_fumble_alignment = alignment[len(pre_fumble_subset) :]

    # make a subset of note_array which contains all notes that start within fumble_extent seconds of fumble_point
    fumble_subset = note_array[
        (note_array["onset_sec"] >= fumble_point) & (note_array["onset_sec"] < fumble_point + fumble_extent)
    ]

    # nudge the onsets of the notes in fumble_subset by fumble_extent seconds
    fumble_subset["onset_sec"] += fumble_extent

    original_ids = fumble_subset["id"].tolist()
    fumbled_ids = [f"{oid}_fumble" for oid in original_ids]
    fumble_subset["id"] = fumbled_ids

    fumble_alignment = []
    for oid, fid in zip(original_ids, fumbled_ids):
        fumble_alignment.append(
            dict(
                label='match',
                score_id=oid,
                performance_id=fid,
            )        
        )

    # nudge the onsets of the notes in note_array after fumble_point + fumble_extent by fumble_extent seconds
    post_fumble_subset = note_array[
        note_array["onset_sec"] >= fumble_point + fumble_extent
    ]
    post_fumble_subset["onset_sec"] += fumble_extent

    final_note_array = np.concatenate([pre_fumble_subset, fumble_subset, post_fumble_subset])
    final_note_array = final_note_array[np.argsort(final_note_array["onset_sec"])]

    final_alignment = pre_fumble_alignment + fumble_alignment + post_fumble_alignment
    
    return final_note_array, final_alignment


def generate_multiple_fumbles(
    spart: Part,
    parameters: np.ndarray,
    alignment: List[dict],
    snote_ids: List[str],
    unique_onset_idxs: List[np.ndarray],
    unique_onsets: np.ndarray,
    rng: np.random.RandomState = RNG,
) -> PerformedPart:
    
    snote_array = spart.note_array()
    tempo_beat_period = 60 / (120 * np.ones_like(unique_onsets))
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

    pnote_array = timing_perf.note_array()
    first_ponset = pnote_array["onset_sec"].min()
    last_ponset = pnote_array["onset_sec"].max()

    # equally distribute 5 points in time between first_ponset and last_ponset to be the fumble points.
    n_fumbles = 5
    fumble_points = np.linspace(
        first_ponset + (last_ponset - first_ponset) / (n_fumbles + 1),
        last_ponset - (last_ponset - first_ponset) / (n_fumbles + 1),
        n_fumbles,
    )

    fumble_extent = 1  # seconds

    # find the note ids in pnote_array that occur on or immediately after each fumble point
    fumble_nids = []
    for fpoint in fumble_points:
        fumble_nid = pnote_array[
            pnote_array["onset_sec"] >= fpoint
        ]["id"][0]
        fumble_nids.append(fumble_nid)

    final_note_array = pnote_array.copy()

    for fumble_nid in fumble_nids:
        final_note_array, out_alignment = generate_fumble(
            note_array=final_note_array,
            fumble_id=fumble_nid,
            fumble_extent=fumble_extent,
            alignment=out_alignment,
        )
    
    final_ppart = PerformedPart.from_note_array(final_note_array)

    return final_ppart, out_alignment

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
        for freq in [0.5]: #[0.0625, 0.125, 0.25, 0.5]:
            for min_tempo in [120]: #[30, 60, 90]:
                for max_tempo in [145]: #[100, 150, 200]:
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
        for ftype in ["lin"]:#["exp", "log", 1 / 3, 0.5, 2, 3]:
            for min_tempo in [120]:#[30, 60, 90]:
                for max_tempo in [145]: #[100, 150, 200]:
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
    for bpm in [120, 140]:
        for insertion_ratio in [0.0]:
            for deletion_ratio in [0.0, 0.01, 0.05, 0.1, 0.2]:

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

def generate_repetition(
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

    bpm = 120

    out_name = f"{piece_name}-repetition-bpm_{bpm}"

    print(f"generating {out_name}")

    if os.path.exists(os.path.join(out_dir_audio, f"{out_name}.wav")):
        return

    gen_perf, output_alignment = generate_inserted_repetitions(
            spart=spart,
            parameters=parameters,
            alignment=alignment,
            snote_ids=snote_ids,
            unique_onset_idxs=unique_onset_idxs,
            unique_onsets=unique_onsets,
            bpm=120,
            rng=RNG,
        )
    
    save_wav_fluidsynth(
        input_data=gen_perf,
        out=os.path.join(out_dir_audio, f"{out_name}.wav"),
    )

    # TODO: Update Match files to handle many-to-one alignments
    # pt.save_match(
    #     alignment=output_alignment,
    #     performance_data=gen_perf,
    #     score_data=spart,
    #     out=os.path.join(out_dir_match, f"{out_name}.match"),
    #     assume_unfolded=True,
    # )
    
    pt.save_performance_midi(
        performance_data=gen_perf,
        out=os.path.join(out_dir_midi, f"{out_name}.mid"),
    )

    np.savez_compressed(
        os.path.join(out_dir_match, f"{out_name}.npz"),
        alignment=output_alignment,
        performance_note_array=gen_perf.note_array(),
        score_note_array=spart.note_array(),
    )

    pt.save_parangonada_csv(
        alignment=output_alignment,
        performance_data=gen_perf,
        score_data=spart,
        outdir=out_dir_match
    )

def generate_fumbles(
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

    out_name = f"{piece_name}-fumbles"

    print(f"generating {out_name}")

    if os.path.exists(os.path.join(out_dir_audio, f"{out_name}.wav")):
        return
    
    gen_perf, output_alignment = generate_multiple_fumbles(
        spart=spart,
        parameters=parameters,
        alignment=alignment,
        snote_ids=snote_ids,
        unique_onset_idxs=unique_onset_idxs,
        unique_onsets=unique_onsets,
    )

    save_wav_fluidsynth(
        input_data=gen_perf,
        out=os.path.join(out_dir_audio, f"{out_name}.wav"),
    )

    # TODO: Update Match files to handle many-to-one alignments
    # pt.save_match(
    #     alignment=output_alignment,
    #     performance_data=gen_perf,
    #     score_data=spart,
    #     out=os.path.join(out_dir_match, f"{out_name}.match"),
    #     assume_unfolded=True,
    # )
    
    pt.save_performance_midi(
        performance_data=gen_perf,
        out=os.path.join(out_dir_midi, f"{out_name}.mid"),
    )

    np.savez_compressed(
        os.path.join(out_dir_match, f"{out_name}.npz"),
        alignment=output_alignment,
        performance_note_array=gen_perf.note_array(),
        score_note_array=spart.note_array(),
    )

    pt.save_parangonada_csv(
        alignment=output_alignment,
        performance_data=gen_perf,
        score_data=spart,
        outdir=out_dir_match
    )

def generate_unchanged(
    out_dir_audio,
    out_dir_match,
    piece_name,
    spart,
    alignment,
    parameters,
    snote_ids,
    unique_onset_idxs,
    unique_onsets,
) -> None:

    out_name = f"{piece_name}-unchanged"

    print(f"generating {out_name}")

    if os.path.exists(os.path.join(out_dir_audio, f"{out_name}.wav")):
        return
    
    snote_array = spart.note_array()
    tempo_beat_period = 60 / (120 * np.ones_like(unique_onsets))
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

    save_wav_fluidsynth(
        input_data=timing_perf,
        out=os.path.join(out_dir_audio, f"{out_name}.wav"),
    )

    pt.save_match(
        alignment=alignment,
        performance_data=timing_perf,
        score_data=spart,
        out=os.path.join(out_dir_match, f"{out_name}.match"),
        assume_unfolded=True,
    )

    pt.save_performance_midi(
        performance_data=timing_perf,
        out=os.path.join(out_dir_midi, f"{out_name}.mid"),
    )

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("Generate synthetic performances")

    parser.add_argument("--filename", "-f", default=None, help="Score in MusicXML format",
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

    elif args.type == 5:

        generate_repetition(
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

    elif args.type == 6:

        generate_fumbles(
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

    elif args.type == 7:

        generate_unchanged(
            out_dir_audio,
            out_dir_match,
            piece_name,
            spart,
            alignment,
            parameters,
            snote_ids,
            unique_onset_idxs,
            unique_onsets,
        )