#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Evaluate Symbolic score followers
"""
from datetime import datetime
import os
import glob
from pathlib import Path
import time
from typing import List
import numpy as np
import partitura as pt
import copy
import pandas as pd

import scipy
from scipy.stats import skew, skewtest, kurtosis
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from matchmaker.features.midi import (
    PianoRollProcessor,
    PitchClassPianoRollProcessor,
    PitchIOIProcessor,
)
from matchmaker.io.midi import MockFramedMidiStream, POLLING_PERIOD
from matchmaker.utils.misc import RECVQueue
from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArzt
from matchmaker.dp.oltw_dixon import OnlineTimeWarpingDixon
from matchmaker.prob.hmm import (
    PitchIOIHMM,
    BernoulliGaussianPitchIOIObservationModel,
    gumbel_transition_matrix,
    gumbel_init_dist,
    compute_ioi_matrix,
    compute_discrete_pitch_profiles,
)
from utils import save_nparray_to_csv, save_score_following_result

from matchmaker.utils.tempo_models import ReactiveTempoModel, KalmanTempoModel

from partitura.utils.music import performance_from_part
from partitura.musicanalysis.performance_codec import get_time_maps_from_alignment
from partitura.performance import PerformedPart

RNG = np.random.RandomState(1984)


def sanitize_warping_path(wp):
    wp1 = wp[0]
    unique_wp1 = np.unique(wp1)
    unique_wp1_idxs = [np.where(wp1 == ui)[0] for ui in unique_wp1]

    unique_wp2 = np.array([np.min(wp[1][ui]) for ui in unique_wp1_idxs])

    return np.array([unique_wp1, unique_wp2])


# def transfer_positions(wp, ref_ann, frame_rate):
#     """
#     Transfer the positions of the reference annotations to the target annotations using the warping path.

#     Parameters
#     ----------
#     wp : np.array with shape (2, T)
#         array of warping path.
#     ref_ann : List[float]
#         reference annotations.
#     frame_rate : float
#         frame rate of the annotations to convert frame index to time.
#     """
#     x, y = wp[0] / frame_rate, wp[1] / frame_rate
#     f = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate",)
#     target_ann = f(ref_ann)
#     return target_ann


def transfer_positions(wp, ref_anns, frame_rate, is_hmm=False, state_space=None):
    """
    Transfer the positions of the reference annotations to the target annotations using the warping path.

    Parameters
    ----------
    wp : np.array with shape (2, T)
        array of warping path.
    ref_ann : List[float]
        reference annotations.
    """
    if is_hmm:
        x = state_space[wp[0]]
        y = wp[1] / frame_rate
    else:
        x, y = wp[0] / frame_rate, wp[1] / frame_rate

    pred_idxs = [np.where(x >= r)[0] for r in ref_anns]
    predicted_targets = [
        y[idxs.min()] if len(idxs) > 0 else np.nan for idxs in pred_idxs
    ]
    return predicted_targets


def evaluate_alignment(target_ponsets, tracked_ponsets):
    # in ms
    asynchrony = (target_ponsets - tracked_ponsets) * 1000
    abs_asynch = abs(asynchrony)
    mean_asynch = np.nanmean(abs_asynch)
    median_asynch = np.nanmedian(abs_asynch)
    std_asynch = np.nanstd(abs_asynch)
    skewness = skew(asynchrony[~np.isnan(asynchrony)])
    kurt = kurtosis(asynchrony[~np.isnan(asynchrony)])

    lt_25ms = np.sum(abs_asynch[~np.isnan(asynchrony)] <= 25) / len(target_ponsets)
    lt_50ms = np.sum(abs_asynch[~np.isnan(asynchrony)] <= 50) / len(target_ponsets)
    lt_100ms = np.sum(abs_asynch[~np.isnan(asynchrony)] <= 100) / len(target_ponsets)
    lt_200ms = np.sum(abs_asynch[~np.isnan(asynchrony)] <= 200) / len(target_ponsets)
    lt_300ms = np.sum(abs_asynch[~np.isnan(asynchrony)] <= 300) / len(target_ponsets)
    lt_500ms = np.sum(abs_asynch[~np.isnan(asynchrony)] <= 500) / len(target_ponsets)
    lt_1000ms = np.sum(abs_asynch[~np.isnan(asynchrony)] <= 1000) / len(target_ponsets)
    lt_2000ms = np.sum(abs_asynch[~np.isnan(asynchrony)] <= 2000) / len(target_ponsets)
    return (
        mean_asynch,
        median_asynch,
        std_asynch,
        skewness,
        kurt,
        lt_25ms,
        lt_50ms,
        lt_100ms,
        lt_200ms,
        lt_300ms,
        lt_500ms,
        lt_1000ms,
        lt_2000ms,
        len(target_ponsets),
    )


def evaluate_performance_oltw_arzt(
    match_fn: str,
    polling_period: float,
    out_results_fn: str,
    remove_insertions: bool = False,
    use_perf_as_ref: bool = True,
    features: str = "pianoroll",
    note_noise: float = 0.1,
    tempo_scaling: float = 1.0,
    local_cost_fun: str = "Manhattan",
    window_size: int = 100,
    step_size: int = 10,
    iter: int = 1,
) -> None:

    if not os.path.exists(os.path.dirname(out_results_fn)):
        os.mkdir(os.path.dirname(out_results_fn))

    if not os.path.exists(out_results_fn):

        with open(out_results_fn, "w") as f:
            f.write(
                "\t".join(
                    [
                        "piece_name",
                        "mean_asynch",
                        "median_asynch",
                        "std_asynch",
                        "skewness",
                        "kurtosis",
                        "25ms",
                        "50ms",
                        "100ms",
                        "200ms",
                        "300ms",
                        "500ms",
                        "1000ms",
                        "n_onsets",
                    ]
                )
                + "\n"
            )

    use_mean_tempo = True
    bpm = 100
    # Load everything from match files to ensure correct score unfolding
    performance, alignment, score = pt.load_match(
        filename=match_fn,
        create_score=True,
        first_note_at_zero=True,
    )

    if remove_insertions:

        matched_note_ids = [
            al["performance_id"] for al in alignment if al["label"] == "match"
        ]

        matched_notes = [
            note for note in performance[0].notes if note["id"] in matched_note_ids
        ]

        performance[0].notes = matched_notes

    pnote_array = performance.note_array()

    snote_array = score.note_array()

    if use_mean_tempo:
        ptime_to_stime_map, stime_to_ptime_map = get_time_maps_from_alignment(
            ppart_or_note_array=pnote_array,
            spart_or_note_array=snote_array,
            alignment=alignment,
        )

        et, st = stime_to_ptime_map(
            [snote_array["onset_beat"].max(), snote_array["onset_beat"].min()]
        )

        bpm = (
            60
            * (snote_array["onset_beat"].max() - snote_array["onset_beat"].min())
            / (et - st)
        )

    if not use_perf_as_ref:
        score_perf = performance_from_part(
            part=score[0],
            bpm=bpm,
        )
    else:

        score_perf = PerformedPart(
            notes=copy.deepcopy(performance[0].notes),
            controls=copy.deepcopy(performance[0].controls),
            programs=copy.deepcopy(performance[0].programs),
            ppq=performance[0].ppq,
            mpq=performance[0].mpq,
        )

        matched_note_ids = [
            al["performance_id"] for al in alignment if al["label"] == "match"
        ]

        matched_notes = [
            note for note in score_perf.notes if note["id"] in matched_note_ids
        ]

        score_perf.notes = matched_notes
        note_ons = np.array([n["note_on"] for n in score_perf.notes])
        note_offs = np.array([n["note_off"] for n in score_perf.notes])
        durs = note_offs - note_ons

        if note_noise > 0:
            note_ons += RNG.uniform(
                -note_noise * 0.5, note_noise * 0.5, size=len(note_ons)
            )
            note_ons -= note_ons.min()

            durs += RNG.uniform(-note_noise * 0.5, note_noise * 0.5, size=len(note_ons))
            np.clip(
                durs,
                a_min=0.1,
                a_max=None,
                out=durs,
            )
        iois = np.diff(note_ons)

        # tempo_scaling = RNG.uniform(0.9, 1.1)

        rescaled_note_ons = np.r_[0, np.cumsum(iois * tempo_scaling)]

        rescaled_durs = durs * tempo_scaling

        for non, ndur, note in zip(rescaled_note_ons, rescaled_durs, score_perf.notes):
            note["note_on"] = max(non, 0)
            note["note_off"] = max(non + ndur, note["note_on"] + 0.1)

    queue = RECVQueue()

    if features == "pianoroll":
        features = [
            PianoRollProcessor(piano_range=True),
        ]
    elif features == "pitchclass":
        features = [PitchClassPianoRollProcessor()]

    midi_stream = MockFramedMidiStream(
        file_path=performance,
        queue=queue,
        polling_period=polling_period,
        features=features,
    )

    ref_queue = RECVQueue()
    reference_features_stream = MockFramedMidiStream(
        file_path=score_perf,
        queue=ref_queue,
        polling_period=polling_period,
        features=features,
    )

    reference_features_stream.start()
    reference_features_stream.join()
    reference_features = np.vstack(list(ref_queue.queue)).astype(np.float32)

    midi_stream.start()
    midi_stream.join()
    # get all outputs of the queue at once
    observations = np.vstack(list(queue.queue)).astype(np.float32)

    score_follower = OnlineTimeWarpingArzt(
        reference_features=reference_features,
        local_cost_fun=local_cost_fun,
        window_size=window_size,
        step_size=step_size,
        start_window_size=60,
        frame_rate=1,
    )
    predicted_positions = np.array(
        [score_follower(obs) for obs in observations],
        dtype=int,
    )
    score_beats = np.arange(
        np.ceil(snote_array["onset_beat"].min()),
        np.floor(snote_array["onset_beat"].max()) + 1,
    )

    perf_beats = stime_to_ptime_map(score_beats)

    if use_perf_as_ref:
        perf_beats = np.r_[
            perf_beats.min(), np.cumsum(np.diff(perf_beats) * tempo_scaling)
        ]

    tracked_beats = transfer_positions(
        wp=score_follower.warping_path,
        ref_anns=perf_beats,
        frame_rate=1 / polling_period,
    )
    eval_results = evaluate_alignment(
        target_ponsets=perf_beats,
        tracked_ponsets=tracked_beats,
    )

    with open(out_results_fn, "a") as f:

        str_output = [os.path.basename(match_fn)] + [
            f"{res:.3f}" for res in eval_results
        ]
        f.write("\t".join(str_output) + "\n")

    save_score_following_result(
        score_follower,
        Path(os.path.dirname(out_results_fn)),
        score_beats,
        tracked_beats,
        1 / polling_period,
        name=iter,
    )


def save_score_following_result(
    model, save_dir, score_beats, tracked_beats, frame_rate, name=None, is_hmm=False
):
    run_name = name or "results"
    save_path = save_dir / f"wp_{run_name}.tsv"
    save_nparray_to_csv(model.warping_path.T, save_path.as_posix())

    plt.figure(figsize=(15, 15))
    plt.title(
        f"[{save_dir.name}] \n Matchmaker alignment path with ground-truth labels",
        fontsize=25,
    )
    plt.xlabel("Performance (sec)", fontsize=15)
    plt.ylabel("Score (sec)", fontsize=15)

    # plot online DTW path
    x = model.state_space[model.warping_path[0]]
    y = model.warping_path[1] / frame_rate
    for ref, target in zip(x, y):
        plt.plot(target, ref, ".", color="purple", alpha=0.5, markersize=3)

    # plot ground-truth labels
    for ref, target in zip(score_beats, tracked_beats):
        plt.plot(target, ref, "x", color="r", alpha=1, markersize=6)
    plt.savefig(save_dir / f"{run_name}.png")


def evaluate_performance_oltw_dixon(
    match_fn: str,
    polling_period: float,
    out_results_fn: str,
    remove_insertions: bool = False,
    use_perf_as_ref: bool = True,
    features: str = "pianoroll",
    note_noise: float = 0.1,
    tempo_scaling: float = 1.0,
    window_size: int = 5,
    local_cost_fun: str = "cityblock",
    iter: int = 1,
) -> None:

    if not os.path.exists(os.path.dirname(out_results_fn)):
        os.mkdir(os.path.dirname(out_results_fn))

    if not os.path.exists(out_results_fn):

        with open(out_results_fn, "w") as f:
            f.write(
                "\t".join(
                    [
                        "piece_name",
                        "mean_asynch",
                        "median_asynch",
                        "std_asynch",
                        "skewness",
                        "kurtosis",
                        "25ms",
                        "50ms",
                        "100ms",
                        "200ms",
                        "300ms",
                        "500ms",
                        "1000ms",
                        "n_onsets",
                    ]
                )
                + "\n"
            )

    use_mean_tempo = True
    bpm = 100
    # Load everything from match files to ensure correct score unfolding
    performance, alignment, score = pt.load_match(
        filename=match_fn,
        create_score=True,
        first_note_at_zero=True,
    )

    if remove_insertions:

        matched_note_ids = [
            al["performance_id"] for al in alignment if al["label"] == "match"
        ]

        matched_notes = [
            note for note in performance[0].notes if note["id"] in matched_note_ids
        ]

        performance[0].notes = matched_notes

    pnote_array = performance.note_array()

    snote_array = score.note_array()

    if use_mean_tempo:
        ptime_to_stime_map, stime_to_ptime_map = get_time_maps_from_alignment(
            ppart_or_note_array=pnote_array,
            spart_or_note_array=snote_array,
            alignment=alignment,
        )

        et, st = stime_to_ptime_map(
            [snote_array["onset_beat"].max(), snote_array["onset_beat"].min()]
        )

        bpm = (
            60
            * (snote_array["onset_beat"].max() - snote_array["onset_beat"].min())
            / (et - st)
        )

    if not use_perf_as_ref:
        score_perf = performance_from_part(
            part=score[0],
            bpm=bpm,
        )
    else:

        score_perf = PerformedPart(
            notes=copy.deepcopy(performance[0].notes),
            controls=copy.deepcopy(performance[0].controls),
            programs=copy.deepcopy(performance[0].programs),
            ppq=performance[0].ppq,
            mpq=performance[0].mpq,
        )

        matched_note_ids = [
            al["performance_id"] for al in alignment if al["label"] == "match"
        ]

        matched_notes = [
            note for note in score_perf.notes if note["id"] in matched_note_ids
        ]

        score_perf.notes = matched_notes
        note_ons = np.array([n["note_on"] for n in score_perf.notes])
        note_offs = np.array([n["note_off"] for n in score_perf.notes])
        durs = note_offs - note_ons

        if note_noise > 0:
            note_ons += RNG.uniform(
                -note_noise * 0.5,
                note_noise * 0.5,
                size=len(note_ons),
            )
            note_ons -= note_ons.min()

            durs += RNG.uniform(
                -note_noise * 0.5,
                note_noise * 0.5,
                size=len(note_ons),
            )
            np.clip(
                durs,
                a_min=0.1,
                a_max=None,
                out=durs,
            )
        iois = np.diff(note_ons)

        rescaled_note_ons = np.r_[0, np.cumsum(iois * tempo_scaling)]

        rescaled_durs = durs * tempo_scaling

        for non, ndur, note in zip(rescaled_note_ons, rescaled_durs, score_perf.notes):
            note["note_on"] = max(non, 0)
            note["note_off"] = max(non + ndur, note["note_on"] + 0.1)

    if features == "pianoroll":
        features = [
            PianoRollProcessor(piano_range=True),
        ]
    elif features == "pitchclass":
        features = [PitchClassPianoRollProcessor()]

    ref_queue = RECVQueue()
    reference_features_stream = MockFramedMidiStream(
        file_path=score_perf,
        queue=ref_queue,
        polling_period=polling_period,
        features=features,
    )

    reference_features_stream.start()
    reference_features_stream.join()
    reference_features = np.vstack(list(ref_queue.queue)).astype(np.float32)

    score_follower = OnlineTimeWarpingDixon(
        reference_features=reference_features,
        window_size=window_size,
        local_cost_fun=local_cost_fun,
        frame_rate=1,
    )

    midi_stream = MockFramedMidiStream(
        file_path=performance,
        queue=score_follower.queue,
        polling_period=polling_period,
        features=features,
    )

    midi_stream.start()
    score_follower.run()
    midi_stream.join()

    score_beats = np.arange(
        np.ceil(snote_array["onset_beat"].min()),
        np.floor(snote_array["onset_beat"].max()) + 1,
    )

    perf_beats = stime_to_ptime_map(score_beats)

    if use_perf_as_ref:
        perf_beats = np.r_[
            perf_beats.min(), np.cumsum(np.diff(perf_beats) * tempo_scaling)
        ]
    tracked_beats = transfer_positions(
        wp=score_follower.warping_path,
        ref_anns=perf_beats,
        frame_rate=1 / polling_period,
    )
    eval_results = evaluate_alignment(
        target_ponsets=perf_beats,
        tracked_ponsets=tracked_beats,
    )

    with open(out_results_fn, "a") as f:

        str_output = [os.path.basename(match_fn)] + [
            f"{res:.3f}" for res in eval_results
        ]
        f.write("\t".join(str_output) + "\n")


def evaluate_performance_hmm(
    match_fn: str,
    polling_period: float,
    out_results_fn: str,
    remove_insertions: bool = False,
    features: str = "pianoroll",
    iter: int = 1,
) -> None:

    if not os.path.exists(os.path.dirname(out_results_fn)):
        os.mkdir(os.path.dirname(out_results_fn))

    if not os.path.exists(out_results_fn):

        with open(out_results_fn, "w") as f:
            f.write(
                "\t".join(
                    [
                        "piece_name",
                        "mean_asynch",
                        "median_asynch",
                        "std_asynch",
                        "skewness",
                        "kurtosis",
                        "25ms",
                        "50ms",
                        "100ms",
                        "200ms",
                        "300ms",
                        "500ms",
                        "1000ms",
                        "2000ms",
                        "n_onsets",
                    ]
                )
                + "\n"
            )

    bpm = 100
    # Load everything from match files to ensure correct score unfolding
    performance, alignment, score = pt.load_match(
        filename=match_fn,
        create_score=True,
        first_note_at_zero=True,
    )

    if remove_insertions:

        matched_note_ids = [
            al["performance_id"] for al in alignment if al["label"] == "match"
        ]

        matched_notes = [
            note for note in performance[0].notes if note["id"] in matched_note_ids
        ]

        performance[0].notes = matched_notes

    pnote_array = performance.note_array()

    snote_array = score.note_array()

    ptime_to_stime_map, stime_to_ptime_map = get_time_maps_from_alignment(
        ppart_or_note_array=pnote_array,
        spart_or_note_array=snote_array,
        alignment=alignment,
    )

    if features == "pianoroll":
        features = [
            PitchIOIProcessor(piano_range=True),
        ]
    elif features == "pitchclass":
        features = [PitchIOIProcessor()]

    unique_sonsets = np.unique(snote_array["onset_beat"])

    unique_sonset_idxs = [
        np.where(snote_array["onset_beat"] == ui)[0] for ui in unique_sonsets
    ]

    chord_pitches = [snote_array["pitch"][uix] for uix in unique_sonset_idxs]

    pitch_profiles = compute_discrete_pitch_profiles(
        chord_pitches=chord_pitches,
        piano_range=True,
        inserted_states=True,
    )

    ioi_matrix = compute_ioi_matrix(
        unique_onsets=unique_sonsets,
        inserted_states=True,
    )

    state_space = ioi_matrix[0]
    n_states = len(state_space)

    observation_model = BernoulliGaussianPitchIOIObservationModel(
        pitch_profiles=pitch_profiles,
        ioi_matrix=ioi_matrix,
        ioi_precision=1,
    )

    transition_matrix = (
        gumbel_transition_matrix(  # TODO another possibilities? (Gaussian/Nakamura)
            n_states=n_states,
            inserted_states=True,
        )
    )

    initial_probabilities = gumbel_init_dist(
        n_states=n_states,
    )

    tempo_model = KalmanTempoModel(  # TODO another possibilities? (LinearSMS, JADAM)
        init_score_onset=unique_sonsets.min(),
        init_beat_period=60 / bpm,
    )

    queue = RECVQueue()
    midi_stream = MockFramedMidiStream(
        file_path=performance,
        queue=queue,
        polling_period=polling_period,
        features=features,
    )
    score_follower = PitchIOIHMM(
        observation_model=observation_model,
        transition_matrix=transition_matrix,
        score_onsets=state_space,
        initial_probabilities=initial_probabilities,
        has_insertions=True,
        tempo_model=tempo_model,
    )

    before_time = time.time()
    midi_stream.start()
    midi_stream.join()
    print(f"Processed {midi_stream.counter} frames")
    print(
        f"Average MIDI feature processing time: {np.mean(midi_stream.elapsed_times)}, (median: {np.median(midi_stream.elapsed_times)}), (std: {np.std(midi_stream.elapsed_times)})"
    )

    observations = list(queue.queue)

    predicted_positions = np.array(
        [
            score_follower(obs[0], i)
            for i, obs in enumerate(observations)
            if obs[0] is not None
        ],
        dtype=int,
    )
    after_time = time.time()
    elapsed_time = after_time - before_time
    print(f"Elapsed time: {elapsed_time}")
    score_beats = np.arange(
        np.ceil(snote_array["onset_beat"].min()),
        np.floor(snote_array["onset_beat"].max()) + 1,
    )

    perf_beats = stime_to_ptime_map(score_beats)

    # import pdb
    # pdb.set_trace()
    tracked_beats = transfer_positions(
        wp=score_follower.warping_path,
        ref_anns=score_beats,
        frame_rate=1 / polling_period,
        is_hmm=True,
        state_space=score_follower.state_space,  # TODO check if this is correct
    )
    eval_results = evaluate_alignment(
        target_ponsets=perf_beats,
        tracked_ponsets=tracked_beats,
    )

    with open(out_results_fn, "a") as f:

        str_output = [os.path.basename(match_fn)] + [
            f"{res:.3f}" for res in eval_results
        ]
        f.write("\t".join(str_output) + "\n")

    save_score_following_result(
        score_follower,
        Path(os.path.dirname(out_results_fn)),
        score_beats,
        tracked_beats,
        1 / polling_period,
        name=iter,
    )


def vienna():
    polling_period = 0.01

    out_results_fn = "../results/vienna_100_10_score.txt"

    match_files = glob.glob(
        os.path.join(
            "/Users/carlos/Repos/vienna4x22_v100/match",
            "*.match",
            # "/Users/carlos/Repos/batik_plays_mozart_fork/match_adjusted", "*.match"
        )
    )

    match_files.sort()

    for i, match_fn in enumerate(match_files):
        print(f"Evaluating {os.path.basename(match_fn)} ...")

        if os.path.basename(match_fn) == "kv457_2_adj.match":
            continue
        evaluate_performance_oltw_arzt(
            match_fn=match_fn,
            polling_period=polling_period,
            out_results_fn=out_results_fn,
            use_perf_as_ref=False,
        )


def get_dataset(dataset: str) -> List[str]:

    if dataset == "asap":
        asap_data = pd.read_csv("./data/metadata-asap-test.csv")
        # asap_pieces = asap_data[asap_data["robust_note_alignment"] > 0][
        #     "midi_performance"
        # ].values.tolist()
        asap_pieces = asap_data["midi_performance"].values.tolist()
        asap_pieces_id = [p.split("/")[-1].split(".")[0] for p in asap_pieces]

        # asap_dir = "/Volumes/Rach3M02/asap-dataset/"

        match_files = glob.glob(
            # os.path.join("/Volumes/Rach3M02/asap-dataset/**", "*.match"),
            os.path.join(
                os.path.expanduser("~/workspace/asap-dataset"), "**", "*.match"
            ),
            recursive=True,
        )

        match_files = [
            mf
            for mf in match_files
            # if mf.replace(asap_dir, "").replace(".match", ".mid") in asap_pieces
            if mf.split("/")[-1].split(".")[0] in asap_pieces_id
        ]

    if dataset == "vienna":
        match_files = glob.glob(
            os.path.join(
                # "/Users/carlos/Repos/vienna4x22_v100/match",
                os.path.expanduser("~/workspace/vienna4x22/match"),
                "*.match",
            )
        )

    if dataset == "batik":
        match_files = glob.glob(
            os.path.join(
                # "/Users/carlos/Repos/batik_plays_mozart_fork/match_adjusted",
                os.path.expanduser("~/dataset/Batik_Audio/match_adjusted"),
                "*.match",
            )
        )

    match_files.sort()

    return match_files


if __name__ == "__main__":

    algorithm = "hmm"
    dataset = "asap"
    polling_period = 0.01
    window_size = 20
    step_size = 5
    distance = "cityblock"
    noise = 0.1
    use_score = False
    tempo_scaling = 1.0
    features = "pianoroll"

    if algorithm == "dixon":
        if use_score:
            run_name = (
                f"{dataset}_{algorithm}_{features}_ws{window_size}_{distance}_score"
            )
        else:
            run_name = f"{dataset}_{algorithm}_{features}_ws{window_size}_{distance}_noise_{noise:.2f}_tempo_{tempo_scaling:.2f}"
    elif algorithm == "arzt":
        if use_score:
            run_name = f"{dataset}_{algorithm}_{features}_ws{window_size}_ss{step_size}_{distance}_score"
        else:
            run_name = f"{dataset}_{algorithm}_{features}_ws{window_size}_ss{step_size}_{distance}_noise_{noise:.2f}_tempo_{tempo_scaling:.2f}"
    elif algorithm == "hmm":
        run_name = f"{dataset}_{algorithm}_{features}"

    run_name = f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}_{run_name}"
    if not os.path.exists(f"results/{run_name}"):
        os.mkdir(f"results/{run_name}")

    out_results_fn = os.path.join(f"results/{run_name}", f"results.tsv")

    match_files = get_dataset(dataset)

    for i, match_fn in enumerate(match_files, 1):
        if os.path.basename(match_fn) in ("kv457_2_adj.match", "kv280_2_adj.match"):
            continue

        if "Fugue/bwv_858/VuV01M.match" not in match_fn:
            continue

        print(f"Evaluating {match_fn} ...")

        if algorithm == "dixon":
            evaluate_performance_oltw_dixon(
                match_fn=match_fn,
                polling_period=polling_period,
                out_results_fn=out_results_fn,
                use_perf_as_ref=not use_score,
                features=features,
                note_noise=noise,
                tempo_scaling=tempo_scaling,
                local_cost_fun=distance,
                window_size=window_size,
            )
        elif algorithm == "arzt":
            evaluate_performance_oltw_arzt(
                match_fn=match_fn,
                polling_period=polling_period,
                out_results_fn=out_results_fn,
                use_perf_as_ref=not use_score,
                features=features,
                note_noise=noise,
                tempo_scaling=tempo_scaling,
                window_size=window_size,
                step_size=step_size,
                iter=i,
            )
        elif algorithm == "hmm":
            evaluate_performance_hmm(
                match_fn=match_fn,
                polling_period=polling_period,
                out_results_fn=out_results_fn,
                features=features,
                remove_insertions=True,
                iter=i,
            )
