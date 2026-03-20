import csv
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import partitura
import partitura as pt
import scipy
import yaml
from matchmaker.prob.hmm import (
    BernoulliGaussianPitchIOIObservationModel,
    PitchIOIHMM,
    compute_discrete_pitch_profiles,
    compute_ioi_matrix,
    gumbel_init_dist,
    gumbel_transition_matrix,
)
from matchmaker.utils.eval import get_evaluation_results, transfer_positions
from matchmaker.utils.tempo_models import KalmanTempoModel
from midi2audio import FluidSynth
from numpy.typing import NDArray
from partitura.utils.fluidsynth import DEFAULT_SOUNDFONT as SOUND_FONT_PATH
from pydantic_settings import BaseSettings

WORKING_DIR = Path(__file__).parent.parent
TOLERANCES_IN_MS = [50, 100, 300, 500, 1000, 2000]
TOLERANCES_IN_BEATS = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
METRICS_MS = ["mean", "median", "std", "skewness", "kurtosis"] + [
    f"{t}ms" for t in TOLERANCES_IN_MS
]
METRICS_BEAT = ["mean_b", "median_b", "std_b", "skewness_b", "kurtosis_b"] + [
    f"{t}b" for t in TOLERANCES_IN_BEATS
]
METRICS_TRACKING = ["tracked", "max_deviation", "n_failed_segments"]
METRICS_ALL = METRICS_MS + METRICS_BEAT + METRICS_TRACKING
DEFAULT_CONFIG_PATH = WORKING_DIR / "config/default.yaml"
EXP_CONFIG_PATH = WORKING_DIR / "config/experiment.yaml"


class MatchmakerEvalConfig(BaseSettings):
    sample_rate: int
    frame_rate: int
    window_size: int
    feature_type: str
    distance_func: str
    method: str
    hop_length: int
    n_fft: int
    max_run_count: int
    dataset: Optional[str] = None  # for experiment
    adjust_tempo: bool = False  # whether to adjust tempo based on performance audio

    # attributes for inference
    attr_infer: list[str] = [
        "sample_rate",
        "frame_rate",
        "window_size",
        "feature_type",
        "distance_func",
        "method",
        "hop_length",
        "n_fft",
    ]

    # attributes for experiment (for logging purpose)
    attr_exp: list[str] = [
        "sample_rate",
        "frame_rate",
        "window_size",
        "feature_type",
        "distance_func",
        "dataset",
        "method",
    ]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def initialize_config(**kwargs) -> dict:
    default_config = load_config(DEFAULT_CONFIG_PATH.as_posix())

    sample_rate = kwargs.get("sample_rate", default_config["sample_rate"])
    frame_rate = kwargs.get("frame_rate", default_config["frame_rate"])
    window_size = kwargs.get("window_size", default_config["window_size"])
    feature_type = kwargs.get("feature_type", default_config["feature_type"])
    distance_func = kwargs.get("distance_func", default_config["distance_func"])
    max_run_count = kwargs.get("max_run_count", default_config["max_run_count"])
    method = kwargs.get("method", default_config["method"])
    dataset = kwargs.get("dataset")

    hop_length = sample_rate // frame_rate
    n_fft = 2 * hop_length

    # initialize config
    conf = MatchmakerEvalConfig(
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        window_size=window_size,
        feature_type=feature_type,
        distance_func=distance_func,
        dataset=dataset,
        method=method,
        hop_length=hop_length,
        n_fft=n_fft,
        max_run_count=max_run_count,
    )
    return conf


def get_list_of_exp_config():
    config = load_config(DEFAULT_CONFIG_PATH.as_posix())
    experiment_config = load_config(EXP_CONFIG_PATH.as_posix())
    for key in experiment_config.keys():
        config[key] = experiment_config[key]

    configs = []
    for dataset in config["dataset_exp"]:
        for method in config["method_exp"]:
            for sample_rate in config["sample_rate_exp"]:
                for frame_rate in config["frame_rate_exp"]:
                    for window_size in config["window_size_exp"]:
                        for distance_func in config["distance_func_exp"]:
                            for feature_type in config["feature_type_exp"]:
                                exp_config = initialize_config(
                                    method=method,
                                    sample_rate=sample_rate,
                                    frame_rate=frame_rate,
                                    window_size=window_size,
                                    feature_type=feature_type,
                                    dataset=dataset,
                                    distance_func=distance_func,
                                )
                        configs.append(exp_config)
    return configs


def save_config(config, save_dir):
    config_path = save_dir / "config.yaml"
    with open(config_path, "w") as f:
        config_dict = {
            k: v
            for k, v in config.__dict__.items()
            if not k.startswith("__") and not k.startswith("attr_")
        }
        yaml.dump(config_dict, f)


def convert_score_to_audio(score_path: str, sample_rate: int) -> Path:
    file_extension = score_path.suffix
    # Convert score to MIDI if it is in XML format
    if file_extension.lower() in {".xml", ".musicxml"}:
        score = partitura.load_score(str(score_path))
        tmp_score_midi_path = score_path.parent / "tmp_midi_score.mid"
        print(f"Saving score as midi: {tmp_score_midi_path}")
        partitura.save_score_midi(score, tmp_score_midi_path.as_posix())
        score_path = tmp_score_midi_path
    elif file_extension.lower() not in {".mid", ".midi"}:
        raise ValueError("Invalid score file format")

    # Convert MIDI to audio
    score_audio_path = score_path.with_suffix(".wav")
    fs = FluidSynth(SOUND_FONT_PATH, sample_rate=sample_rate)
    fs.midi_to_audio(score_path, score_audio_path)

    # print(
    #     f"Score Audio path: {score_audio_path}, duration (sec): {librosa.get_duration(path=score_audio_path)}"
    # )
    return score_audio_path


def compute_event_pooled_summary(
    results: dict,
    run_dir: Path,
    config,
    tracked_only: bool = True,
) -> dict:
    """
    Compute event-wise pooled metrics.

    Accuracy metrics (mean, median, tolerances): pooled across all events.
    RTF/latency metrics: piece-wise averaged.

    Parameters
    ----------
    tracked_only : bool
        If True, only include tracked pieces. If False, include all pieces.
    """
    n_total = len(results["Index"])
    tracked_flags = results.get("tracked", [False] * n_total)

    # Collect events from selected pieces
    all_gt_perf = []
    all_pred_perf = []
    all_gt_score_beats = []
    all_pred_score_beats = []
    selected_indices = []

    for idx, is_tracked in zip(results["Index"], tracked_flags):
        if tracked_only and not is_tracked:
            continue
        selected_indices.append(idx)

        wp_file = run_dir / f"wp_{idx}.tsv"
        gt_file = run_dir / f"gt_{idx}.tsv"
        if not wp_file.exists() or not gt_file.exists():
            continue

        wp = np.loadtxt(wp_file, delimiter="\t")
        gt = np.loadtxt(gt_file, delimiter="\t")

        gt_score = gt[:, 0]  # score positions (beats or seconds)
        gt_perf = gt[:, 1]  # perf times (seconds)

        # Score → perf prediction (ms metrics)
        pred_perf = transfer_positions(wp.T, gt_score, config.frame_rate, mode="beat")
        valid = np.isfinite(pred_perf) & np.isfinite(gt_perf)
        all_gt_perf.append(gt_perf[valid])
        all_pred_perf.append(pred_perf[valid])

        # Perf → score prediction (beat metrics)
        valid_gt_perf = np.isfinite(gt_perf)
        pred_score = transfer_positions(
            wp.T, gt_perf[valid_gt_perf], config.frame_rate, reverse=True, mode="beat"
        )
        valid_b = np.isfinite(pred_score)
        all_gt_score_beats.append(gt_score[valid_gt_perf][valid_b])
        all_pred_score_beats.append(pred_score[valid_b])

    summary = {}
    n_tracked = sum(1 for t in tracked_flags if t)
    n_selected = len(selected_indices)
    summary["tracking_rate"] = round(n_tracked / n_total, 4) if n_total > 0 else 0.0

    if all_gt_perf:
        pooled_gt = np.concatenate(all_gt_perf)
        pooled_pred = np.concatenate(all_pred_perf)
        total_count = len(pooled_gt)
        ms_results = get_evaluation_results(
            pooled_gt,
            pooled_pred,
            total_count,
            tolerances=TOLERANCES_IN_MS,
            in_seconds=True,
        )
        for k, v in ms_results.items():
            summary[k] = v

    if all_gt_score_beats:
        pooled_gt_b = np.concatenate(all_gt_score_beats)
        pooled_pred_b = np.concatenate(all_pred_score_beats)
        total_count_b = len(pooled_gt_b)
        beat_results = get_evaluation_results(
            pooled_gt_b,
            pooled_pred_b,
            total_count_b,
            tolerances=TOLERANCES_IN_BEATS,
            in_seconds=False,
        )
        for k, v in beat_results.items():
            # Keys like "0.1b" already have 'b' suffix; others need "_b"
            if k.endswith("b"):
                summary[k] = v
            else:
                summary[f"{k}_b"] = v

    # RTF and latency: piece-wise average of selected pieces
    for key in ["rtf", "f_avg_latency", "i_avg_latency"]:
        if key in results:
            if tracked_only:
                vals = [v for v, t in zip(results[key], tracked_flags) if t]
            else:
                vals = list(results[key])
            if vals:
                summary[key] = float(f"{np.nanmean(vals):.4f}")

    summary["piece_count"] = n_total
    summary["tracked_count"] = n_tracked
    summary["selected_count"] = n_selected
    return summary


def save_results_to_csv(results: dict, save_path: str):
    from itertools import zip_longest

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(results.keys())
        writer.writerows(zip_longest(*results.values(), fillvalue=""))


def save_nparray_to_csv(array: NDArray, save_path: str):
    with open(save_path, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerows(array)


def save_score_following_result(
    model, save_dir, score_annots, perf_ann_path: Path, frame_rate, name=None
):
    run_name = name or "results"
    save_path = save_dir / f"wp_{run_name}.tsv"
    save_nparray_to_csv(model.warping_path.T, save_path.as_posix())

    dist = scipy.spatial.distance.cdist(
        model.reference_features,
        model.input_features[: model.warping_path[1][-1]],
        metric=model.distance_func,
    )  # [d, wy]
    plt.figure(figsize=(15, 15))
    plt.imshow(dist, aspect="auto", origin="lower", interpolation="nearest")
    plt.title(
        f"[{save_dir.name}] \n Matchmaker alignment path with ground-truth labels",
        fontsize=25,
    )
    plt.xlabel("Performance Audio frame", fontsize=15)
    plt.ylabel("Score Audio frame", fontsize=15)

    # plot online DTW path
    ref_paths, target_paths = model.warping_path[0], model.warping_path[1]
    for n in range(len(ref_paths)):
        plt.plot(
            target_paths[n], ref_paths[n], ".", color="lime", alpha=0.5, markersize=3
        )

    # plot ground-truth labels
    perf_annots = pd.read_csv(
        filepath_or_buffer=perf_ann_path, delimiter="\t", header=None
    )[0]
    for i, (ref, target) in enumerate(zip(score_annots, perf_annots)):
        plt.plot(
            target * frame_rate, ref * frame_rate, "x", color="r", alpha=1, markersize=3
        )
    plt.savefig(save_dir / f"{run_name}.png")


# def plot_and_save_score_following_result(
#     wp,
#     ref_features,
#     input_features,
#     distance_func,
#     save_dir,
#     score_annots,
#     perf_ann_path: Path,
#     frame_rate,
#     name=None,
# ):
#     run_name = name or "results"
#     save_path = save_dir / f"wp_{run_name}.tsv"
#     save_nparray_to_csv(wp.T, save_path.as_posix())

#     dist = scipy.spatial.distance.cdist(
#         ref_features,
#         input_features[: wp[1][-1]],
#         metric=distance_func,
#     )  # [d, wy]
#     plt.figure(figsize=(15, 15))
#     plt.imshow(dist, aspect="auto", origin="lower", interpolation="nearest")
#     plt.title(
#         f"[{save_dir.name}] \n Matchmaker alignment path with ground-truth labels",
#         fontsize=25,
#     )
#     plt.xlabel("Performance Audio frame", fontsize=15)
#     plt.ylabel("Score Audio frame", fontsize=15)

#     # plot online DTW path
#     ref_paths, target_paths = wp[0], wp[1]
#     for n in range(len(ref_paths)):
#         plt.plot(
#             target_paths[n], ref_paths[n], ".", color="purple", alpha=0.5, markersize=3
#         )

#     # plot ground-truth labels
#     perf_annots = pd.read_csv(
#         filepath_or_buffer=perf_ann_path, delimiter="\t", header=None
#     )[0]
#     for i, (ref, target) in enumerate(zip(score_annots, perf_annots)):
#         plt.plot(
#             target * frame_rate, ref * frame_rate, "x", color="r", alpha=1, markersize=3
#         )
#     plt.savefig(save_dir / f"{run_name}.png")


def build_matchmaker_hmm(score_path):
    bpm = 100

    snote_array = pt.load_score_midi(score_path).note_array()
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

    matchmaker = PitchIOIHMM(
        observation_model=observation_model,
        transition_matrix=transition_matrix,
        score_onsets=state_space,
        initial_probabilities=initial_probabilities,
        has_insertions=True,
        tempo_model=tempo_model,
    )
    return matchmaker
