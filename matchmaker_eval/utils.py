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
from matchmaker.utils.tempo_models import KalmanTempoModel
from midi2audio import FluidSynth
from numpy.typing import NDArray
from pydantic_settings import BaseSettings

SOUND_FONT_PATH = "~/soundfonts/sf2/MuseScore_General.sf2"
WORKING_DIR = Path(__file__).parent.parent
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


def save_results_to_csv(results: dict, save_path: str):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(results.keys())
        writer.writerows(zip(*results.values()))


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
