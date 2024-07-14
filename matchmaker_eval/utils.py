import csv
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import pandas as pd
import partitura
import scipy
import yaml
from midi2audio import FluidSynth
from numpy.typing import NDArray
from pydantic_settings import BaseSettings

SOUND_FONT_PATH = "~/.fluidsynth/MuseScore_General.sf2"
WORKING_DIR = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = WORKING_DIR / "config/default.yaml"
EXP_CONFIG_PATH = WORKING_DIR / "config/experiment.yaml"


class MatchmakerEvalConfig(BaseSettings):
    sample_rate: int
    frame_rate: int
    chunk_size: int
    window_size: int
    features: list[str]
    distance_func: str
    dataset: str
    algorithm: str
    hop_length: int
    n_fft: int
    frame_per_seg: int
    max_run_count: int

    # attributes for experiment (for logging purpose)
    attr_exp: list[str] = [
        "sample_rate",
        "frame_rate",
        "window_size",
        "features",
        "distance_func",
        "dataset",
        "algorithm",
    ]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def initialize_config(**kwargs) -> dict:
    default_config = load_config(DEFAULT_CONFIG_PATH.as_posix())

    sample_rate = kwargs.get("sample_rate", default_config["sample_rate"])
    frame_rate = kwargs.get("frame_rate", default_config["frame_rate"])
    chunk_size = kwargs.get("chunk_size", default_config["chunk_size"])
    window_size = kwargs.get("window_size", default_config["window_size"])
    features = kwargs.get("features", default_config["features"])
    distance_func = kwargs.get("distance_func", default_config["distance_func"])
    max_run_count = kwargs.get("max_run_count", default_config["max_run_count"])
    dataset = kwargs.get("dataset", default_config["dataset"])
    algorithm = kwargs.get("algorithm", default_config["algorithm"])

    hop_length = sample_rate // frame_rate
    n_fft = 2 * hop_length
    frame_per_seg = chunk_size

    # initialize config
    conf = MatchmakerEvalConfig(
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        chunk_size=chunk_size,
        window_size=window_size,
        features=features,
        distance_func=distance_func,
        dataset=dataset,
        algorithm=algorithm,
        hop_length=hop_length,
        n_fft=n_fft,
        frame_per_seg=frame_per_seg,
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
        for sample_rate in config["sample_rate_exp"]:
            for frame_rate in config["frame_rate_exp"]:
                for window_size in config["window_size_exp"]:
                    for distance_func in config["distance_func_exp"]:
                        exp_config = initialize_config(
                            algorithm=config["algorithm"],
                            sample_rate=sample_rate,
                            frame_rate=frame_rate,
                            window_size=window_size,
                            features=config["feature_exp"],
                            dataset=dataset,
                            distance_func=distance_func,
                        )
                        configs.append(exp_config)
    return configs


def save_config(config, save_dir):
    config_path = save_dir / "config.yaml"
    with open(config_path, "w") as f:
        config_dict = {
            k: v for k, v in config.__dict__.items() if not k.startswith("__")
        }
        yaml.dump(config_dict, f)


def convert_score_to_audio(score_path: str, sample_rate: int) -> Path:
    file_extension = score_path.suffix
    # Convert score to MIDI if it is in XML format
    if file_extension.lower() in {".xml", ".musicxml"}:
        score = partitura.load_score(str(score_path))
        tmp_score_path = score_path.parent / "tmp_midi_score.mid"
        print(f"Saving score as midi: {tmp_score_path}")
        partitura.save_score_midi(score, tmp_score_path.as_posix())
        score_path = tmp_score_path
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
    model, save_dir, score_ann, target_ann, frame_rate, name=None
):
    run_name = name or "results"
    save_path = save_dir / f"wp_{run_name}.tsv"
    save_nparray_to_csv(model.warping_path.T, save_path.as_posix())

    dist = scipy.spatial.distance.cdist(
        model.reference_features,
        model.input_features[: model.warping_path[1][-1]],
        metric=model.local_cost_fun,
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
            target_paths[n], ref_paths[n], ".", color="purple", alpha=0.5, markersize=3
        )

    # plot ground-truth labels
    ref_annots = pd.read_csv(filepath_or_buffer=score_ann, delimiter="\t", header=None)[
        0
    ]
    target_annots = pd.read_csv(
        filepath_or_buffer=target_ann, delimiter="\t", header=None
    )[0]
    for i, (ref, target) in enumerate(zip(ref_annots, target_annots)):
        # if i % 5 != 0:
        #     continue
        plt.plot(
            target * frame_rate, ref * frame_rate, "x", color="r", alpha=1, markersize=3
        )
    plt.savefig(save_dir / f"{run_name}.png")
