from pathlib import Path

import yaml
from pydantic_settings import BaseSettings

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
    experiment_config = load_config(EXP_CONFIG_PATH.as_posix())
    configs = []
    for distance_func in experiment_config["distance_func_exp"]:
        config = initialize_config(
            distance_func=distance_func,
        )
        configs.append(config)
    return configs


def save_config(config, save_dir):
    config_path = save_dir / "config.yaml"
    with open(config_path, "w") as f:
        config_dict = {
            k: v for k, v in config.__dict__.items() if not k.startswith("__")
        }
        yaml.dump(config_dict, f)
