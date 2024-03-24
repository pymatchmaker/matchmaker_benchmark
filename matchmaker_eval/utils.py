from pydantic_settings import BaseSettings

# default configs
DEFAULT_LOCAL_COST: str = "euclidean"
MAX_RUN_COUNT: int = 30
SAMPLE_RATE = 16000  # temporary
HOP_LENGTH = 640
N_FFT = 2 * HOP_LENGTH
N_MELS = 66
CHUNK_SIZE = 1 * HOP_LENGTH
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH
FRAME_PER_SEG: int = int(CHUNK_SIZE / HOP_LENGTH)
WINDOW_SIZE: int = 5 * int(FRAME_RATE)  # 5 seconds
FEATURES = ["chroma"]
DATASET = "asap"
ALGORITHM = "oltw_dixon"

# experiment configs
SAMPLE_RATE_EXP = [16000, 22050]  #  2
FRAME_RATE_EXP = [25, 50, 100]  # 3
WINDOW_SIZE_EXP = [1, 3, 5]  # 3
FEATURES_EXP = ["chroma"]
SCIPY_DISTANCES = [
    "euclidean",
    "cosine",
    "dice",
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "jensenshannon",
    "minkowski",
    "sqeuclidean",
]


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
    attr_exp: list[str] = [
        "sample_rate",
        "frame_rate",
        "window_size",
        "features",
        "distance_func",
        "dataset",
        "algorithm",
    ]


def initialize_config(**kwargs) -> dict:
    sample_rate = kwargs.get("sample_rate", SAMPLE_RATE)
    frame_rate = kwargs.get("frame_rate", FRAME_RATE)
    chunk_size = kwargs.get("chunk_size", CHUNK_SIZE)
    window_size = kwargs.get("window_size", WINDOW_SIZE)
    features = kwargs.get("features", FEATURES)
    distance_func = kwargs.get("distance_func", DEFAULT_LOCAL_COST)
    dataset = kwargs.get("dataset", DATASET)
    algorithm = kwargs.get("algorithm", ALGORITHM)

    hop_length = sample_rate // frame_rate
    n_fft = 2 * hop_length
    frame_per_seg = int(chunk_size / hop_length)
    max_run_count = 30

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
    configs = []
    for distance_func in SCIPY_DISTANCES:
        config = initialize_config(
            distance_func=distance_func,
        )
        configs.append(config)
    return configs
