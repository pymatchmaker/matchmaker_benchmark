from pathlib import Path
import tempfile
from typing import Any, Callable, List, Tuple, Union
import librosa
import numpy as np
import partitura as pt
from partitura.performance import PerformanceLike
from partitura.score import Part, merge_parts
import pandas as pd
from joblib import Parallel, delayed
from partitura.utils.generic import interp1d
from tqdm import tqdm
import multiprocessing
from partitura.score import Score, Part, ScoreLike
from partitura.performance import Performance, PerformedPart, PerformanceLike

from matchmaker.features.audio import (
    HOP_LENGTH,
    SAMPLE_RATE,
    ChromagramProcessor,
    MelSpectrogramProcessor,
    MFCCProcessor,
    LogSpectralEnergyProcessor,
    FRAME_RATE,
    CQTProcessor,
)

# from matchmaker.io.audio import AudioStream
# from matchmaker.utils.misc import save_mixed_audio
from matchmaker.utils.misc import (
    # RECVQueue,
    adjust_tempo_for_performance_audio,
    save_mixed_audio,
)

# import matplotlib.pyplot as plt
# from partitura.io.exportmidi import get_ppq

# from matchmaker.prob.hmm import GaussianAudioPitchHMM
# from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArzt
# from matchmaker.utils.eval import get_evaluation_results, TOLERANCES, transfer_positions

import warnings
import os

# Force ignore warnings before any other imports
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# warnings.filterwarnings("ignore", module=".*librosa.*")
# warnings.filterwarnings("ignore", module=".*partitura.*")


WORKING_DIR = Path(__file__).parent.parent
DATASET_DIR = {
    "asap": Path("../datasets/asap-dataset-matchmaker"),
    "batik": Path("../datasets/Batik_Audio"),
    "vienna": Path("../datasets/vienna4x22"),
}
METADATA_PATH = {
    "asap": WORKING_DIR / "data/metadata-asap.csv",
    "batik": WORKING_DIR / "data/metadata-batik.csv",
    "vienna": WORKING_DIR / "data/metadata-vienna.csv",
}

METADATA = {
    "asap": pd.read_csv(METADATA_PATH["asap"]),
    "batik": pd.read_csv(METADATA_PATH["batik"]),
    "vienna": pd.read_csv(METADATA_PATH["vienna"]),
}
# OUTPUT_DIR = WORKING_DIR / "output"


def process_feature_offline(
    target_audio: np.ndarray,
    f_time: float,
    last_chunk: Union[np.ndarray, None],
    hop_length: int,
    processor: Callable[[np.ndarray], Any],
) -> Tuple[Any, float, np.ndarray]:
    if last_chunk is None:  # add zero padding at the first block
        target_audio = np.concatenate(
            (np.zeros(hop_length, dtype=np.float32), target_audio)
        )
    else:
        # add last chunk at the beginning of the block
        target_audio = np.concatenate((last_chunk, target_audio))

    features = processor(target_audio)
    last_chunk = target_audio[-hop_length:]

    return (features, f_time, last_chunk)


def build_score_annotations(
    score_part,
    tempo,
    level="beat",
    musical_beat: bool = False,
):
    score_annots = []
    if level == "beat":  # TODO: add bar-level, note-level
        if musical_beat:
            score_part.use_musical_beat()  # for asap dataset
        note_array = np.unique(score_part.note_array()["onset_beat"])
        start_beat = np.ceil(note_array.min())
        end_beat = np.floor(note_array.max())
        beats = np.arange(start_beat, end_beat + 1)

        beat_timestamp = [
            score_part.inv_beat_map(beat)
            / score_part.quarter_duration_map(score_part.inv_beat_map(beat))
            * (60 / tempo)
            for beat in beats
        ]

        score_annots = np.array(beat_timestamp)
    return score_annots


# def build_score_annotations(score_part, tempo, level="beat"):
#     score_annots = []
#     if level == "beat":  # TODO: add bar-level, note-level
#         onsets_in_beats = np.unique(score_part.note_array()["onset_beat"])
#         start_beat = np.ceil(onsets_in_beats.min())
#         end_beat = np.floor(onsets_in_beats.max())
#         beats = np.arange(start_beat, end_beat + 1)

#         beat_timestamp = [
#             score_part.inv_beat_map(beat) / get_ppq(score_part) * (60 / tempo)
#             for beat in beats
#         ]

#         score_annots = np.array(beat_timestamp)
#     return score_annots

# def build_score_annotations(
#     score_part: Part,
#     tempo: float,
#     level="beat",
#     musical_beat: bool = False,
# ):

#     if musical_beat:
#         score_part.use_musical_beat()  # for asap

#     snote_array = score_part.note_array()

#     unique_onsets = np.unique(snote_array["onset_beat"])

#     iois = np.diff(unique_onsets)

#     if callable(tempo) or isinstance(tempo, np.ndarray):
#         if callable(tempo):
#             # bpm parameter is a callable that returns a bpm value
#             # for each score onset
#             bp = 60 / tempo(unique_onsets)

#         elif isinstance(tempo, np.ndarray):
#             if tempo.ndim != 2:
#                 raise ValueError("`bpm` should be a 2D array")

#             bpm_fun = interp1d(
#                 x=tempo[:, 0],
#                 y=tempo[:, 1],
#                 kind="previous",
#                 bounds_error=False,
#                 fill_value=(tempo[0, 1], tempo[-1, 1]),
#             )
#             bp = 60 / bpm_fun(unique_onsets)

#         p_onsets = np.r_[0, np.cumsum(iois * bp[:-1])]

#     else:
#         # convert bpm to beat period
#         bp = 60 / float(tempo)
#         p_onsets = np.r_[0, np.cumsum(iois * bp)]

#     if level == "beat":
#         ponset_fun = interp1d(
#             x=unique_onsets,
#             y=p_onsets,
#             kind="linear",
#         )
#         start_beat = np.ceil(unique_onsets.min())
#         end_beat = np.floor(unique_onsets.max())
#         beats = np.arange(start_beat, end_beat + 1)

#         score_annots = ponset_fun(beats)

#     elif level == "onset":
#         score_annots = p_onsets

#     return score_annots


def process_audio_offline(
    file_path: Union[str, Path, ScoreLike, PerformanceLike],
    sample_rate: int,
    hop_length: int,
    processor: Callable[[np.ndarray], Any],
    tempo: float = 120,
) -> List[Any]:
    """Process audio file in offline mode.

    This method simulates real-time processing by reading chunks from
    an audio file at regular intervals. The processing speed can be
    controlled using the `wait` parameter.

    Note
    ----
    The audio file is processed in chunks of size `hop_length`,
    and features are extracted for each chunk.
    """
    # Calculate the duration (in seconds), hop interval (in seconds), etc.
    duration = int(librosa.get_duration(path=file_path))

    # Load the audio and pad it to handle edge cases
    audio_y, _ = librosa.load(file_path, sr=sample_rate)
    padded_audio = np.concatenate(
        (
            audio_y,
            # e.g., add 10% of the original duration in zeros
            np.zeros(int(duration * 0.1 * sample_rate), dtype=np.float32),
        )
    )

    # Trim the padded audio so its length is a multiple of hop_length
    excess = len(padded_audio) % hop_length
    trimmed_audio = padded_audio if excess == 0 else padded_audio[:-excess]

    # Process the audio in frames of hop_length
    frames = []
    last_chunk = None
    for i in range(0, len(trimmed_audio), hop_length):
        # Slice the current frame
        target_audio = trimmed_audio[i : i + hop_length]

        # Compute the corresponding frame time in seconds (audio time)
        # This sets the frame "timestamp" based on how many samples have passed
        f_time = i / sample_rate

        # Process the current frame
        output, f_time, last_chunk = process_feature_offline(
            target_audio,
            f_time,
            last_chunk,
            hop_length,
            processor,
        )
        frames.append((output, f_time))

    return np.array(frames, dtype=object)


def extract_features(fn: Path) -> None:
    processor_chroma = ChromagramProcessor(
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )

    processor_mel = MelSpectrogramProcessor(
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )

    processor_mfcc = MFCCProcessor(
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )

    processor_lse = LogSpectralEnergyProcessor(
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    processor_cqt = CQTProcessor(
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )

    processors = [
        ("chroma", processor_chroma),
        ("mel", processor_mel),
        ("lse", processor_lse),
        ("mfcc", processor_mfcc),
        ("cqt", processor_cqt),
    ]

    for proc_name, processor in processors:

        output_path = fn.with_name(f"{fn.stem}_{proc_name}.npz")

        if not output_path.exists():

            output_path.parent.mkdir(parents=True, exist_ok=True)

            frames = process_audio_offline(
                file_path=fn,
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                processor=processor,
            )

            np.savez_compressed(output_path, frames=frames)

            check = np.random.rand() < 0.25
            if check:
                # Load the saved frames and compare with the original frames
                loaded_data = np.load(output_path, allow_pickle=True)
                loaded_frames = loaded_data["frames"]

                assert len(frames) == len(loaded_frames)
                for ff, lf in zip(frames, loaded_frames):
                    feat, f_time = ff
                    lfeat, lf_time = lf

                    assert np.all(feat == lfeat)
                    assert f_time == lf_time


def process_file(dataset, af):
    fn = DATASET_DIR[dataset] / Path(af)
    extract_features(fn)


# def build_score_annotations(
#     score_part: Part,
#     tempo: float,
#     level="beat",
#     musical_beat: bool = False,
# ):
#     score_annots = []
#     if level == "beat":  # TODO: add bar-level, note-level
#         if musical_beat:
#             score_part.use_musical_beat()  # for asap
#         unique_onsets = np.unique(score_part.note_array()["onset_beat"])
#         start_beat = np.ceil(unique_onsets.min())
#         end_beat = np.floor(unique_onsets.max())
#         beats = np.arange(start_beat, end_beat + 1)

#         beat_timestamp = [
#             score_part.inv_beat_map(beat)
#             / score_part.quarter_duration_map(score_part.inv_beat_map(beat))
#             * (60 / tempo)
#             for beat in beats
#         ]

#         score_annots = np.array(beat_timestamp)
#     return score_annots


def process_score(af, rf, dataset):

    try:
        ref_score = pt.load_musicxml(
            filename=rf,
            force_note_ids="keep",
            ignore_invisible_objects=False,
        )

        tempo = adjust_tempo_for_performance_audio(
            ref_score,
            af,
        )

        if len(ref_score) == 1:
            score_part = ref_score[0]
        else:
            score_part = merge_parts(ref_score.parts)

        sfn = af.with_name(f"{af.stem}_scoresynth.wav")

        if not sfn.exists():
            sfn.parent.mkdir(parents=True, exist_ok=True)
            pt.save_wav_fluidsynth(
                input_data=ref_score,
                out=sfn,
                bpm=tempo,
            )

        extract_features(fn=sfn)

        safn = af.with_name(f"{af.stem}_score_annotations.txt")
        if not safn.exists():
            score_annotations = build_score_annotations(
                score_part=score_part,
                tempo=tempo,
                level="beat",
                musical_beat=dataset == "asap",
            )
            # save_mixed_audio(
            #     sfn,
            #     score_annotations,
            #     save_path=af.with_name(f"{af.stem}_beats_mixed.wav"),
            # )

            np.savetxt(safn, score_annotations, fmt="%.6f")
            check = np.random.rand() < 0.10
            if check:
                loaded_annotations = np.loadtxt(safn, dtype=float)
                assert np.allclose(
                    score_annotations, loaded_annotations
                ), "Loaded annotations do not match the original"
    except Exception as e:

        return (af, rf, e)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser("Preprocess Data")

    parser.add_argument(
        "--synth-scores",
        action="store_true",
        help="Synthesize scores if this flag is used",
        default=False,
    )

    args = parser.parse_args()

    preprocessing_task = "synth_scores" if args.synth_scores else "extract_features"

    if preprocessing_task == "extract_features":
        tasks = []

        for dataset in ["vienna", "asap", "batik"]:
            audio_files = METADATA[dataset]["audio_performance"]
            for af in audio_files:
                tasks.append((dataset, af))

        num_cores = multiprocessing.cpu_count()

        # Wrap the iterable with tqdm for a progress bar
        results = Parallel(n_jobs=num_cores)(
            delayed(process_file)(dataset, af)
            for dataset, af in tqdm(tasks, desc="Processing audio files")
        )

    else:

        tasks = []

        for dataset in ["vienna", "asap", "batik"]:
            for ix, row in METADATA[dataset].iterrows():
                af = DATASET_DIR[dataset] / Path(row["audio_performance"])
                rf = DATASET_DIR[dataset] / Path(row["xml_score"])

                tasks.append((af, rf, dataset))
        num_cores = multiprocessing.cpu_count()

        results = Parallel(n_jobs=num_cores)(
            delayed(process_score)(af, rf, dataset)
            for af, rf, dataset in tqdm(tasks, desc="Processing scores files")
        )

        failed = []
        # Optionally, filter out the failed ones if process_file returns the path on error
        for r in results:
            if r is not None:
                failed.append(r)

        if failed:
            print("\nFailed files:")
            for f in failed:
                print(f)
