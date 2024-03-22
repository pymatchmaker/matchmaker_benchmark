from pathlib import Path

import librosa
import mido
import numpy as np
import pandas as pd
import scipy
from matchmaker.dp import OnlineTimeWarpingArzt, OnlineTimeWarpingDixon
from matchmaker.features.audio import compute_features_from_audio
from matchmaker.io.audio import MockAudioStream
from midi2audio import FluidSynth
from numpy.typing import NDArray
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features

SOUND_FONT_PATH = "~/.fluidsynth/MuseScore_General.sf2"
TOLERANCES = [50, 100, 200, 300, 500]
FEATURES = ["chroma"]

DEFAULT_LOCAL_COST: str = "euclidean"
MAX_RUN_COUNT: int = 30
SAMPLE_RATE = 16000  # temporary
HOP_LENGTH = 640
N_FFT = 2 * HOP_LENGTH
N_MELS = 66
NORM = np.inf
CHUNK_SIZE = 1 * HOP_LENGTH
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH
FRAME_PER_SEG: int = int(CHUNK_SIZE / HOP_LENGTH)
WINDOW_SIZE: int = 5 * int(FRAME_RATE)  # 5 seconds


def _get_DLNCO_features_from_audio(
    audio,
    feature_sequence_length,
    Fs=SAMPLE_RATE,
    feature_rate=FRAME_RATE,
    verbose=False,
):
    f_pitch_onset = audio_to_pitch_onset_features(f_audio=audio, Fs=Fs)
    f_DLNCO = pitch_onset_features_to_DLNCO(
        f_peaks=f_pitch_onset,
        feature_rate=feature_rate,
        feature_sequence_length=feature_sequence_length,
        visualize=verbose,
    )

    return f_DLNCO


def run_offline_alignment(
    score_audio_path: Path, ref_audio_path: Path, same_feature=False
):
    # read audio
    audio_1, _ = librosa.load(score_audio_path.as_posix(), sr=SAMPLE_RATE)
    audio_2, _ = librosa.load(ref_audio_path.as_posix(), sr=SAMPLE_RATE)

    if same_feature:
        # extract chroma stft features
        f_chroma_librosa_1 = librosa.feature.chroma_stft(
            y=audio_1,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
        )
        f_chroma_librosa_2 = librosa.feature.chroma_stft(
            y=audio_2,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
        )
        f_DLNCO_1, f_DLNCO_2 = None, None
    else:
        # extract chroma cens features
        f_chroma_librosa_1 = librosa.feature.chroma_cens(
            y=audio_1,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
        )
        f_chroma_librosa_2 = librosa.feature.chroma_cens(
            y=audio_2,
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
        )
        # generate DLNCO features
        f_DLNCO_1 = _get_DLNCO_features_from_audio(
            audio=audio_1,
            feature_sequence_length=f_chroma_librosa_1.shape[1],
        )

        f_DLNCO_2 = _get_DLNCO_features_from_audio(
            audio=audio_2,
            feature_sequence_length=f_chroma_librosa_2.shape[1],
        )

    wp_chroma_dlnco = sync_via_mrmsdtw(
        f_chroma1=f_chroma_librosa_1,
        f_onset1=f_DLNCO_1,
        f_chroma2=f_chroma_librosa_2,
        f_onset2=f_DLNCO_2,
        input_feature_rate=FRAME_RATE,
        verbose=False,
    )
    return wp_chroma_dlnco


def transfer_positions(wp, ref_ann, frame_rate):
    """
    Transfer the positions of the reference annotations to the target annotations using the warping path.

    Parameters
    ----------
    wp : np.array with shape (2, T)
        array of warping path.
    ref_ann : List[float]
        reference annotations.
    frame_rate : float
        frame rate of the annotations to convert frame index to time.
    """
    x, y = wp[0] / frame_rate, wp[1] / frame_rate
    f = scipy.interpolate.interp1d(x, y, kind="linear")
    target_ann = f(ref_ann)
    return target_ann


def run_evaluation(wp, ref_ann, target_ann):
    ref_annots = pd.read_csv(filepath_or_buffer=ref_ann, delimiter="\t", header=None)[0]
    target_annots = pd.read_csv(
        filepath_or_buffer=target_ann, delimiter="\t", header=None
    )[0]

    target_annots_predicted = transfer_positions(wp, ref_annots, FRAME_RATE)

    errors_in_delay = (
        target_annots - target_annots_predicted
    ) * 1000  # in milliseconds
    absolute_errors_in_delay = np.abs(errors_in_delay)

    results = {
        "mean": np.mean(absolute_errors_in_delay),
        "median": np.median(absolute_errors_in_delay),
        "std": np.std(absolute_errors_in_delay),
        "skewness": scipy.stats.skew(errors_in_delay),
        "kurtosis": scipy.stats.kurtosis(errors_in_delay),
    }
    for tau in TOLERANCES:
        results[f"{tau}ms"] = np.mean(absolute_errors_in_delay <= tau)

    return results


def convert_score_to_audio(midi_path: Path, save_path: Path) -> Path:
    # Convert MIDI to audio
    fs = FluidSynth(SOUND_FONT_PATH, sample_rate=SAMPLE_RATE)
    fs.midi_to_audio(midi_path, save_path.as_posix())

    print(
        f"Score Audio path: {save_path}, duration: {librosa.get_duration(path=save_path.as_posix())}"
    )
    return save_path


def regenerate_tempo_adjusted_midi(midi_path: Path, target_duration: float) -> Path:
    mid = mido.MidiFile(midi_path.as_posix())
    ratio = target_duration / mid.length
    print(f"Tempo ratio (target audio / midi length): {ratio}")

    has_set_tempo = False
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                has_set_tempo = True
                print(f"Original tempo: {mido.tempo2bpm(msg.tempo)}, {msg.tempo}")
                new_tempo = mido.bpm2tempo(mido.tempo2bpm(msg.tempo) / ratio)
                print(f"New tempo: {mido.tempo2bpm(new_tempo)}, {new_tempo}")
                msg.tempo = int(new_tempo)
    if not has_set_tempo:
        # If the track does not have a 'set_tempo' message, add one with the default BPM
        default_bpm = 120
        print(f"[Default] Original tempo: {default_bpm}, 500000")
        new_tempo = mido.bpm2tempo(default_bpm / ratio)
        print(f"New tempo: {new_tempo}")
        for track in mid.tracks:
            track.insert(0, mido.MetaMessage("set_tempo", tempo=int(new_tempo)))

    new_midi_path = midi_path.with_stem("midi_score_adjusted")
    mid.save(new_midi_path)
    return new_midi_path


def run_score_following(score_audio: str, target_audio: str) -> NDArray[np.float32]:
    """
    Run score following on the given audio and the target audio.

    Parameters
    ----------
    score_audio : str
        path to the score audio.
    target_audio : str
        path to the target audio.

    Returns
    -------
    warping_path: np.ndarray [shape=(2, T)]
        Resulting warping path with pairs of indices of the reference and target audio.
    """
    feature_processors, reference_features = compute_features_from_audio(
        score_audio, features=FEATURES, sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH
    )
    # oltw = OnlineTimeWarpingArzt(
    #     reference_features=reference_features,
    # )
    oltw = OnlineTimeWarpingDixon(
        reference_features=reference_features,
        local_cost_fun=DEFAULT_LOCAL_COST,
        window_size=WINDOW_SIZE,
        max_run_count=MAX_RUN_COUNT,
        frame_per_seg=int(CHUNK_SIZE / HOP_LENGTH),
    )

    audio_stream = MockAudioStream(
        sample_rate=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        queue=oltw.queue,
        features=feature_processors,
        file_path=target_audio,
        chunk_size=CHUNK_SIZE,
    )

    # Run score following
    audio_stream.start()
    oltw.run()

    print(f"=====================oltl run ended=====================")
    audio_stream.stop()

    return oltw.warping_path
