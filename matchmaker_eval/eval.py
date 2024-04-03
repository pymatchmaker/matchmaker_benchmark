from pathlib import Path

import librosa
import mido
import numpy as np
import pandas as pd
import scipy
from matchmaker.dp import OnlineTimeWarpingArzt, OnlineTimeWarpingDixon
from matchmaker.features.audio import compute_features_from_audio
from matchmaker.io.audio import MockAudioStream
from numpy.typing import NDArray
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from utils import MatchmakerEvalConfig
from libfmp.c3 import compute_strict_alignment_path_mask

TOLERANCES = [100, 300, 500, 1000]
ALGORITHMS = {
    "oltw_dixon": OnlineTimeWarpingDixon,
    "oltw_arzt": OnlineTimeWarpingArzt,
}
METRICS = ["mean", "median", "std", "skewness", "kurtosis"] + [
    f"{t}ms" for t in TOLERANCES
]


def _get_DLNCO_features_from_audio(audio, feature_sequence_length, Fs, feature_rate):
    f_pitch_onset = audio_to_pitch_onset_features(f_audio=audio, Fs=Fs)
    f_DLNCO = pitch_onset_features_to_DLNCO(
        f_peaks=f_pitch_onset,
        feature_rate=feature_rate,
        feature_sequence_length=feature_sequence_length,
        visualize=False,
    )

    return f_DLNCO


def run_offline_alignment(
    score_audio_path: Path, ref_audio_path: Path, config, same_feature=False
):
    # read audio
    audio_1, _ = librosa.load(score_audio_path.as_posix(), sr=config.sample_rate)
    audio_2, _ = librosa.load(ref_audio_path.as_posix(), sr=config.sample_rate)

    if same_feature:
        # extract chroma stft features
        f_chroma_librosa_1 = librosa.feature.chroma_stft(
            y=audio_1,
            sr=config.sample_rate,
            hop_length=config.hop_length,
        )
        f_chroma_librosa_2 = librosa.feature.chroma_stft(
            y=audio_2,
            sr=config.sample_rate,
            hop_length=config.hop_length,
        )
        f_DLNCO_1, f_DLNCO_2 = None, None
    else:
        # extract chroma cens features
        f_chroma_librosa_1 = librosa.feature.chroma_cens(
            y=audio_1,
            sr=config.sample_rate,
            hop_length=config.hop_length,
        )
        f_chroma_librosa_2 = librosa.feature.chroma_cens(
            y=audio_2,
            sr=config.sample_rate,
            hop_length=config.hop_length,
        )
        # generate DLNCO features
        f_DLNCO_1 = _get_DLNCO_features_from_audio(
            audio_1, f_chroma_librosa_1.shape[1], config.sample_rate, config.frame_rate
        )

        f_DLNCO_2 = _get_DLNCO_features_from_audio(
            audio_2, f_chroma_librosa_2.shape[1], config.sample_rate, config.frame_rate
        )

    wp = sync_via_mrmsdtw(
        f_chroma1=f_chroma_librosa_1,
        f_onset1=f_DLNCO_1,
        f_chroma2=f_chroma_librosa_2,
        f_onset2=f_DLNCO_2,
        input_feature_rate=config.frame_rate,
        verbose=False,
    )
    wp = compute_strict_alignment_path_mask(wp.T).T
    return wp


def transfer_positions(wp, ref_anns):
    """
    Transfer the positions of the reference annotations to the target annotations using the warping path.

    Parameters
    ----------
    wp : np.array with shape (2, T)
        array of warping path.
    ref_ann : List[float]
        reference annotations.
    """
    x, y = wp[0], wp[1]
    predicted_targets = [y[np.where(x >= r)[0][0]] for r in ref_anns]
    return predicted_targets


def run_evaluation(wp, ref_ann, target_ann, frame_rate):
    ref_annots = np.rint(
        pd.read_csv(filepath_or_buffer=ref_ann, delimiter="\t", header=None)[0]
        * frame_rate
    )
    target_annots = np.rint(
        pd.read_csv(filepath_or_buffer=target_ann, delimiter="\t", header=None)[0]
        * frame_rate
    )

    target_annots_predicted = transfer_positions(wp, ref_annots)
    errors_in_delay = (
        (target_annots - target_annots_predicted) / frame_rate * 1000
    )  # in milliseconds

    # # Find the values that are at the 5th and 95th percentiles
    # lower_threshold = np.percentile(errors_in_delay, 5)
    # upper_threshold = np.percentile(errors_in_delay, 95)
    # errors_in_delay = errors_in_delay[
    #     (errors_in_delay > lower_threshold) & (errors_in_delay < upper_threshold)
    # ]
    absolute_errors_in_delay = np.abs(errors_in_delay)

    results = {
        "mean": float(f"{np.mean(absolute_errors_in_delay):.4f}"),
        "median": float(f"{np.median(absolute_errors_in_delay):.4f}"),
        "std": float(f"{np.std(absolute_errors_in_delay):.4f}"),
        "skewness": float(f"{scipy.stats.skew(errors_in_delay):.4f}"),
        "kurtosis": float(f"{scipy.stats.kurtosis(errors_in_delay):.4f}"),
    }
    for tau in TOLERANCES:
        results[f"{tau}ms"] = float(f"{np.mean(absolute_errors_in_delay <= tau):.4f}")

    return results


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


def run_score_following(
    score_audio: str, target_audio: str, config: MatchmakerEvalConfig
) -> NDArray[np.float32]:
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
        score_audio,
        features=config.features,
        sample_rate=config.sample_rate,
        hop_length=config.hop_length,
    )
    algorithm = ALGORITHMS[config.algorithm]
    matchmaker = algorithm(
        reference_features=reference_features,
        local_cost_fun=config.distance_func,
        window_size=config.window_size,
        max_run_count=config.max_run_count,
        frame_per_seg=config.frame_per_seg,
        frame_rate=config.frame_rate,
    )

    audio_stream = MockAudioStream(
        sample_rate=config.sample_rate,
        hop_length=config.hop_length,
        queue=matchmaker.queue,
        features=feature_processors,
        file_path=target_audio,
        chunk_size=config.chunk_size,
    )

    # Run score following
    audio_stream.start()
    matchmaker.run()

    print(f"=====================oltl run ended=====================")
    audio_stream.stop()

    return matchmaker, matchmaker.warping_path
