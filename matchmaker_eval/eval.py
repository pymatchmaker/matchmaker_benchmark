from pathlib import Path
from typing import Union

import librosa
import mido
import numpy as np
import pandas as pd
import partitura as pt
import scipy
from libfmp.c3 import compute_strict_alignment_path_mask
from matchmaker.dp import OnlineTimeWarpingArzt, OnlineTimeWarpingDixon
from matchmaker.features.audio import (
    compute_features_from_audio,
    ChromagramIOIProcessor,
)
from matchmaker.io.audio import AudioStream, MockAudioStream
from matchmaker.io.midi import MockFramedMidiStream
from matchmaker.prob.hmm import (
    BernoulliGaussianPitchIOIObservationModel,
    PitchIOIHMM,
    jiang_transition_matrix_from_sequence,
    gumbel_init_dist,
    compute_ioi_matrix,
)
from matchmaker.utils.misc import RECVQueue
from matchmaker.utils.tempo_models import KalmanTempoModel
from numpy.typing import NDArray
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from utils import (
    MatchmakerEvalConfig,
    convert_score_to_audio,
    create_frame_index_from_onset_sec,
)

TOLERANCES = [100, 300, 500, 1000]
ALGORITHMS = {
    "oltw_dixon": OnlineTimeWarpingDixon,
    "oltw_arzt": OnlineTimeWarpingArzt,
    "hmm": PitchIOIHMM,
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

    absolute_errors_in_delay = np.abs(errors_in_delay)
    filtered_abs_errors_in_delay = absolute_errors_in_delay[
        absolute_errors_in_delay <= TOLERANCES[-1]
    ]

    results = {
        "mean": float(f"{np.mean(filtered_abs_errors_in_delay):.4f}"),
        "median": float(f"{np.median(filtered_abs_errors_in_delay):.4f}"),
        "std": float(f"{np.std(filtered_abs_errors_in_delay):.4f}"),
        "skewness": float(f"{scipy.stats.skew(filtered_abs_errors_in_delay):.4f}"),
        "kurtosis": float(f"{scipy.stats.kurtosis(filtered_abs_errors_in_delay):.4f}"),
    }
    for tau in TOLERANCES:
        results[f"{tau}ms"] = float(f"{np.mean(absolute_errors_in_delay <= tau):.4f}")

    results["count"] = len(filtered_abs_errors_in_delay)
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
    score_path: Path, perf_path: Union[Path, str], config: MatchmakerEvalConfig
) -> NDArray[np.float32]:
    """
    Run score following on the score audio and the performance file.

    Parameters
    ----------
    score_path : Path
        path to the score file (.mid or .xml).
    perf_path : Path or str
        path to the performance file (.wav or .mid), or empty string for live performance mode.

    Returns
    -------
    warping_path: np.ndarray [shape=(2, T)]
        Resulting warping path with pairs of indices of the reference and target audio.
    """
    # Convert score to audio
    score_audio_path = convert_score_to_audio(score_path, config.sample_rate)

    algorithm = ALGORITHMS[config.algorithm]
    queue = RECVQueue()
    if config.algorithm == "hmm":
        frame_rate = config.frame_rate
        onsets_in_sec = np.unique(
            pt.load_performance_midi(score_path).note_array()["onset_sec"]
        )
        index_list = create_frame_index_from_onset_sec(
            onsets_in_sec, frame_rate
        )  # [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9, ...]
        transition_matrix, state_space = jiang_transition_matrix_from_sequence(
            index_list, frame_rate, 0.1
        )
        n_states = len(state_space)

        feature_processors = [
            ChromagramIOIProcessor(
                sample_rate=config.sample_rate, hop_length=config.hop_length
            )
        ]
        ioi_matrix = compute_ioi_matrix(index_list)

        score_chromagram = librosa.feature.chroma_cens(
            librosa.load(score_audio_path, sr=config.sample_rate)[0],
            sr=config.sample_rate,
            hop_length=config.hop_length,
        ).T[:n_states]
        observation_model = BernoulliGaussianPitchIOIObservationModel(
            pitch_profiles=score_chromagram,
            ioi_matrix=ioi_matrix,
            ioi_precision=1,
        )
        initial_probabilities = gumbel_init_dist(
            n_states=n_states,
        )
        tempo_model = KalmanTempoModel(
            init_score_onset=np.where(index_list == 1)[0][0],
            init_beat_period=60 / 120 * frame_rate,
        )
        queue = RECVQueue()
        matchmaker = PitchIOIHMM(
            observation_model=observation_model,
            transition_matrix=transition_matrix,
            score_onsets=index_list,
            initial_probabilities=initial_probabilities,
            has_insertions=True,
            tempo_model=tempo_model,
        )
    else:
        # Extract features from the score audio
        feature_processors, reference_features = compute_features_from_audio(
            str(score_audio_path),
            features=config.features,
            sample_rate=config.sample_rate,
            hop_length=config.hop_length,
        )
        score_audio_path.unlink()
        matchmaker = algorithm(
            reference_features=reference_features,
            local_cost_fun=config.distance_func,
            window_size=config.window_size,
            max_run_count=config.max_run_count,
            frame_per_seg=config.frame_per_seg,
            frame_rate=config.frame_rate,
        )

    perf_stream = None
    if perf_path == "":  # live performance mode
        perf_stream = AudioStream(
            sample_rate=config.sample_rate,
            hop_length=config.hop_length,
            queue=matchmaker.queue,
            features=feature_processors,
            chunk_size=config.chunk_size,
        )
    elif perf_path.suffix.lower() in {".mid", ".midi"}:
        match_path = perf_path.with_suffix(".match")
        performance, alignment, score = pt.load_match(
            filename=match_path, create_score=True, first_note_at_zero=True
        )
        perf_stream = MockFramedMidiStream(
            file_path=performance,
            queue=queue,
            features=feature_processors,
        )
    elif perf_path.suffix.lower() in {".wav"}:
        perf_stream = MockAudioStream(
            sample_rate=config.sample_rate,
            hop_length=config.hop_length,
            queue=queue,
            features=feature_processors,
            file_path=str(perf_path),
            chunk_size=config.chunk_size,
            include_ftime=True,
        )

    # Run score following
    perf_stream.start()

    if config.algorithm == "hmm":
        perf_stream.join()
        observations = list(queue.queue)

        predicted_positions = np.array(
            [
                matchmaker((obs[0], obs[1]), i)
                for i, obs in enumerate(observations)
                if obs[0] is not None
            ],
            dtype=int,
        )
    else:
        matchmaker.run()

    print(f"=====================oltl run ended=====================")
    perf_stream.stop()

    return matchmaker, matchmaker.warping_path
