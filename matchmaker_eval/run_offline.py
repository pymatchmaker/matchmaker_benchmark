"""Reproduce the manuscript offline reference through the common benchmark.

Non-causal chroma CENS + DLNCO / MrMsDTW, with the original fixed settings.
No online leaderboard publishing; the workflow archives complete reference results.
"""
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
for p in (BENCH, BENCH / "matchmaker_eval"):
    sys.path.insert(0, str(p))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
from partitura.io.exportmidi import get_ppq  # noqa: E402
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw  # noqa: E402
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO  # noqa: E402
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features  # noqa: E402

import run_submission as rs  # noqa: E402
from matchmaker_eval.submission import DECLARATION_KEY  # noqa: E402
from matchmaker import Matchmaker  # noqa: E402
from matchmaker.utils.misc import generate_score_audio  # noqa: E402

FEATURE_RATE = 50
STEP_WEIGHTS = np.array([1.5, 1.5, 2.0])
THRESHOLD_REC = 10 ** 6
FEATURE_SR = 22050


def safe_chroma_cens(y, sr, hop_length, chunk_sec=60.0, pad_sec=4.0):
    total_len = len(y)
    chunk_samples = int(chunk_sec * sr)
    pad_samples = int(pad_sec * sr)
    if total_len <= chunk_samples + pad_samples:
        return librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
    out = []
    start = 0
    while start < total_len:
        end = min(start + chunk_samples, total_len)
        left = max(0, start - pad_samples)
        right = min(total_len, end + pad_samples)
        c = librosa.feature.chroma_cens(y=y[left:right], sr=sr, hop_length=hop_length)
        fc = left + np.arange(c.shape[1]) * hop_length
        out.append(c[:, (fc >= start) & (fc < end)])
        start = end
    return np.concatenate(out, axis=1)

def features(audio, sr):
    hop = sr // FEATURE_RATE
    chroma = safe_chroma_cens(y=audio, sr=sr, hop_length=hop)
    onset = audio_to_pitch_onset_features(f_audio=audio, Fs=sr)
    dlnco = pitch_onset_features_to_DLNCO(
        f_peaks=onset, feature_rate=FEATURE_RATE,
        feature_sequence_length=chroma.shape[1], visualize=False)
    return chroma, dlnco


def offline_path(mm, perf_path):
    sr0 = mm.sample_rate
    score_audio = generate_score_audio(mm.score_part, mm.tempo, sr0).astype(np.float64)
    sr = FEATURE_SR
    if sr != sr0:
        score_audio = librosa.resample(score_audio, orig_sr=sr0, target_sr=sr)
    perf_audio, _ = librosa.load(str(perf_path), sr=sr)
    c1, o1 = features(score_audio, sr)
    c2, o2 = features(perf_audio.astype(np.float64), sr)
    wp = sync_via_mrmsdtw(f_chroma1=c1, f_onset1=o1, f_chroma2=c2, f_onset2=o2,
                          input_feature_rate=FEATURE_RATE, verbose=False,
                          step_weights=STEP_WEIGHTS, threshold_rec=THRESHOLD_REC)
    hop = sr // FEATURE_RATE  # the frame period is hop / sr exactly
    score_sec = wp[0].astype(float) * hop / sr
    perf_sec = wp[1].astype(float) * hop / sr
    tick = get_ppq(mm.score_part)
    beats = mm.score_part.beat_map(score_sec * tick * (mm.tempo / 60))
    order = np.argsort(perf_sec, kind="stable")
    return perf_sec[order], np.asarray(beats, dtype=float)[order]



def run_piece(method, piece, input_type, index, run_dir, save_plots, estimated_bpm=False):
    import json
    from matchmaker_eval.verify_tracking import score_meter_map, musical_beat_factors
    mm = Matchmaker(score_file=str(piece.score_path), performance_file=str(piece.performance_path(input_type)),
                    input_type=input_type, method="arzt", wait=False, unfold_score=True)
    t0 = time.perf_counter()
    perf_sec, beats = offline_path(mm, piece.performance_path(input_type))
    mm.alignment_duration = time.perf_counter()-t0
    mm.score_follower._alignment_path = list(zip(perf_sec.tolist(), beats.tolist()))
    mm._has_run = True
    perf, score_beat = rs.resolve_gt(piece.match_path, mm.score_part.note_array(), performance=None)
    gt = np.column_stack([perf, score_beat])
    metrics = rs.run_evaluation(mm, gt=gt, tolerances=rs.TOLERANCES_IN_BEATS, domain="score",
                                debug=True, save_dir=run_dir, run_name=str(index),
                                plot_dist_matrix=False, make_plot=False)
    meter = score_meter_map(mm.score_part)
    (run_dir/f"meter_{index}.json").write_text(json.dumps(meter,indent=2)+"\n")
    tracking = rs.check_tracking(np.column_stack([perf_sec,beats]), gt,
                                 segment_duration=rs.SEGMENT_DURATION,
                                 threshold=rs.TRACKING_THRESHOLD[input_type],
                                 beat_factors=musical_beat_factors(gt[:,1],meter))
    metrics.update(tracked=bool(tracking["tracked"]), max_deviation=float(tracking["max_deviation"]),
                   n_failed_segments=int(tracking["n_failed"]),
                   tracking_criterion={k:tracking[k] for k in ("error_unit","threshold","window_duration")},
                   used_estimated_bpm=False,
                   offline=True,
                   offline_settings=dict(feature_rate=FEATURE_RATE, render_tempo="score",
                       feature_sample_rate=FEATURE_SR, step_weights=STEP_WEIGHTS.tolist(), threshold_rec=THRESHOLD_REC))
    (run_dir/f"{index}.json").write_text(json.dumps(metrics,indent=2,default=float)+"\n")
    return metrics


def offline_metadata(method, input_type):
    return {"name":"Offline (DLNCO + MrMsDTW)", "authors":["synctoolbox"], "input_type":input_type,
            "description":"Non-causal MrMsDTW of performance and rendered score audio; chroma CENS + DLNCO.",
            rs.ESTIMATED_BPM_KEY:False, "url":"https://github.com/meinardmueller/synctoolbox",
            DECLARATION_KEY:True, "kind":"reference"}


def main():
    import hashlib
    original_parts=rs.scoring_parts
    rs.scoring_parts=lambda: {**original_parts(), "offline_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    rs.run_piece=run_piece
    rs.builtin_metadata=offline_metadata
    if "--method" not in sys.argv:
        sys.argv += ["--method","offline","--input-type","audio"]
    return rs.main()

if __name__ == "__main__":
    sys.exit(main())
