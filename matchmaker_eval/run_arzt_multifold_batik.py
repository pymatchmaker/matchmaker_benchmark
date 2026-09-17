#!/usr/bin/env python3
"""Run OnlineTimeWarpingArztMultiFold baseline on Batik 36 folded performances and compute metrics."""

from __future__ import annotations

import contextlib
import copy
import csv
import json
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Setup paths
BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH / "matchmaker_eval"))
sys.path.insert(0, "/Users/jiyun/workspace/matchmaker-laurenceyoon")

from run_graph_kf_experiment import (
    causal_errors,
    dump,
    error_stats,
    extract_features,
    load_tracking,
    paths,
    prepare_worker,
)
from matchmaker import Matchmaker
from matchmaker.matchmaker import DEFAULT_KWARGS
from matchmaker.io.queue import RECVQueue
from matchmaker.io.stream import STREAM_END


class Stream:
    def __init__(self):
        self.queue = RECVQueue()


def sparc(movement, fs=50.0, padlevel=4, fc=10.0, amp_th=0.05):
    """Canonical SPARC implementation (Balasubramanian et al., 2012 / 2015)."""
    movement = np.asarray(movement, float)
    if len(movement) < 10:
        return np.nan
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = abs(np.fft.fft(movement, nfft))
    max_Mf = max(Mf)
    if max_Mf <= 0 or not np.isfinite(max_Mf):
        return np.nan
    Mf = Mf / max_Mf

    fc_inx = ((f <= fc) * 1).nonzero()[0]
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
    if len(inx) < 2:
        return -1.0
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    df = f_sel[-1] - f_sel[0]
    if df == 0:
        return -1.0
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / df, 2) + pow(np.diff(Mf_sel), 2)))
    return float(new_sal)


def load_velocity_from_wp(wp, fs=50.0):
    t = wp[:, 0]
    x = wp[:, 1]
    good = np.isfinite(t) & np.isfinite(x)
    t, x = t[good], x[good]
    keep = np.r_[True, np.diff(t) > 0]
    t, x = t[keep], x[keep]
    if len(t) < 5 or t[-1] - t[0] < 2:
        return None
    tq = np.arange(t[0], t[-1], 1 / fs)
    pos = np.interp(tq, t, x)
    vel = np.gradient(pos, 1 / fs)
    if len(vel) > 2 * int(fs):
        vel = vel[int(fs):-int(fs)]
    return vel


def run_case(task):
    case, root_dir, data_root = task
    root = Path(root_dir)
    piece_dir = root / "pieces" / f"{case['index']:03d}_{case['title']}"
    dest = piece_dir / "folded_arzt_multifold"
    dest.mkdir(parents=True, exist_ok=True)

    result_path = dest / "result.json"
    if result_path.exists():
        return case["title"], "reused"

    check = load_tracking(root / "protocol" / "verify_tracking.py")
    score, audio, match = paths("batik", case["row"], data_root)

    kw = copy.deepcopy(DEFAULT_KWARGS["audio"]["arzt_multi_fold"])
    stream = Stream()
    mm = Matchmaker(
        score_file=score,
        performance_file=audio,
        method="arzt_multi_fold",
        stream=stream,
        unfold_score=False,
        tempo=None,
        kwargs=kw,
    )

    features, feature_seconds, audio_seconds = extract_features(
        audio, mm.processor, mm.sample_rate, mm.hop_length
    )
    for obs in features:
        mm.stream.queue.put(obs)
    mm.stream.queue.put(STREAM_END)

    inference_start = time.perf_counter()
    for _ in mm.score_follower.run(verbose=False):
        pass
    inference_seconds = time.perf_counter() - inference_start

    wp = np.asarray(mm.score_follower.alignment_path).T
    np.savetxt(
        dest / "wp.tsv",
        wp,
        delimiter="\t",
        header="perf_sec\tscore_beat",
        comments="",
    )

    # Read GT
    gt_file = root / "gt" / f"{case['index']:03d}_{case['title']}.tsv"
    gt_data = np.genfromtxt(gt_file, delimiter="\t", names=True)
    target = np.column_stack([gt_data["perf_sec"], gt_data["notated_beat"]])

    tracking = check(wp, target)
    errors = causal_errors(wp, target)
    np.save(dest / "errors.npy", errors)

    stats = error_stats(errors)

    # SPARC
    v = load_velocity_from_wp(wp)
    sparc_val = float(sparc(v)) if v is not None else np.nan

    result = dict(
        index=case["index"],
        title=case["title"],
        split=case["split"],
        has_repeat=case["has_repeat"],
        method="folded_arzt_multifold",
        mode="folded_baseline",
        reference="folded",
        coordinate="notated",
        tracked=bool(tracking["tracked"]),
        max_deviation=tracking["max_deviation"],
        n_failed=tracking["n_failed"],
        beat=stats,
        sparc=sparc_val,
        audio_seconds=audio_seconds,
        inference_seconds=inference_seconds,
        feature_seconds=feature_seconds,
        replay_rtf=(inference_seconds + feature_seconds) / audio_seconds,
        inference_rtf=inference_seconds / audio_seconds,
    )
    dump(result_path, result)
    return (
        case["title"],
        f"tracked={tracking['tracked']}, meanAE={stats['mean']:.3f}b, rtf={result['replay_rtf']:.3f}",
    )


def main():
    root_dir = Path(
        "/Users/jiyun/workspace/matchmaker-benchmark/output/flexible_batik36_p05_ophmm_sj1e5_20260905"
    )
    data_root = "/Users/jiyun/data"
    plan = json.loads((root_dir / "experiment_plan.json").read_text())
    cases = plan["cases"]

    # Filter to repeat pieces or all 36
    repeat_cases = [c for c in cases if c["has_repeat"]]
    print(f"Total cases: {len(cases)}, Repeat cases: {len(repeat_cases)}")

    tasks = [(c, str(root_dir), data_root) for c in cases]
    print(f"Running Arzt Multi-Fold on {len(tasks)} cases with 4 workers...")

    results_summary = []
    with ProcessPoolExecutor(max_workers=4, initializer=prepare_worker) as pool:
        futures = {pool.submit(run_case, t): t[0] for t in tasks}
        for n, f in enumerate(as_completed(futures), 1):
            c = futures[f]
            try:
                title, status = f.result()
                print(f"[{n}/{len(tasks)}] {title}: {status}", flush=True)
            except Exception as e:
                print(f"[{n}/{len(tasks)}] {c['title']} FAILED with error: {e}", flush=True)

    print("\n================ BENCHMARK SUMMARY ================")
    all_results = []
    for c in cases:
        dest = (
            root_dir
            / "pieces"
            / f"{c['index']:03d}_{c['title']}"
            / "folded_arzt_multifold"
        )
        if (dest / "result.json").exists():
            res = json.loads((dest / "result.json").read_text())
            all_results.append((c, res))

    repeats = [r for c, r in all_results if c["has_repeat"]]
    norep = [r for c, r in all_results if not c["has_repeat"]]

    tr_rep = sum(r["tracked"] for r in repeats)
    tr_norep = sum(r["tracked"] for r in norep)
    tr_tot = sum(r["tracked"] for _, r in all_results)

    tracked_repeats = [r for r in repeats if r["tracked"]]
    mean_ae_rep = (
        np.mean([r["beat"]["mean"] for r in tracked_repeats])
        if tracked_repeats
        else np.nan
    )
    le05_rep = (
        np.mean([r["beat"]["le_05"] for r in tracked_repeats])
        if tracked_repeats
        else np.nan
    )
    le1_rep = (
        np.mean([r["beat"]["le_1"] for r in tracked_repeats])
        if tracked_repeats
        else np.nan
    )
    sparc_rep = (
        np.mean([r["sparc"] for r in tracked_repeats if np.isfinite(r["sparc"])])
        if tracked_repeats
        else np.nan
    )
    rtf_rep = np.median([r["replay_rtf"] for r in repeats]) if repeats else np.nan

    print(f"\n--- Repeat pieces (30): ---")
    print(f"Tracking Rate (TR): {tr_rep}/{len(repeats)} ({tr_rep/len(repeats)*100:.1f}%)")
    print(f"MeanAE (tracked):   {mean_ae_rep:.3f} beats")
    print(f"<=0.5b (tracked):   {le05_rep*100:.1f}%")
    print(f"<=1.0b (tracked):   {le1_rep*100:.1f}%")
    print(f"SPARC (tracked):    {sparc_rep:.2f}")
    print(f"Median Replay RTF:  {rtf_rep:.3f}")

    print(f"\n--- Non-repeat controls (6): ---")
    print(f"Tracking Rate (TR): {tr_norep}/{len(norep)} ({tr_norep/len(norep)*100:.1f}%)")

    print(f"\n--- Total pieces (36): ---")
    print(f"Tracking Rate (TR): {tr_tot}/{len(all_results)} ({tr_tot/len(all_results)*100:.1f}%)")


if __name__ == "__main__":
    main()
