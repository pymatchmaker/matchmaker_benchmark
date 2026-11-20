"""Stream audio at playback speed through a follower and record how far behind it reports.

Each position's delay is the wall time it was reported minus its performance
time, so it includes the frames the follower waits for as well as processing.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "matchmaker_eval"):
    sys.path.insert(0, str(p))

import matchmaker_eval.run_submission as rs  # noqa: E402
from matchmaker import Matchmaker  # noqa: E402


def delay_stats(reported_at, perf_times):
    delay_ms = (np.asarray(reported_at) - np.asarray(perf_times)) * 1000
    return {
        "delay_median_ms": float(np.median(delay_ms)),
        "delay_p95_ms": float(np.percentile(delay_ms, 95)),
        "delay_max_ms": float(delay_ms.max()),
        "delay_end_ms": float(delay_ms[-1]),
    }


def paced_matchmaker(reported_at, followers):
    def build(**kwargs):
        mm = Matchmaker(**{**kwargs, "wait": True})
        run = mm.run

        def timed_run(**run_kwargs):
            start = time.perf_counter()
            for beat in run(**run_kwargs):
                reported_at.append(time.perf_counter() - start)
                yield beat

        mm.run = timed_run
        followers.append(mm.score_follower)
        return mm

    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    methods = yaml.safe_load((ROOT / "data" / "builtin_methods.yaml").read_text())["audio"]
    parser.add_argument("--method", choices=sorted(methods), required=True)
    parser.add_argument("--fold", default="valid")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score_piece = rs.run_piece

    def run_piece(method, piece, input_type, index, run_dir, save_plots, estimated_bpm=False):
        reported_at, followers = [], []
        original = rs.Matchmaker
        rs.Matchmaker = paced_matchmaker(reported_at, followers)
        try:
            result = score_piece(method, piece, input_type, index, run_dir, save_plots, estimated_bpm)
        finally:
            rs.Matchmaker = original

        result.update(delay_stats(reported_at, followers[0].alignment_path[0]), paced_input=True)
        (run_dir / f"{index}.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
        return result

    rs.run_piece = run_piece
    scoring_parts = rs.scoring_parts
    rs.scoring_parts = lambda: {
        **scoring_parts(), "paced_input": True,
        "delay_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    rs.evaluate_submission(builtin=args.method, input_type="audio", fold=args.fold,
                           shard=args.shard, num_shards=args.num_shards, run_dir=args.output,
                           piece_timeout=None)


if __name__ == "__main__":
    main()
