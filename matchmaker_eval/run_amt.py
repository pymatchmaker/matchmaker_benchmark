"""Replay frozen online-AMT MIDI through symbolic followers, scored against audio GT.

The transcription model is not run, so timings cover the follower only. MIDI
event times are used as they are, without latency compensation.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "matchmaker_eval"):
    sys.path.insert(0, str(p))

import matchmaker_eval.run_submission as rs  # noqa: E402
from matchmaker import Matchmaker  # noqa: E402
from matchmaker_eval.provenance import file_sha256  # noqa: E402
from matchmaker_eval.run_diagnostics import download  # noqa: E402
from matchmaker_eval.submission import DECLARATION_KEY  # noqa: E402

N_PIECES = 146


def fetch_verified(url, path, sha256):
    if not path.exists() or file_sha256(path) != sha256:
        download(url, path)
    if file_sha256(path) != sha256:
        raise ValueError(f"Hash mismatch: {path.name}")
    return path


def load_cases(cache):
    """The pinned AMT data, its base URL, and one case per piano piece."""
    from matchmaker_eval.fetch_data import frozen_source

    pin = frozen_source("amt")
    base_url = f"https://raw.githubusercontent.com/{pin['repository']}/{pin['commit']}/amt/"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = fetch_verified(base_url + "manifest.json", cache / "manifest.json",
                              pin["manifest_sha256"])
    cases = json.loads(manifest.read_text())["cases"]
    if len(cases) != N_PIECES or len({c["piece_id"] for c in cases}) != N_PIECES:
        raise ValueError("The AMT manifest does not cover the piano fold")
    return pin, base_url, {c["piece_id"]: c for c in cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", choices=["arzt", "pthmm", "SLT_OLTW"], default="arzt")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("amt-data"))
    args = parser.parse_args()
    pin, base_url, cases = load_cases(args.cache)
    score_piece = rs.run_piece

    def run_piece(method, piece, input_type, index, run_dir, save_plots, estimated_bpm=False):
        case = cases[piece.piece_id]
        midi = fetch_verified(base_url + case["path"], args.cache / case["path"], case["sha256"])

        # Stream the transcribed MIDI into the symbolic follower...
        def matchmaker_on_transcription(**kwargs):
            kwargs.update(input_type="midi", performance_file=str(midi), method=args.method)
            return Matchmaker(**kwargs)

        original = rs.Matchmaker
        rs.Matchmaker = matchmaker_on_transcription
        try:
            # ...but score it as audio: the recording's ground truth and audio TR threshold.
            result = score_piece(method, piece, "audio", index, run_dir, save_plots, False)
        finally:
            rs.Matchmaker = original

        result.update(amt_data_commit=pin["commit"], amt_output_sha256=case["sha256"],
                      amt_front_end_executed=False, timing_scope="symbolic follower only",
                      symbolic_method=args.method)
        (run_dir / f"{index}.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
        return result

    rs.run_piece = run_piece
    rs.builtin_metadata = lambda method, input_type: {
        "name": f"AMT + {args.method}", "input_type": "audio", "kind": "reference",
        "authors": ["Matchmaker benchmark"],
        "description": "Frozen online AMT output replayed through a symbolic follower.",
        rs.ESTIMATED_BPM_KEY: False, DECLARATION_KEY: True,
    }
    scoring_parts = rs.scoring_parts
    rs.scoring_parts = lambda: {
        **scoring_parts(), "amt_inputs": pin, "amt_symbolic_method": args.method,
        "amt_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    result = rs.evaluate_submission(builtin=args.method, input_type="audio", fold="eval",
                                    shard=args.shard, num_shards=args.num_shards,
                                    run_dir=args.output)
    if result["n_failed"]:
        raise SystemExit("Incomplete AMT replay")


if __name__ == "__main__":
    main()
