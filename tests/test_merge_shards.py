"""Merging shards must not lose what the leaderboard reads off a record.

The evaluation workflow always shards, so anything ``merge_shards.py`` drops is
simply absent from the published row. ``kind`` and ``method`` were dropped,
which turned every reference method into an unnamed "submission".
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "matchmaker_eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from matchmaker_eval.leaderboard import row_from_metrics
from merge_shards import MergeError, merge


def shard_record(index: int, **overrides) -> dict:
    record = {
        "submission": "pthmm-midi",
        "method": "pthmm",
        "kind": "reference",
        "metadata": {"name": "Pitch HMM", "authors": ["matchmaker"]},
        "fold": "eval",
        "fold_file": "data/folds/eval.csv",
        "fold_sha256": "abc123",
        "input_type": "midi",
        "environment": {"matchmaker": "0.3.0"},
        "timestamp": "2026-01-01T00:00:00+00:00",
        "fold_size": 2,
        "n_pieces": 1,
        "n_tracked": 1,
        "partial": True,
        "failures": [],
        "pieces": [
            {
                "index": index,
                "piece_id": f"p{index}",
                "dataset": "asap",
                "title": f"t{index}",
                "tracked": True,
            }
        ],
    }
    record.update(overrides)
    return record


def write_shards(root: Path, records) -> Path:
    for i, record in enumerate(records):
        shard = root / f"shard-{i}"
        shard.mkdir(parents=True)
        (shard / "metrics.json").write_text(json.dumps(record))
    return root


class TestMergePreservesIdentity:
    def test_kind_and_method_survive_the_merge(self, tmp_path):
        root = write_shards(tmp_path / "shards", [shard_record(0), shard_record(1)])
        merged = merge(root, tmp_path / "out")
        assert merged["kind"] == "reference"
        assert merged["method"] == "pthmm"

    def test_the_leaderboard_row_keeps_the_reference_label(self, tmp_path):
        root = write_shards(tmp_path / "shards", [shard_record(0), shard_record(1)])
        merged = merge(root, tmp_path / "out")
        row = row_from_metrics(merged)
        assert row["kind"] == "reference"
        assert row["method"] == "pthmm"

    def test_a_complete_merge_is_not_partial(self, tmp_path):
        root = write_shards(tmp_path / "shards", [shard_record(0), shard_record(1)])
        merged = merge(root, tmp_path / "out")
        assert merged["n_pieces"] == 2
        assert merged["partial"] is False

    def test_a_short_merge_stays_partial(self, tmp_path):
        root = write_shards(tmp_path / "shards", [shard_record(0)])
        merged = merge(root, tmp_path / "out")
        assert merged["partial"] is True


class TestMergeRejectsMismatchedShards:
    @pytest.mark.parametrize("key,value", [
        ("method", "arzt"),
        ("kind", "submission"),
        ("input_type", "audio"),
        ("fold_sha256", "different"),
    ])
    def test_shards_must_describe_the_same_run(self, tmp_path, key, value):
        root = write_shards(
            tmp_path / "shards",
            [shard_record(0), shard_record(1, **{key: value})],
        )
        with pytest.raises(MergeError, match=key):
            merge(root, tmp_path / "out")

    def test_a_duplicated_piece_index_is_rejected(self, tmp_path):
        root = write_shards(tmp_path / "shards", [shard_record(0), shard_record(0)])
        with pytest.raises(MergeError, match="more than one shard"):
            merge(root, tmp_path / "out")
