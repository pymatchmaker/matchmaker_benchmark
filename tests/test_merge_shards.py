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


class TestMergeReferences:
    """One references run produces shards for many entries, flat in one dir."""

    def _flat_download(self, tmp_path, entries, shards=2):
        root = tmp_path / "refshards"
        root.mkdir()
        for entry in entries:
            for s in range(shards):
                d = root / f"refshard-{entry}-{s}"
                d.mkdir()
                rec = shard_record(s)
                rec["submission"] = entry
                rec["fold_size"] = shards
                (d / "metrics.json").write_text(json.dumps(rec))
        return root

    def test_shards_are_grouped_by_entry(self, tmp_path):
        from merge_references import group

        root = self._flat_download(tmp_path, ["arzt-audio", "pthmm-midi"])
        grouped = group(root)
        assert sorted(grouped) == ["arzt-audio", "pthmm-midi"]
        assert all(len(v) == 2 for v in grouped.values())

    def test_a_method_name_containing_a_dash_still_groups(self, tmp_path):
        """`refshard-<entry>-<n>`: only the trailing number is the shard."""
        from merge_references import group

        root = self._flat_download(tmp_path, ["baseline-constant-tempo-audio"])
        assert sorted(group(root)) == ["baseline-constant-tempo-audio"]

    def test_each_entry_merges_separately(self, tmp_path):
        from merge_references import merge_all

        root = self._flat_download(tmp_path, ["arzt-audio", "pthmm-midi"])
        report = merge_all(root, tmp_path / "out", tmp_path / "staging")
        assert sorted(report) == ["arzt-audio", "pthmm-midi"]
        for info in report.values():
            assert info["n_pieces"] == 2
            assert info["partial"] is False
        assert (tmp_path / "out" / "arzt-audio" / "metrics.json").exists()
        assert (tmp_path / "out" / "pthmm-midi" / "metrics.json").exists()

    def test_an_entry_missing_shards_stays_partial(self, tmp_path):
        from merge_references import merge_all

        root = self._flat_download(tmp_path, ["arzt-audio"], shards=2)
        # Drop one shard, as a failed matrix job would.
        import shutil

        shutil.rmtree(root / "refshard-arzt-audio-1")
        report = merge_all(root, tmp_path / "out", tmp_path / "staging")
        assert report["arzt-audio"]["partial"] is True

    def test_an_empty_download_is_an_error(self, tmp_path):
        from merge_references import MergeError, merge_all

        (tmp_path / "empty").mkdir()
        with pytest.raises(MergeError, match="no refshard"):
            merge_all(tmp_path / "empty", tmp_path / "out", tmp_path / "s")

    def test_unrelated_directories_are_ignored(self, tmp_path):
        from merge_references import group

        root = self._flat_download(tmp_path, ["arzt-audio"])
        (root / "some-other-artifact").mkdir()
        assert sorted(group(root)) == ["arzt-audio"]


class TestTheTempoLabelSurvivesMerging:
    """A sharded run must not lose the fact that it was given the tempo."""

    def _merge(self, tmp_path, **overrides):
        records = [shard_record(1, **overrides), shard_record(2, **overrides)]
        root = write_shards(tmp_path / "shards", records)
        return merge(root, tmp_path / "merged")

    def test_the_flag_is_carried(self, tmp_path):
        """Dropping it published a pfkorz run unmarked, beside entries
        that never had the tempo."""
        merged = self._merge(tmp_path, estimated_bpm=True)
        assert merged["estimated_bpm"] is True

    def test_an_ordinary_run_stays_false(self, tmp_path):
        assert self._merge(tmp_path, estimated_bpm=False)["estimated_bpm"] is False

    def test_shards_disagreeing_on_it_is_refused(self, tmp_path):
        """Two different measurements must not be averaged into one row."""
        records = [
            shard_record(1, estimated_bpm=True),
            shard_record(2, estimated_bpm=False),
        ]
        root = write_shards(tmp_path / "shards", records)
        with pytest.raises(MergeError, match="estimated_bpm"):
            merge(root, tmp_path / "merged")

    def test_the_count_is_recomputed_from_the_pieces(self, tmp_path):
        one = shard_record(1, estimated_bpm=True)
        one["pieces"][0]["used_estimated_bpm"] = True
        two = shard_record(2, estimated_bpm=True)
        two["pieces"][0]["used_estimated_bpm"] = False
        root = write_shards(tmp_path / "shards", [one, two])
        merged = merge(root, tmp_path / "merged")
        assert merged["n_estimated_bpm"] == 1
