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

    def test_merge_retains_the_meter_and_criterion_needed_to_rescore(self, tmp_path):
        criterion = {"error_unit": "musical_beat", "threshold": 0.1, "window_duration": 30.0}
        meter = {"schema": 1, "position_unit": "denominator_note", "changes": [
            {"start_score_beat": 0.0, "numerator": 6, "denominator": 8,
             "denominator_notes_per_beat": 3},
            {"start_score_beat": 6.0, "numerator": 4, "denominator": 4,
             "denominator_notes_per_beat": 1},
        ]}
        record = shard_record(0, fold_size=1)
        record["pieces"][0]["tracking_criterion"] = criterion
        root = write_shards(tmp_path / "shards", [record])
        (root / "shard-0" / "meter_0.json").write_text(json.dumps(meter))
        merged = merge(root, tmp_path / "out")
        assert merged["pieces"][0]["tracking_criterion"] == criterion
        assert json.loads((tmp_path / "out" / "meter_0.json").read_text()) == meter


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


@pytest.mark.parametrize("missing_case", [False, True])
def test_diagnostic_merge_requires_every_selected_case(tmp_path, monkeypatch, missing_case):
    from argparse import Namespace
    from matchmaker_eval import diagnostics

    config = dict(suite="perturbations", input_type="midi", method="chosen")
    monkeypatch.setattr(diagnostics, "CONFIGS", {
        "chosen-midi": config, "other-midi": {**config, "method": "other"},
    })
    cases = [{"id": name, "suite": "perturbations", "input_type": "midi"}
             for name in ("a", "b")]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cases": cases}))
    shard = tmp_path / "shards" / "chosen-0"
    shard.mkdir(parents=True)
    record = dict(config="chosen-midi", shard=0, failures=[],
                  data={"commit": "data-revision"},
                  environment={"benchmark_commit": "code-revision"}, github_run_id="1",
                  pieces=[dict(case_id=c["id"], category="example", tracked=True, mean_b=0.)
                          for c in cases[:1 if missing_case else 2]])
    (shard / "metrics.json").write_text(json.dumps(record))
    args = Namespace(source=tmp_path / "shards", output=tmp_path / "out",
                     manifest=manifest, shards=1, only="chosen", input_type="midi")
    if missing_case:
        with pytest.raises(SystemExit, match="Incomplete diagnostic results"):
            diagnostics.merge(args)
    else:
        diagnostics.merge(args)
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["complete"] is (not missing_case)
    assert list(summary["configurations"]) == ["chosen-midi"]


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

    def test_auxiliary_folds_do_not_default_to_the_piano_destination(self):
        from merge_references import output_for_fold

        assert output_for_fold("beyond_piano") != output_for_fold("eval")

    @pytest.mark.parametrize("fold,expected", [
        ("beyond_piano", ["arzt-audio"]),
        ("eval", ["arzt-audio", "pthmm-midi"]),
    ])
    def test_wrong_fold_or_missing_method_cannot_overwrite_results(
        self, tmp_path, monkeypatch, fold, expected,
    ):
        from merge_references import main

        root = self._flat_download(tmp_path, ["arzt-audio"])
        published = tmp_path / "published" / "arzt-audio"
        published.mkdir(parents=True)
        record = published / "metrics.json"
        record.write_text('{"previous": true}')
        monkeypatch.setattr(sys, "argv", [
            "merge_references.py", str(root), "--fold", fold,
            "--expected-entries-json", json.dumps(expected),
            "--output", str(published.parent),
        ])
        assert main() == 1
        assert record.read_text() == '{"previous": true}'


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


def test_diagnostic_sharding_is_exhaustive_and_disjoint():
    from matchmaker_eval.diagnostics import select_cases
    m = {'cases': [dict(id=str(i), suite='perturbations', input_type='audio') for i in range(41)]}
    config = {'suite': 'perturbations', 'input_type': 'audio'}
    shards = [select_cases(m, config, s, 4) for s in range(4)]
    assert sorted(i for batch in shards for i, _ in batch) == list(range(41))


def test_diagnostic_duplicate_cases_rejected():
    from matchmaker_eval.diagnostics import select_cases
    c = dict(id='a', suite='perturbations', input_type='audio')
    with pytest.raises(ValueError, match='duplicated'):
        select_cases({'cases':[c,c]}, {'suite': 'perturbations', 'input_type': 'audio'})


def test_diagnostic_merge_fails_incomplete_and_preserves_evidence(tmp_path):
    from argparse import Namespace
    from matchmaker_eval.diagnostics import CONFIGS, merge
    source=tmp_path/'source';source.mkdir()
    manifest=tmp_path/'manifest.json'
    cases=[dict(id=k,suite=c['suite'],input_type=c['input_type']) for k,c in CONFIGS.items()]
    # Unique IDs per suite/modality are enough; all output shards are absent.
    manifest.write_text(json.dumps({'cases':cases}))
    with pytest.raises(SystemExit,match='Incomplete'):
        merge(Namespace(source=source,output=tmp_path/'out',manifest=manifest,shards=4,
                        only='',input_type='both'))
    assert not json.loads((tmp_path/'out/summary.json').read_text())['complete']


def test_diagnostic_merge_rejects_different_patched_sources(tmp_path, monkeypatch):
    from argparse import Namespace
    from matchmaker_eval import diagnostics as D

    config = {'suite': 'test', 'input_type': 'midi', 'method': 'test'}
    monkeypatch.setattr(D, 'CONFIGS', {'test-midi': config})
    source = tmp_path / 'shards'
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'cases': [dict(id=str(i), **config) for i in range(2)]}))
    for i in range(2):
        folder = source / str(i)
        folder.mkdir(parents=True)
        record = dict(config='test-midi', shard=i, failures=[], data={'commit': 'data'},
                      environment={'benchmark_commit': 'benchmark', 'matchmaker_base_commit': 'base',
                                   'matchmaker_patches': [{'head': str(i)}]}, github_run_id='run',
                      pieces=[dict(case_id=str(i), category='test', tracked=True, mean_b=0.)])
        (folder / 'metrics.json').write_text(json.dumps(record))
    with pytest.raises(SystemExit, match='inconsistent provenance'):
        D.merge(Namespace(source=source, output=tmp_path / 'out', manifest=manifest, shards=2,
                          only='', input_type='both'))
