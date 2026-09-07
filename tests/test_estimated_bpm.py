"""Followers may be given the performance's tempo — visibly, and by exception.

The default is that a tracker knows nothing about the performance it is about
to hear. A follower that is handed the estimated tempo is doing a different task,
so it must declare it, and the leaderboard must mark it.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "matchmaker_eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from matchmaker_eval.leaderboard import render_table, row_from_metrics
from matchmaker_eval.submission import (
    ESTIMATED_BPM_KEY,
    SubmissionError,
    builtin_metadata,
    read_metadata,
)


def metrics(**overrides) -> dict:
    record = {
        "submission": "demo",
        "method": "demo",
        "kind": "submission",
        "input_type": "midi",
        "metadata": {"name": "Demo", "authors": ["a"]},
        "n_pieces": 146,
        "n_tracked": 100,
        "summary_all": {"tracking_rate": 0.685, "beat": {"median": 0.1}},
        "summary_tracked": {"beat": {"median": 0.05}, "ms": {"median": 20}},
        "estimated_bpm": False,
    }
    record.update(overrides)
    return record


class TestDefaultIsOff:
    def test_a_submission_without_the_key_does_not_get_the_tempo(self, tmp_path):
        (tmp_path / "metadata.yaml").write_text(
            "name: X\nauthors: [a]\ninput_type: midi\ndescription: d\n"
            "eval_fold_untouched: true\n"
        )
        assert read_metadata(tmp_path)[ESTIMATED_BPM_KEY] is False

    def test_most_references_do_not_use_it(self):
        for method in ("pthmm", "arzt", "dixon", "hmm", "outerhmm"):
            assert builtin_metadata(method, "midi")[ESTIMATED_BPM_KEY] is False

    def test_a_record_without_the_field_reads_as_off(self):
        row = row_from_metrics(metrics(estimated_bpm=None))
        assert row["estimated_bpm"] is False


class TestDeclaring:
    def test_a_submission_can_opt_in(self, tmp_path):
        (tmp_path / "metadata.yaml").write_text(
            "name: X\nauthors: [a]\ninput_type: midi\ndescription: d\n"
            "eval_fold_untouched: true\nestimated_bpm: true\n"
        )
        assert read_metadata(tmp_path)[ESTIMATED_BPM_KEY] is True

    def test_a_non_boolean_declaration_is_refused(self, tmp_path):
        (tmp_path / "metadata.yaml").write_text(
            "name: X\nauthors: [a]\ninput_type: midi\ndescription: d\n"
            "eval_fold_untouched: true\nestimated_bpm: sometimes\n"
        )
        with pytest.raises(SubmissionError, match="must be true or false"):
            read_metadata(tmp_path)

    @pytest.mark.parametrize("input_type", ["midi", "audio"])
    def test_the_particle_filter_declares_it(self, input_type):
        assert builtin_metadata("pfkorz", input_type)[ESTIMATED_BPM_KEY] is True


class TestTheLeaderboardMarksIt:
    def test_the_row_carries_the_flag(self):
        assert row_from_metrics(metrics(estimated_bpm=True))["estimated_bpm"] is True

    def test_the_table_marks_the_entry(self):
        board = {"entries": [row_from_metrics(metrics(estimated_bpm=True))]}
        board["entries"][0]["rank"] = 1
        text = render_table(board)
        assert "demo*" in text

    def test_an_ordinary_entry_is_not_marked(self):
        board = {"entries": [row_from_metrics(metrics())]}
        board["entries"][0]["rank"] = 1
        assert "demo*" not in render_table(board)

    def test_the_footnote_explains_the_asterisk(self):
        board = {"entries": [row_from_metrics(metrics(estimated_bpm=True))]}
        board["entries"][0]["rank"] = 1
        text = render_table(board)
        assert "*" in text.splitlines()[-1]
        assert "tempo" in text.splitlines()[-1]

    def test_no_footnote_when_nobody_used_it(self):
        board = {"entries": [row_from_metrics(metrics())]}
        board["entries"][0]["rank"] = 1
        assert "given the performance's tempo" not in render_table(board)

    def test_the_csv_has_the_column(self):
        from matchmaker_eval.leaderboard import CSV_COLUMNS

        assert "estimated_bpm" in CSV_COLUMNS

    def test_the_published_glossary_explains_it(self):
        from matchmaker_eval.leaderboard import build

        assert "estimated_bpm" in build()["metrics"]


class TestLookup:
    def test_a_piece_reports_none_when_no_tempo_is_published(self):
        from matchmaker_eval.folds import clear_bpm_cache, load_fold

        clear_bpm_cache()
        piece = load_fold("example", input_type="midi")[0]
        assert piece.estimated_bpm("midi") is None

    def test_it_reads_the_column_from_the_dataset_metadata(self, tmp_path, monkeypatch):
        import matchmaker_eval.folds as F

        F.clear_bpm_cache()
        root = tmp_path / "asap"
        root.mkdir()
        (root / "metadata-asap.csv").write_text(
            "audio,score,midi,match,estimated_bpm\n"
            "asap/audio/a.mp3,asap/score/s.musicxml,asap/midi/a.mid,"
            "asap/match/a.match,132.5\n"
        )
        monkeypatch.setattr(F, "DATA_ROOT", tmp_path)
        table = F._bpm_table("asap")
        assert table["asap/midi/a.mid"] == 132.5
        assert table["asap/audio/a.mp3"] == 132.5
        F.clear_bpm_cache()

    def test_a_metadata_file_without_the_column_yields_nothing(
        self, tmp_path, monkeypatch
    ):
        import matchmaker_eval.folds as F

        F.clear_bpm_cache()
        root = tmp_path / "asap"
        root.mkdir()
        (root / "metadata-asap.csv").write_text("audio,score,midi,match\na,b,c,d\n")
        monkeypatch.setattr(F, "DATA_ROOT", tmp_path)
        assert F._bpm_table("asap") == {}
        F.clear_bpm_cache()

    def test_the_column_name_is_configurable(self):
        from matchmaker_eval.folds import ESTIMATED_BPM_COLUMN, _bpm_column

        assert _bpm_column()  # falls back to the default when unset
        assert ESTIMATED_BPM_COLUMN == "estimated_bpm"


class TestMissingTempoIsFatal:
    def test_declaring_it_without_the_data_raises(self):
        """Silently running without it would mislabel the row."""
        from matchmaker_eval.folds import EstimatedBpmUnavailable, load_fold
        from run_submission import run_piece

        piece = load_fold("example", input_type="midi")[0]
        with pytest.raises(EstimatedBpmUnavailable, match="no estimated tempo"):
            run_piece("pthmm", piece, "midi", 1, Path("/tmp"), False,
                      estimated_bpm=True)


class TestManifestShapes:
    """Branches use either per-dataset manifests or one at the repository root."""

    def test_a_root_manifest_is_read(self, tmp_path, monkeypatch):
        import matchmaker_eval.folds as F

        F.clear_bpm_cache()
        monkeypatch.setattr(F, "DATA_ROOT", tmp_path)
        (tmp_path / "valid--metadata-valid.csv").write_text(
            "dataset,audio,midi,score,match,estimated_bpm\n"
            "batik,batik/audio/a.mp3,batik/midi/a.mid,batik/score/a.musicxml,"
            "batik/match/a.match,69\n"
        )
        assert F._bpm_table("batik")["batik/midi/a.mid"] == 69.0
        F.clear_bpm_cache()

    def test_a_root_manifest_is_filtered_by_dataset(self, tmp_path, monkeypatch):
        """One file covers every dataset, so rows must not leak across them."""
        import matchmaker_eval.folds as F

        F.clear_bpm_cache()
        monkeypatch.setattr(F, "DATA_ROOT", tmp_path)
        (tmp_path / "valid--metadata-valid.csv").write_text(
            "dataset,midi,estimated_bpm\n"
            "batik,batik/midi/a.mid,69\n"
            "asap,asap/midi/b.mid,120\n"
        )
        assert F._bpm_table("batik") == {"batik/midi/a.mid": 69.0}
        F.clear_bpm_cache()
        assert F._bpm_table("asap") == {"asap/midi/b.mid": 120.0}
        F.clear_bpm_cache()

    def test_an_alternative_column_name_can_be_configured(
        self, tmp_path, monkeypatch
    ):
        """The config takes a list, for a branch caught mid-rename."""
        import matchmaker_eval.folds as F

        F.clear_bpm_cache()
        monkeypatch.setattr(F, "DATA_ROOT", tmp_path)
        monkeypatch.setattr(F, "_bpm_column", lambda: ["estimated_bpm", "tempo"])
        root = tmp_path / "vienna"
        root.mkdir()
        (root / "metadata-vienna.csv").write_text(
            "audio,score,midi,match,tempo\n"
            "vienna/audio/a.mp3,vienna/score/a.musicxml,vienna/midi/a.mid,"
            "vienna/match/a.match,88\n"
        )
        assert F._bpm_table("vienna")["vienna/midi/a.mid"] == 88.0
        F.clear_bpm_cache()

    def test_the_configured_names_are_a_list(self):
        from matchmaker_eval.folds import _bpm_column

        names = _bpm_column()
        assert isinstance(names, list)
        assert names[0] == "estimated_bpm"
