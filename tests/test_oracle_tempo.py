"""Followers may be given the performance's tempo — visibly, and by exception.

The default is that a tracker knows nothing about the performance it is about
to hear. A follower that is handed the oracle tempo is doing a different task,
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
    ORACLE_TEMPO_KEY,
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
        "oracle_tempo": False,
    }
    record.update(overrides)
    return record


class TestDefaultIsOff:
    def test_a_submission_without_the_key_does_not_get_the_tempo(self, tmp_path):
        (tmp_path / "metadata.yaml").write_text(
            "name: X\nauthors: [a]\ninput_type: midi\ndescription: d\n"
            "eval_fold_untouched: true\n"
        )
        assert read_metadata(tmp_path)[ORACLE_TEMPO_KEY] is False

    def test_most_references_do_not_use_it(self):
        for method in ("pthmm", "arzt", "dixon", "hmm", "outerhmm"):
            assert builtin_metadata(method, "midi")[ORACLE_TEMPO_KEY] is False

    def test_a_record_without_the_field_reads_as_off(self):
        row = row_from_metrics(metrics(oracle_tempo=None))
        assert row["oracle_tempo"] is False


class TestDeclaring:
    def test_a_submission_can_opt_in(self, tmp_path):
        (tmp_path / "metadata.yaml").write_text(
            "name: X\nauthors: [a]\ninput_type: midi\ndescription: d\n"
            "eval_fold_untouched: true\noracle_tempo: true\n"
        )
        assert read_metadata(tmp_path)[ORACLE_TEMPO_KEY] is True

    def test_a_non_boolean_declaration_is_refused(self, tmp_path):
        (tmp_path / "metadata.yaml").write_text(
            "name: X\nauthors: [a]\ninput_type: midi\ndescription: d\n"
            "eval_fold_untouched: true\noracle_tempo: sometimes\n"
        )
        with pytest.raises(SubmissionError, match="must be true or false"):
            read_metadata(tmp_path)

    @pytest.mark.parametrize("input_type", ["midi", "audio"])
    def test_the_particle_filter_declares_it(self, input_type):
        assert builtin_metadata("pfkorz", input_type)[ORACLE_TEMPO_KEY] is True


class TestTheLeaderboardMarksIt:
    def test_the_row_carries_the_flag(self):
        assert row_from_metrics(metrics(oracle_tempo=True))["oracle_tempo"] is True

    def test_the_table_marks_the_entry(self):
        board = {"entries": [row_from_metrics(metrics(oracle_tempo=True))]}
        board["entries"][0]["rank"] = 1
        text = render_table(board)
        assert "demo*" in text

    def test_an_ordinary_entry_is_not_marked(self):
        board = {"entries": [row_from_metrics(metrics())]}
        board["entries"][0]["rank"] = 1
        assert "demo*" not in render_table(board)

    def test_the_footnote_explains_the_asterisk(self):
        board = {"entries": [row_from_metrics(metrics(oracle_tempo=True))]}
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

        assert "oracle_tempo" in CSV_COLUMNS

    def test_the_published_glossary_explains_it(self):
        from matchmaker_eval.leaderboard import build

        assert "oracle_tempo" in build()["metrics"]


class TestLookup:
    def test_a_piece_reports_none_when_no_tempo_is_published(self):
        from matchmaker_eval.folds import clear_tempo_cache, load_fold

        clear_tempo_cache()
        piece = load_fold("example", input_type="midi")[0]
        assert piece.oracle_tempo("midi") is None

    def test_it_reads_the_column_from_the_dataset_metadata(self, tmp_path, monkeypatch):
        import matchmaker_eval.folds as F

        F.clear_tempo_cache()
        root = tmp_path / "asap"
        root.mkdir()
        (root / "metadata-asap.csv").write_text(
            "audio,score,midi,match,estimated_bpm\n"
            "asap/audio/a.mp3,asap/score/s.musicxml,asap/midi/a.mid,"
            "asap/match/a.match,132.5\n"
        )
        monkeypatch.setattr(F, "DATA_ROOT", tmp_path)
        table = F._tempo_table("asap")
        assert table["asap/midi/a.mid"] == 132.5
        assert table["asap/audio/a.mp3"] == 132.5
        F.clear_tempo_cache()

    def test_a_metadata_file_without_the_column_yields_nothing(
        self, tmp_path, monkeypatch
    ):
        import matchmaker_eval.folds as F

        F.clear_tempo_cache()
        root = tmp_path / "asap"
        root.mkdir()
        (root / "metadata-asap.csv").write_text("audio,score,midi,match\na,b,c,d\n")
        monkeypatch.setattr(F, "DATA_ROOT", tmp_path)
        assert F._tempo_table("asap") == {}
        F.clear_tempo_cache()

    def test_the_column_name_is_configurable(self):
        from matchmaker_eval.folds import ORACLE_TEMPO_COLUMN, _tempo_column

        assert _tempo_column()  # falls back to the default when unset
        assert ORACLE_TEMPO_COLUMN == "estimated_bpm"


class TestMissingTempoIsFatal:
    def test_declaring_it_without_the_data_raises(self):
        """Silently running without it would mislabel the row."""
        from matchmaker_eval.folds import OracleTempoUnavailable, load_fold
        from run_submission import run_piece

        piece = load_fold("example", input_type="midi")[0]
        with pytest.raises(OracleTempoUnavailable, match="no oracle tempo"):
            run_piece("pthmm", piece, "midi", 1, Path("/tmp"), False,
                      oracle_tempo=True)
