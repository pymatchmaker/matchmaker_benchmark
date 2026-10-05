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


class TestTheFoldCarriesIt:
    """The tempo is frozen in the fold, beside the paths it belongs to."""

    def test_the_example_piece_publishes_one(self):
        from matchmaker_eval.folds import load_fold

        piece = load_fold("example", input_type="midi")[0]
        assert piece.performance_tempo == 58.0

    def test_every_eval_row_has_one(self):
        """A missing value would silently disarm `estimated_bpm: true` runs."""
        from matchmaker_eval.folds import load_fold

        for fold in ("eval", "valid"):
            blank = [p.piece_id for p in load_fold(fold) if not p.estimated_bpm]
            assert blank == [], f"{fold}: {len(blank)} row(s) without a tempo"

    def test_it_is_one_value_per_performance_not_per_container(self):
        """mp3 and MIDI are two renderings of the same playing, so one column.

        A per-container tempo would have to be two columns, and a follower's
        audio and MIDI runs would then be measured against different tempi.
        """
        from matchmaker_eval.folds import FOLD_COLUMNS

        assert [c for c in FOLD_COLUMNS if "bpm" in c] == ["estimated_bpm"]

    def test_an_absent_value_reads_as_none(self):
        from matchmaker_eval.folds import Piece

        piece = Piece("p", "asap", "Bach", "t", "s.musicxml", "p.mid", "", "p.match")
        assert piece.performance_tempo is None

    def test_a_non_numeric_value_is_refused(self):
        """Better a named failure than a follower handed a nonsense tempo."""
        from matchmaker_eval.folds import FoldError, Piece

        piece = Piece(
            "p", "asap", "Bach", "t", "s.musicxml", "p.mid", "", "p.match",
            estimated_bpm="presto",
        )
        with pytest.raises(FoldError, match="not a number"):
            piece.performance_tempo


class TestOlderFoldsStillLoad:
    """The column arrived after folds were already in circulation."""

    def test_a_fold_without_the_column_loads(self, tmp_path):
        from matchmaker_eval.folds import REQUIRED_FOLD_COLUMNS, load_fold

        csv_path = tmp_path / "old.csv"
        csv_path.write_text(
            ",".join(REQUIRED_FOLD_COLUMNS) + "\n"
            "local/x/y,local,Bach,x,s.musicxml,p.mid,p.wav,p.match,3\n"
        )
        piece = load_fold(csv_path)[0]
        assert piece.estimated_bpm == ""
        assert piece.performance_tempo is None

    def test_a_required_column_is_still_required(self, tmp_path):
        from matchmaker_eval.folds import FoldError, load_fold

        csv_path = tmp_path / "broken.csv"
        csv_path.write_text("piece_id,dataset\nlocal/x/y,local\n")
        with pytest.raises(FoldError, match="missing columns"):
            load_fold(csv_path)


class TestMissingTempoFallsBack:
    """matchmaker already has a fallback: the score's marking, then 120 BPM.

    Failing the run instead would be worse than the thing it guards against —
    a follower that never sees a tempo simply performs the ordinary task.
    """

    def _piece_without_tempo(self):
        import dataclasses

        from matchmaker_eval.folds import load_fold

        piece = load_fold("example", input_type="midi")[0]
        return dataclasses.replace(piece, estimated_bpm="")

    def test_it_runs_anyway(self, tmp_path):
        from run_submission import run_piece

        flat, _ = run_piece("pthmm", self._piece_without_tempo(), "midi", 1,
                            tmp_path, False, estimated_bpm=True)
        assert flat["used_estimated_bpm"] is False
        assert flat["tracked"] is True

    def test_a_piece_with_a_tempo_records_that_it_used_one(self, tmp_path):
        from matchmaker_eval.folds import load_fold
        from run_submission import run_piece

        piece = load_fold("example", input_type="midi")[0]
        flat, _ = run_piece("pthmm", piece, "midi", 1, tmp_path, False,
                            estimated_bpm=True)
        assert flat["used_estimated_bpm"] is True

    def test_an_entry_that_never_asked_records_false(self, tmp_path):
        from matchmaker_eval.folds import load_fold
        from run_submission import run_piece

        piece = load_fold("example", input_type="midi")[0]
        flat, _ = run_piece("pthmm", piece, "midi", 1, tmp_path, False,
                            estimated_bpm=False)
        assert flat["used_estimated_bpm"] is False


class TestManifestShapes:
    """make_folds reads the manifests; nothing does so at run time any more."""

    def _columns(self, header):
        from matchmaker_eval.make_folds import resolve_columns

        return resolve_columns(header)

    def test_the_tempo_column_is_recognised(self):
        assert self._columns(
            ["audio", "score", "midi", "match", "estimated_bpm"]
        )["estimated_bpm"] == "estimated_bpm"

    def test_an_alternative_column_name_is_accepted(self):
        """For a data branch caught mid-rename."""
        assert self._columns(
            ["audio", "score", "midi", "match", "tempo"]
        )["estimated_bpm"] == "tempo"

    def test_a_manifest_without_it_resolves_to_nothing(self):
        assert self._columns(["audio", "score", "midi", "match"])[
            "estimated_bpm"
        ] is None

    def test_a_per_dataset_manifest_yields_the_tempo(self, tmp_path):
        from matchmaker_eval.make_folds import read_repo_metadata

        path = tmp_path / "metadata-asap.csv"
        path.write_text(
            "audio,score,midi,match,estimated_bpm\n"
            "asap/audio/a.mp3,asap/score/s.musicxml,asap/midi/a.mid,"
            "asap/match/a.match,132.5\n"
        )
        piece = list(read_repo_metadata("asap", path))[0]
        assert piece.estimated_bpm == "132.5"
        assert piece.performance_tempo == 132.5
        assert piece.midi_performance == "midi/a.mid"

    def test_a_root_manifest_yields_the_tempo(self, tmp_path):
        """One file covers every dataset, keyed by a `dataset` column."""
        from matchmaker_eval.make_folds import read_root_metadata

        path = tmp_path / "metadata-valid.csv"
        path.write_text(
            "dataset,audio,midi,score,match,estimated_bpm\n"
            "batik,batik/audio/a.mp3,batik/midi/a.mid,batik/score/a.musicxml,"
            "batik/match/a.match,69\n"
            "asap,asap/audio/b.mp3,asap/midi/b.mid,asap/score/b.musicxml,"
            "asap/match/b.match,120\n"
        )
        by_dataset = {p.dataset: p for p in read_root_metadata(path)}
        assert by_dataset["batik"].performance_tempo == 69.0
        assert by_dataset["asap"].performance_tempo == 120.0

    def test_the_offline_path_leaves_the_tempo_blank(self, tmp_path):
        """It must not guess: data/perf_tempo_estimate/ disagrees on 4 rows.

        A blank stops an `estimated_bpm: true` run; a plausible wrong number
        would quietly change what that run measured.
        """
        from matchmaker_eval.make_folds import read_metadata_rows

        path = tmp_path / "metadata-vienna.csv"
        path.write_text(
            "composer,title,xml_score,midi_performance,audio_performance,"
            "match,difficulty\n"
            "Chopin,Chopin_op10_no3,musicxml/Chopin_op10_no3.musicxml,"
            "midi/Chopin_op10_no3_p08.mid,audio/Chopin_op10_no3_p08.wav,"
            "match/Chopin_op10_no3_p08.match,1\n"
        )
        assert list(read_metadata_rows("vienna", path))[0].estimated_bpm == ""
