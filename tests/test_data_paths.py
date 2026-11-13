"""Where the benchmark looks for a dataset, and what happens when it is absent.

The data lives in a separate repository and is downloaded on demand, so the
answer to "file not found" should almost always be "fetch it", not "fail". The
one thing that used to break that was the legacy-directory fallback: a local
copy sharing a name with an upstream release was trusted without checking it
was laid out the way the folds address files.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "matchmaker_eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    import matchmaker_eval.folds as F

    monkeypatch.setattr(F, "DATA_ROOT", tmp_path)
    return tmp_path


class TestDatasetRoot:
    def test_the_repository_layout_wins(self, data_root):
        from matchmaker_eval.folds import dataset_root

        (data_root / "asap" / "score").mkdir(parents=True)
        (data_root / "asap-dataset-matchmaker" / "score").mkdir(parents=True)
        assert dataset_root("asap") == data_root / "asap"

    def test_nothing_on_disk_gives_the_fetch_target(self, data_root):
        """So a download lands where the folds will look for it."""
        from matchmaker_eval.folds import dataset_root

        assert dataset_root("asap") == data_root / "asap"

    def test_a_legacy_copy_in_the_expected_shape_is_reused(self, data_root):
        from matchmaker_eval.folds import dataset_root

        legacy = data_root / "vienna4x22"
        (legacy / "score").mkdir(parents=True)
        (legacy / "midi").mkdir()
        assert dataset_root("vienna") == legacy

    def test_a_legacy_copy_in_the_upstream_shape_is_ignored(self, data_root):
        """The bug: same name, different insides.

        Batik unpacks with `scores_effective/`, not the flat `score/` the fold
        CSVs name. Returning it sent every path into a differently-shaped tree
        and the run failed on paths nobody had configured.
        """
        from matchmaker_eval.folds import dataset_root

        (data_root / "batik_plays_mozart" / "scores_effective").mkdir(parents=True)
        assert dataset_root("batik") == data_root / "batik", (
            "an upstream-shaped copy must not shadow the fetch target"
        )

    def test_an_empty_legacy_directory_is_ignored(self, data_root):
        from matchmaker_eval.folds import dataset_root

        (data_root / "batik_plays_mozart").mkdir()
        assert dataset_root("batik") == data_root / "batik"

    def test_local_pieces_come_from_the_repository(self, data_root):
        from matchmaker_eval.folds import REPO_ROOT as R, dataset_root

        assert dataset_root("local") == R

    @pytest.mark.parametrize("layout", ["score", "midi", "match", "audio"])
    def test_any_expected_subdirectory_counts(self, data_root, layout):
        from matchmaker_eval.folds import has_expected_layout

        d = data_root / "whatever"
        (d / layout).mkdir(parents=True)
        assert has_expected_layout(d)

    def test_a_file_is_not_a_layout(self, data_root):
        from matchmaker_eval.folds import has_expected_layout

        f = data_root / "notadir"
        f.write_text("")
        assert not has_expected_layout(f)


class TestMissingDataMessage:
    def test_it_says_how_to_fetch(self, data_root):
        from matchmaker_eval.folds import FoldError, load_fold, require_files

        pieces = load_fold("example", input_type="midi")
        # Point the local dataset somewhere empty so the files are missing.
        import matchmaker_eval.folds as F

        monkey = data_root / "nothing"
        monkey.mkdir()
        original = F.REPO_ROOT
        F.REPO_ROOT = monkey
        try:
            with pytest.raises(FoldError) as excinfo:
                require_files(pieces, "midi")
        finally:
            F.REPO_ROOT = original
        message = str(excinfo.value)
        assert "fetch_data.py" in message
        assert "--input-type midi" in message, (
            "the suggested command should be runnable as printed"
        )


class TestMetadataRunnerPaths:
    """The CSV runners address the upstream layout, not the data repository.

    They used to hardcode ~/data/<upstream name>, ignoring MATCHMAKER_DATA_DIR
    entirely — which is how they came to be run against a directory nobody had
    configured. They now resolve through the data root like everything else,
    and say what is missing instead of failing per piece.
    """

    def test_the_root_follows_the_data_dir(self, data_root):
        from matchmaker_eval.folds import nested_dataset_root

        assert nested_dataset_root("asap") == data_root / "asap-dataset-matchmaker"
        assert nested_dataset_root("batik") == data_root / "batik_plays_mozart"

    def test_an_unknown_dataset_uses_its_own_name(self, data_root):
        from matchmaker_eval.folds import nested_dataset_root

        assert nested_dataset_root("urmp") == data_root / "urmp"

    def test_the_runners_no_longer_hardcode_a_home_directory(self):
        for name in ("test_symbolic.py", "test_audio.py"):
            text = (REPO_ROOT / "matchmaker_eval" / name).read_text()
            assert '"~/data' not in text, (
                f"{name} hardcodes a data directory instead of resolving it "
                "through MATCHMAKER_DATA_DIR"
            )
            assert "nested_dataset_root(" in text

    def test_the_message_names_both_ways_forward(self):
        from pathlib import Path as P

        from matchmaker_eval.folds import explain_missing_dataset

        message = explain_missing_dataset("asap", P("/nowhere/x.musicxml"))
        assert "MATCHMAKER_DATA_DIR" in message
        assert "run_submission.py" in message and "--fold valid" in message, (
            "a user who cannot run these should be pointed at the path that "
            "does read the data repository"
        )

    def test_the_runners_point_submitters_at_the_fold_path(self):
        import subprocess

        for name in ("test_symbolic.py", "test_audio.py"):
            done = subprocess.run(
                [sys.executable, str(REPO_ROOT / "matchmaker_eval" / name), "--help"],
                cwd=REPO_ROOT, capture_output=True, text=True,
            )
            assert done.returncode == 0, done.stderr
            assert "--fold valid" in done.stdout
