"""Piece selection must preserve metadata IDs and report incomplete runs."""
from pathlib import Path

import pandas as pd
import pytest

import folds
import test_audio as runner
from utils import AudioEvalConfig


@pytest.fixture
def metadata(tmp_path, monkeypatch):
    # asap is read from the data repository's metadata (paths relative to it)
    (tmp_path / "asap").mkdir()
    pd.DataFrame([
        dict(audio=f"asap/{i}.mp3", score=f"asap/{i}.musicxml",
             match=f"asap/{i}.match", estimated_bpm=100)
        for i in range(1, 4)
    ]).to_csv(tmp_path / "asap" / "metadata-asap.csv", index=False)
    # the runner stops early when the first score is missing (a missing corpus)
    for i in range(1, 4):
        (tmp_path / "asap" / f"{i}.musicxml").touch()
    monkeypatch.setattr(folds, "DATA_ROOT", tmp_path)
    return tmp_path


def test_selected_rows_keep_metadata_ids(metadata, monkeypatch):
    calls = []

    def evaluate(score, audio, config, *args, **kwargs):
        calls.append((Path(score).name, kwargs["run_name"], kwargs["save_plots"],
                      kwargs["matchmaker_kwargs"]))
        return {"tracked": True}

    monkeypatch.setattr(runner, "run_score_following", evaluate)
    result = runner.run_tests_and_eval_by_dataset(
        "asap", AudioEvalConfig(method="arzt", dataset="asap"),
        metadata / "out", indices=[2, 3], save_plots=False,
        matchmaker_kwargs={"use_imm": False},
    )
    assert result["Index"] == [2, 3]
    assert calls == [(f"{i}.musicxml", str(i), False, {"use_imm": False})
                     for i in [2, 3]]


@pytest.mark.parametrize("indices,workers", [([0], 1), ([4], 1), ([1, 1], 1), (None, 0)])
def test_invalid_selection_fails(metadata, indices, workers):
    with pytest.raises(ValueError):
        runner.run_tests_and_eval_by_dataset(
            "asap", AudioEvalConfig(method="arzt", dataset="asap"),
            metadata / "out", indices=indices, workers=workers,
        )


def test_incomplete_run_is_not_summarized(metadata, monkeypatch):
    monkeypatch.setattr(runner, "run_score_following",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("bad audio")))
    args = runner.build_parser().parse_args([
        "--method", "arzt", "--dataset", "asap", "--no-plots",
        "--output-dir", str(metadata / "out"),
    ])
    with pytest.raises(RuntimeError, match="Incomplete evaluation: 0/3"):
        runner.main(args)
    assert not (metadata / "out" / "summary_all.json").exists()
