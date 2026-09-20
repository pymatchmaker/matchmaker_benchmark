import json

import numpy as np
import pytest

from scripts import evaluate_and_report_146 as report


def test_sparc_matches_authors_gaussian_example():
    times = np.arange(-1., 1., .01)
    speed = np.exp(-5 * times**2)
    assert report.compute_sparc(speed, 100.) == pytest.approx(-1.41403, abs=5e-6)
    assert report.compute_sparc(10 * speed, 100.) == pytest.approx(report.compute_sparc(speed, 100.))


def test_sparc_retains_mean_speed_and_has_no_stationary_sentinel():
    times = np.arange(1500) / 50.
    constant = report.compute_sparc(np.ones(len(times)), 50.)
    small = report.compute_sparc(1 + 1e-6 * np.sin(2 * np.pi * 3 * times), 50.)
    large = report.compute_sparc(1 + .6 * np.sin(2 * np.pi * 3 * times), 50.)
    assert small == pytest.approx(constant, abs=1e-5)
    assert large < small
    assert np.isnan(report.compute_sparc(np.zeros(len(times)), 50.))


def test_native_report_pools_events_and_uses_each_methods_tracked_sparc(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "DATASETS", [("example", 3)])
    monkeypatch.setattr(report, "compute_piece_sparc_30s", lambda path: float(path.stem.split("_")[1]))
    for method, flags, errors in [("a", [True, False, True], [0., .5, 2.]),
                                  ("b", [True, True, False], [.25, .5, 0.])]:
        run = tmp_path / method / "example"
        run.mkdir(parents=True)
        (run / "completion.json").write_text(json.dumps({"complete": True, "completed": 3}))
        for index, (tracked, error, count) in enumerate(zip(flags, errors, [2, 4, 3]), 1):
            (run / f"{index}.json").write_text(json.dumps({"tracked": tracked, "rtf": index / 10}))
            times = np.linspace(0, 1, count)
            np.savetxt(run / f"gt_{index}.tsv", np.column_stack((times, times)), delimiter="\t", header="perf_sec\tscore_beat")
            np.savetxt(run / f"wp_{index}.tsv", np.column_stack((times, times + error)), delimiter="\t", header="perf_sec\tscore_beat")
    result = report.evaluate_native(tmp_path, ["a", "b"])
    assert result["common_tracked_count"] == 1
    a, b = result["methods"]["a"], result["methods"]["b"]
    assert a["all_tracked"]["beat"]["mean"] == pytest.approx(1.2)
    assert a["all_tracked"]["beat"]["median"] == 2.
    assert a["common_tracked"]["beat"]["mean"] == 0.
    assert b["common_tracked"]["beat"]["mean"] == .25
    assert a["sparc"] == 2.
    assert b["sparc"] == 1.5
    assert a["rtf_all_pieces"] == pytest.approx(.2)
    (tmp_path / "a/example/completion.json").write_text(json.dumps({"complete": False, "completed": 2}))
    with pytest.raises(ValueError, match="Incomplete run"):
        report.evaluate_native(tmp_path, ["a", "b"])
