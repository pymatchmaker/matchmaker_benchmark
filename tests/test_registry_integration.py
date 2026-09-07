"""The benchmark and matchmaker must agree on what a method is.

Every method-specific fact the benchmark uses now comes from matchmaker's spec
(``matchmaker/methods.yaml``). These tests pin that seam: they fail when the two
repositories drift apart, which is the failure mode the old hardcoded tables
made silent.
"""

import subprocess
import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

warnings.filterwarnings("ignore", module="partitura")

from matchmaker.matchmaker import AVAILABLE_METHODS, DEFAULT_KWARGS
from matchmaker.registry import REGISTRY

from matchmaker_eval import methods as M
from matchmaker_eval.submission import (
    BUILTIN_METHODS_PATH,
    SubmissionError,
    builtin_metadata,
    read_descriptions,
    undescribed_methods,
)

INPUT_TYPES = ("audio", "midi")


class TestMethodsAdapter:
    @pytest.mark.parametrize("input_type", INPUT_TYPES)
    def test_available_methods_matches_matchmaker(self, input_type):
        assert M.available_methods(input_type) == AVAILABLE_METHODS[input_type]

    @pytest.mark.parametrize("input_type", INPUT_TYPES)
    def test_builtin_methods_are_a_subset_of_available(self, input_type):
        assert set(M.builtin_methods(input_type)) <= set(
            M.available_methods(input_type)
        )

    @pytest.mark.parametrize("input_type", INPUT_TYPES)
    def test_every_builtin_resolves_a_processor(self, input_type):
        for method in M.builtin_methods(input_type):
            processor = M.processor_for(input_type, method)
            assert processor in REGISTRY.processors[input_type], (
                f"{input_type}/{method} names processor '{processor}', "
                "which the spec does not declare"
            )

    @pytest.mark.parametrize("input_type", INPUT_TYPES)
    def test_default_kwargs_are_copies(self, input_type):
        for method in M.builtin_methods(input_type):
            kwargs = M.default_kwargs(input_type, method)
            kwargs["scribbled"] = True
            assert "scribbled" not in DEFAULT_KWARGS[input_type][method]

    def test_audio_rates_follow_matchmaker_defaults(self):
        assert M.audio_rates({}) == {
            "sample_rate": M.AUDIO_SAMPLE_RATE,
            "frame_rate": M.AUDIO_FRAME_RATE,
        }

    def test_hop_length_wins_over_frame_rate(self):
        # Matchmaker's own precedence: a config that sets both is resolved by
        # hop_length, so the reported frame rate must follow it.
        rates = M.audio_rates(
            {"sample_rate": 8000, "hop_length": 128, "frame_rate": 20}
        )
        assert rates == {"sample_rate": 8000, "frame_rate": 62.5}

    def test_skf_rates_match_its_spec(self):
        rates = M.audio_rates(M.default_kwargs("audio", "skf"))
        assert rates["sample_rate"] == 8000
        assert rates["frame_rate"] == pytest.approx(8000 / 128)

    def test_is_builtin_rejects_the_other_input_type(self):
        assert M.is_builtin("audio", "skf")
        assert not M.is_builtin("midi", "skf")


class TestBuiltinDescriptions:
    """``data/builtin_methods.yaml`` is prose only; it must not drift."""

    def test_file_is_valid_yaml(self):
        assert isinstance(read_descriptions(), dict)

    @pytest.mark.parametrize("input_type", INPUT_TYPES)
    def test_every_described_method_still_exists(self, input_type):
        described = set(read_descriptions().get(input_type) or {})
        available = set(M.builtin_methods(input_type))
        gone = sorted(described - available)
        assert not gone, (
            f"{BUILTIN_METHODS_PATH.name} describes {input_type} method(s) "
            f"{gone} that matchmaker no longer has"
        )

    @pytest.mark.parametrize("input_type", INPUT_TYPES)
    def test_described_methods_have_the_required_fields(self, input_type):
        for method, entry in (read_descriptions().get(input_type) or {}).items():
            metadata = builtin_metadata(method, input_type)
            assert metadata["name"]
            assert isinstance(metadata["authors"], list) and metadata["authors"]
            assert metadata["description"].strip()
            assert metadata["input_type"] == input_type
            assert metadata["kind"] == "reference"

    #: Prose, plus declarations about how the entry was evaluated. What must
    #: never appear here is matchmaker configuration -- how the follower is
    #: built is the method spec's business, and a second copy would drift.
    ALLOWED = {
        "name", "authors", "description", "url", "affiliation",
        "estimated_bpm",
    }
    FORBIDDEN = {
        "processor", "piano_range", "polling_period", "sample_rate",
        "frame_rate", "hop_length", "class", "args", "default_kwargs",
    }

    def test_descriptions_carry_no_configuration(self):
        for input_type, entries in read_descriptions().items():
            for method, entry in (entries or {}).items():
                extra = set(entry) - self.ALLOWED
                assert not extra, (
                    f"{input_type}/{method} sets {sorted(extra)} in "
                    f"{BUILTIN_METHODS_PATH.name}; only prose and evaluation "
                    "declarations belong here"
                )

    def test_matchmaker_configuration_never_appears_here(self):
        """How a follower is built comes from the spec, not from this file."""
        for input_type, entries in read_descriptions().items():
            for method, entry in (entries or {}).items():
                clash = set(entry) & self.FORBIDDEN
                assert not clash, (
                    f"{input_type}/{method} sets {sorted(clash)}, which would "
                    "be a second copy of matchmaker's method spec"
                )

    def test_unknown_method_is_rejected_before_the_yaml_is_consulted(self):
        with pytest.raises(SubmissionError, match="not a built-in midi method"):
            builtin_metadata("no-such-method", "midi")

    def test_a_method_of_the_other_input_type_is_rejected(self):
        with pytest.raises(SubmissionError, match="not a built-in midi method"):
            builtin_metadata("skf", "midi")

    def test_undescribed_methods_are_reported_not_raised(self):
        missing = undescribed_methods()
        assert isinstance(missing, dict)
        for input_type, names in missing.items():
            assert set(names) <= set(M.builtin_methods(input_type))


class TestEntryPointsUseTheRegistry:
    """The runners' --method choices must come from the registry, not a list."""

    def test_test_symbolic_offers_every_midi_method(self):
        import test_symbolic

        assert self._choices(test_symbolic.build_parser(), "--method") == (
            M.available_methods("midi")
        )

    def test_test_audio_offers_every_audio_method_plus_offline(self):
        import test_audio

        assert self._choices(test_audio.build_parser(), "--method") == (
            M.available_methods("audio") + ["offline"]
        )

    def test_infer_offers_every_audio_method(self):
        import infer

        assert self._choices(infer.build_parser(), "--method") == (
            M.available_methods("audio")
        )

    def test_a_registered_method_shows_up_in_the_choices(self):
        """A submission registers on import; the CLI must pick it up."""
        import test_symbolic
        from matchmaker import register_method
        from matchmaker.matchmaker import unregister_method

        register_method(
            "cli-probe", input_type="midi", build_follower=lambda mm: None
        )
        try:
            assert "cli-probe" in self._choices(
                test_symbolic.build_parser(), "--method"
            )
        finally:
            unregister_method("cli-probe", "midi")

    @staticmethod
    def _choices(parser, flag):
        for action in parser._actions:
            if flag in action.option_strings:
                return list(action.choices)
        raise AssertionError(f"{flag} not found")


class TestVerifyEquivalence:
    def test_it_can_clone_every_builtin(self):
        """register_clone must accept every method, not a hardcoded few."""
        from verify_equivalence import register_clone
        from matchmaker.matchmaker import unregister_method

        for input_type in INPUT_TYPES:
            for method in M.builtin_methods(input_type):
                name = register_clone(method, input_type)
                try:
                    assert name in AVAILABLE_METHODS[input_type]
                    assert DEFAULT_KWARGS[input_type][name] == M.default_kwargs(
                        input_type, method
                    )
                finally:
                    unregister_method(name, input_type)

    def test_method_choices_cover_both_input_types(self):
        import verify_equivalence

        assert verify_equivalence.is_builtin_for("skf", "audio")
        assert not verify_equivalence.is_builtin_for("skf", "midi")


class TestParangonarReferences:
    """The four parangonar trackers are leaderboard references now."""

    NAMES = ("OPTM", "OTM", "SLT_OLTW", "SL_OLTW")

    @pytest.mark.parametrize("method", NAMES)
    def test_each_is_described(self, method):
        metadata = builtin_metadata(method, "midi")
        assert metadata["name"]
        assert metadata["description"].strip()

    @pytest.mark.parametrize("method", NAMES)
    def test_each_builds_its_own_matcher_not_the_registry_key(self, method):
        """A clone registers under another name and must still be itself.

        `method` is a literal in the spec rather than {from: method}: taking it
        from the registry key made verify_equivalence's `clone-OPTM` blow up in
        parangonar with "Unknown parangonar method".
        """
        spec = REGISTRY.method("midi", method)
        arg = spec.args["method"]
        assert arg.source == "value"
        assert arg.key == method

    @pytest.mark.parametrize("method", NAMES)
    def test_each_runs_event_based(self, method):
        """ParangonarProcessor raises on a frame holding more than one note."""
        spec = REGISTRY.method("midi", method)
        assert spec.event_based
        assert spec.default_kwargs["polling_period"] is None
        assert spec.default_kwargs["processor"] == "parangonar"

    def test_the_processor_is_declared(self):
        assert "parangonar" in REGISTRY.processors["midi"]

    def test_they_are_no_longer_undescribed(self):
        assert "midi" not in undescribed_methods()


class TestFoldPaths:
    """``--fold`` also takes a path to a CSV, which need not be in the repo."""

    def test_fold_file_label_handles_an_external_csv(self, tmp_path):
        from matchmaker_eval.folds import REPO_ROOT
        from run_submission import fold_file_label

        external = tmp_path / "subset.csv"
        external.write_text(
            (REPO_ROOT / "data" / "folds" / "example.csv").read_text()
        )
        assert fold_file_label(external) == str(external)

    def test_fold_file_label_stays_relative_inside_the_repo(self):
        from run_submission import fold_file_label

        assert fold_file_label("example") == "data/folds/example.csv"


class TestNoStaleMethodTables:
    """No benchmark module may keep its own list of matchmaker's methods."""

    FORBIDDEN = ("HMM_CLASSES",)

    def test_hardcoded_class_tables_are_gone(self):
        from pathlib import Path

        eval_dir = Path(__file__).resolve().parent.parent / "matchmaker_eval"
        for path in eval_dir.glob("*.py"):
            text = path.read_text()
            for name in self.FORBIDDEN:
                assert name not in text, f"{path.name} still defines {name}"


class TestOptionalLatencyStats:
    """``latency_stats`` is not part of the OnlineAlignment contract.

    The built-in audio followers each maintain their own; a submission written
    to the documented base class has none. Evaluation must not require it, or
    no audio submission can be scored at all.
    """

    class _NoLatency:
        input_type = "audio"

        def get_latency_stats(self):
            raise AttributeError("'MyFollower' object has no attribute 'latency_stats'")

    class _WithLatency:
        input_type = "audio"

        def get_latency_stats(self):
            return {"f_avg_latency": 1.5, "i_avg_latency": 0.1}

    class _Midi:
        input_type = "midi"

        def get_latency_stats(self):  # pragma: no cover - must not be called
            raise AssertionError("MIDI runs report no latency")

    def test_a_follower_without_latency_is_still_evaluated(self):
        from eval import latency_stats

        assert latency_stats(self._NoLatency()) == {}

    def test_a_follower_with_latency_still_reports_it(self):
        from eval import latency_stats

        assert latency_stats(self._WithLatency()) == {
            "f_avg_latency": 1.5,
            "i_avg_latency": 0.1,
        }

    def test_midi_runs_are_left_alone(self):
        from eval import latency_stats

        assert latency_stats(self._Midi()) == {}

    def test_a_zero_frame_run_does_not_crash(self):
        from eval import latency_stats

        class ZeroFrames:
            input_type = "audio"

            def get_latency_stats(self):
                raise ZeroDivisionError("division by zero")

        assert latency_stats(ZeroFrames()) == {}


class TestBaselineSubmissions:
    """Both baselines must stay loadable — CI smoke-tests them on every PR."""

    @pytest.mark.parametrize(
        "directory,input_type",
        [
            ("baseline-constant-tempo", "midi"),
            ("baseline-constant-tempo-audio", "audio"),
        ],
    )
    def test_the_baseline_registers_what_its_metadata_declares(
        self, directory, input_type
    ):
        from pathlib import Path

        import yaml

        from matchmaker_eval.submission import load_solution
        from matchmaker.matchmaker import unregister_method

        root = Path(__file__).resolve().parent.parent / "submissions" / directory
        declared = yaml.safe_load((root / "metadata.yaml").read_text())
        assert declared["input_type"] == input_type

        method, registered_type = load_solution(root / "solution.py")
        try:
            assert method == directory, (
                "the registered name must match the directory name; "
                "validate_submission.py rejects a mismatch"
            )
            assert registered_type == input_type
        finally:
            unregister_method(method, registered_type)


class TestSweepSupport:
    """Both runners sweep the same way; sweep.py's separate path is gone."""

    def test_both_runners_have_a_sweep_mode(self):
        for name in ("test_audio.py", "test_symbolic.py"):
            done = subprocess.run(
                [sys.executable, str(REPO_ROOT / "matchmaker_eval" / name), "--help"],
                cwd=REPO_ROOT, capture_output=True, text=True,
            )
            assert done.returncode == 0, done.stderr
            assert "--sweep" in done.stdout, f"{name} cannot run as a sweep agent"

    def test_the_old_standalone_sweep_is_gone(self):
        assert not (REPO_ROOT / "matchmaker_eval" / "sweep.py").exists()

    def test_every_sweep_config_names_a_program_that_exists(self):
        import yaml as _yaml

        configs = sorted((REPO_ROOT / "sweep_config").glob("*.yaml"))
        assert configs
        for path in configs:
            config = _yaml.safe_load(path.read_text())
            program = config.get("program")
            assert program, f"{path.name} names no program"
            assert (REPO_ROOT / program).exists(), (
                f"{path.name} points at {program}, which does not exist"
            )

    def test_every_sweep_config_optimises_a_metric_that_is_logged(self):
        """`average.<10ms` came from the deleted script's own summary shape."""
        import yaml as _yaml

        logged = {"tracking_rate"}
        for path in sorted((REPO_ROOT / "sweep_config").glob("*.yaml")):
            config = _yaml.safe_load(path.read_text())
            name = (config.get("metric") or {}).get("name")
            assert name, f"{path.name} names no metric"
            root = name.split(".")[0]
            assert root in logged or name in logged, (
                f"{path.name} optimises '{name}', which nothing logs"
            )

    def test_a_class_valued_setting_arrives_as_a_class(self):
        """A sweep can only carry strings; followers want the class."""
        from matchmaker_eval.methods import resolve_class_values

        resolved = resolve_class_values({"tempo_model": "KalmanTempoModel"})
        assert not isinstance(resolved["tempo_model"], str)
        assert resolved["tempo_model"].__name__ == "KalmanTempoModel"

    def test_an_unknown_class_name_says_what_is_available(self):
        from matchmaker_eval.methods import resolve_class_values

        with pytest.raises(ValueError, match="unknown tempo_model"):
            resolve_class_values({"tempo_model": "NoSuchModel"})

    def test_sweep_only_keys_do_not_reach_the_follower(self):
        from matchmaker_eval.sweeps import sweep_kwargs

        kwargs = sweep_kwargs(
            "midi",
            "hmm",
            {"dataset": "valid", "method": "hmm", "input_type": "midi",
             "piano_range": False},
        )
        for key in ("dataset", "method", "input_type"):
            assert key not in kwargs
        assert kwargs["piano_range"] is False

    def test_both_runners_sweep_through_the_same_module(self):
        """The shared part exists once; only the entry point is per-runner."""
        for name in ("test_audio.py", "test_symbolic.py"):
            source = (REPO_ROOT / "matchmaker_eval" / name).read_text()
            assert "from sweeps import" in source, f"{name} rolls its own sweep"
            assert "def build_sweep_kwargs" not in source

    def test_the_example_dataset_needs_no_corpus(self):
        """The runners' smoke dataset is the piece committed under resources/."""
        import csv

        with open(REPO_ROOT / "data" / "metadata-example.csv") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        for row in rows:
            assert row["dataset"] == "local", (
                "the example dataset must resolve inside the repository"
            )
            for column in ("xml_score", "midi_performance", "match",
                           "audio_performance"):
                path = REPO_ROOT / "resources" / row[column]
                assert path.exists(), f"{column} -> {path} is not committed"

    def test_a_sweep_can_be_smoke_tested_without_data(self):
        """At least one config sweeps the piece committed under resources/,
        so the plumbing can be checked before a real sweep on `valid`."""
        import yaml as _yaml

        on_example = [
            path.name
            for path in sorted((REPO_ROOT / "sweep_config").glob("*.yaml"))
            if _yaml.safe_load(path.read_text())["parameters"]
            .get("dataset", {})
            .get("value")
            == "example"
        ]
        assert on_example, "no sweep config runs on the example piece"

    def test_the_sweep_entity_can_be_overridden(self, monkeypatch):
        """`wandb agent` sets WANDB_ENTITY; ours is only the fallback."""
        from matchmaker_eval.sweeps import sweep_entity

        monkeypatch.delenv("WANDB_ENTITY", raising=False)
        assert sweep_entity() == "matchmaker"
        monkeypatch.setenv("WANDB_ENTITY", "someone-else")
        assert sweep_entity() == "someone-else"

    def test_the_sweep_project_matches_the_configs(self, monkeypatch):
        from matchmaker_eval.sweeps import sweep_project

        monkeypatch.delenv("WANDB_PROJECT", raising=False)
        assert sweep_project("midi", "hmm") == "midi-hmm-sweep"
        assert sweep_project("audio", "arzt") == "audio-arzt-sweep"

    def test_the_agents_project_wins(self, monkeypatch):
        """A sweep whose runs land in another project is not tracking them."""
        from matchmaker_eval.sweeps import sweep_project

        monkeypatch.setenv("WANDB_PROJECT", "sweep-smoke-test")
        assert sweep_project("midi", "hmm") == "sweep-smoke-test"

    def test_every_sweep_config_declares_where_it_runs(self):
        """Left implicit, wandb invents a project name from the program path,
        and the sweep and its runs end up in two different places."""
        import yaml as _yaml

        for path in sorted((REPO_ROOT / "sweep_config").glob("*.yaml")):
            config = _yaml.safe_load(path.read_text())
            assert config.get("entity"), f"{path.name} declares no entity"
            assert config.get("project"), f"{path.name} declares no project"
