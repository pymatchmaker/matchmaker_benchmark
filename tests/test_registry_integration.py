"""The benchmark and matchmaker must agree on what a method is.

Every method-specific fact the benchmark uses now comes from matchmaker's spec
(``matchmaker/methods.yaml``). These tests pin that seam: they fail when the two
repositories drift apart, which is the failure mode the old hardcoded tables
made silent.
"""

import warnings

import pytest

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

    def test_descriptions_carry_no_configuration(self):
        """Only prose lives here — configuration belongs to matchmaker's spec."""
        allowed = {"name", "authors", "description", "url", "affiliation"}
        for input_type, entries in read_descriptions().items():
            for method, entry in (entries or {}).items():
                extra = set(entry) - allowed
                assert not extra, (
                    f"{input_type}/{method} sets {sorted(extra)} in "
                    f"{BUILTIN_METHODS_PATH.name}; configuration must come from "
                    "matchmaker's spec, not from this file"
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
