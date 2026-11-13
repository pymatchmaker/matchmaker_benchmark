"""The CI workflows must stay wired to commands and options that exist.

The evaluation workflow gets its data by checking out the data repository
rather than downloading files one at a time, and it reads *which* repository
and branch from ``data/data_sources.yaml`` via ``fetch_data.py --source``.
Those are three places that have to agree; these tests keep them agreeing.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
EVALUATE = WORKFLOWS / "evaluate.yml"
VALIDATE = WORKFLOWS / "validate-submission.yml"


def load_workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def steps_of(workflow: dict, job: str) -> list:
    return workflow["jobs"][job]["steps"]


def run_eval_script(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "matchmaker_eval" / args[0]), *args[1:]],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


class TestWorkflowsParse:
    @pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.yml")))
    def test_workflow_is_valid_yaml(self, path):
        assert isinstance(load_workflow(path), dict)


class TestDataSourceOptions:
    """Every fetch_data.py option the workflow calls must exist and work."""

    def test_source_reports_the_repository_and_branch(self):
        done = run_eval_script("fetch_data.py", "--source", "--fold", "eval")
        assert done.returncode == 0, done.stderr
        fields = dict(line.split("=", 1) for line in done.stdout.strip().splitlines())
        assert fields["owner"] and fields["repo"] and fields["branch"]

    def test_source_output_is_shell_eval_safe(self):
        """The workflow does `eval "$(... --source)"`, so KEY=VALUE only."""
        done = run_eval_script("fetch_data.py", "--source", "--fold", "eval")
        for line in done.stdout.strip().splitlines():
            key, sep, value = line.partition("=")
            assert sep == "=", f"not a KEY=VALUE line: {line!r}"
            assert key.isidentifier(), f"not a shell-safe name: {key!r}"
            assert not set(value) & set(" \t'\"$`\\;&|<>()"), (
                f"value needs quoting: {value!r}"
            )

    def test_each_fold_maps_to_its_configured_branch(self):
        config = yaml.safe_load((REPO_ROOT / "data" / "data_sources.yaml").read_text())
        for fold, branch in (config.get("branches") or {}).items():
            done = run_eval_script("fetch_data.py", "--source", "--fold", fold)
            fields = dict(
                line.split("=", 1) for line in done.stdout.strip().splitlines()
            )
            assert fields["branch"] == str(branch), (
                f"fold '{fold}' should read branch '{branch}'"
            )

    def test_verify_exists_and_reports_missing_data(self, tmp_path, monkeypatch):
        """--verify must fail loudly when the data is not there."""
        env_run = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "matchmaker_eval" / "fetch_data.py"),
                "--verify",
                "--fold",
                "eval",
                "--input-type",
                "midi",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "MATCHMAKER_DATA_DIR": str(tmp_path)},
        )
        assert env_run.returncode == 1
        assert "missing" in env_run.stdout


class TestEvaluateWorkflow:
    def test_evaluate_checks_out_the_data_repo(self):
        steps = steps_of(load_workflow(EVALUATE), "evaluate")
        checkout = [
            s
            for s in steps
            if str(s.get("uses", "")).startswith("actions/checkout")
            and (s.get("with") or {}).get("repository")
        ]
        assert len(checkout) == 1, "expected exactly one data-repository checkout"
        with_ = checkout[0]["with"]
        assert with_["path"], "the data must be checked out into its own path"
        assert with_.get("persist-credentials") is False

    def test_the_run_step_pins_blas_threads(self):
        """Submissions and references must be timed the same way."""
        steps = steps_of(load_workflow(EVALUATE), "evaluate")
        run = next(s for s in steps if "run_submission.py" in s.get("run", ""))
        assert (run.get("env") or {}).get("OMP_NUM_THREADS") == "1"


class TestMatchmakerPin:
    """All workflows must install the same matchmaker, or jobs disagree."""

    def _pins(self) -> dict:
        import re

        found = {}
        for path in sorted(WORKFLOWS.glob("*.yml")):
            for match in re.finditer(r"matchmaker\.git@([\w./-]+)", path.read_text()):
                found.setdefault(path.name, set()).add(match.group(1))
        return found

    def test_every_workflow_pins_the_same_ref(self):
        refs = {ref for refs in self._pins().values() for ref in refs}
        assert len(refs) == 1, (
            f"workflows install different matchmaker refs: {sorted(refs)}. "
            "A partial revert leaves the merge job on a different version "
            "from the shards."
        )


class TestSiteAssembly:
    """One script builds the site, so preview and publish cannot diverge."""

    def test_it_assembles_what_the_page_reads(self, tmp_path):
        import build_site

        summary = build_site.build(tmp_path / "site")
        site = tmp_path / "site"
        assert (site / "index.html").exists()
        assert (site / "leaderboard.json").exists()
        assert summary["details"] >= 0

    def test_it_refuses_to_build_without_a_leaderboard(self, tmp_path, monkeypatch):
        import build_site

        monkeypatch.setattr(build_site, "RESULTS_DIR", tmp_path / "empty")
        with pytest.raises(build_site.SiteError, match="leaderboard.json is missing"):
            build_site.build(tmp_path / "site")


class TestReferenceDispatch:
    """A built-in method must be evaluatable on CI, not only locally.

    References had no path through the workflow at all: `plan` derived the
    target from changed submission directories, so the only way to produce a
    reference row was to run it by hand. That is why the audio references sat
    stale and partial.
    """

    def _inputs(self):
        workflow = load_workflow(EVALUATE)
        trigger = workflow.get(True) or workflow.get("on")
        return trigger["workflow_dispatch"]["inputs"]

    def test_a_method_can_be_dispatched(self):
        inputs = self._inputs()
        assert "method" in inputs, "no way to evaluate a built-in as a reference"
        assert "input_type" in inputs, "--method needs an input type"

    def test_input_type_is_constrained(self):
        options = self._inputs()["input_type"]["options"]
        assert sorted(options) == ["audio", "midi"]


class TestReferencesWorkflow:
    """References must be measurable on the same infrastructure as submissions."""

    REFERENCES = WORKFLOWS / "evaluate-references.yml"

    def test_it_shares_the_leaderboard_concurrency_group(self):
        """Two workflows committing results/ at once would race."""
        mine = load_workflow(self.REFERENCES)["concurrency"]["group"]
        theirs = load_workflow(EVALUATE)["concurrency"]["group"]
        assert mine == theirs

    def test_it_uses_the_same_data_route_as_evaluate(self):
        workflow = load_workflow(self.REFERENCES)
        steps = steps_of(workflow, "evaluate")
        checkout = [
            s
            for s in steps
            if str(s.get("uses", "")).startswith("actions/checkout")
            and (s.get("with") or {}).get("repository")
        ]
        assert len(checkout) == 1
        assert checkout[0]["with"].get("persist-credentials") is False
        body = "\n".join(s.get("run", "") for s in steps)
        assert "--verify" in body
        assert "MATCHMAKER_DATA_DIR=" in body

    def test_the_run_step_pins_blas_threads(self):
        """A runner's core count must not move the timing columns."""
        steps = steps_of(load_workflow(self.REFERENCES), "evaluate")
        run = next(s for s in steps if "run_submission.py" in s.get("run", ""))
        env = run.get("env") or {}
        assert env.get("OMP_NUM_THREADS") == "1", (
            "reference timings are compared with each other; leaving the "
            "thread count to the runner makes them incomparable"
        )

    def test_shard_artifact_names_match_what_the_merger_parses(self):
        """The upload name and merge_references' regex are one contract."""
        import re
        import sys as _sys

        _sys.path.insert(0, str(REPO_ROOT / "matchmaker_eval"))
        from merge_references import SHARD_DIR

        steps = steps_of(load_workflow(self.REFERENCES), "evaluate")
        upload = next(
            s
            for s in steps
            if str(s.get("uses", "")).startswith("actions/upload-artifact")
        )
        template = upload["with"]["name"]
        concrete = re.sub(r"\$\{\{[^}]*method[^}]*\}\}", "arzt", template)
        concrete = re.sub(r"\$\{\{[^}]*input_type[^}]*\}\}", "audio", concrete)
        concrete = re.sub(r"\$\{\{[^}]*shard[^}]*\}\}", "3", concrete)
        match = SHARD_DIR.match(concrete)
        assert match, f"merge_references cannot parse artifact name {concrete!r}"
        assert match.group("entry") == "arzt-audio"
        assert match.group("shard") == "3"


class TestBundledDataLicence:
    """resources/ is CC BY-NC-SA data inside an Apache-2.0 repository."""

    def test_the_example_piece_is_attributed(self):
        attribution = REPO_ROOT / "resources" / "ATTRIBUTION.md"
        assert attribution.exists(), (
            "resources/ ships (n)ASAP and MAESTRO material; it needs attribution"
        )
        text = attribution.read_text()
        for required in ("CC BY-NC-SA", "asap-dataset", "maestro"):
            assert required.lower() in text.lower()

    def test_the_licence_text_is_included(self):
        assert (REPO_ROOT / "resources" / "LICENSE-CC-BY-NC-SA-4.0.md").exists()

    def test_the_root_licence_flags_the_exception(self):
        text = (REPO_ROOT / "LICENSE").read_text()
        assert "resources/" in text, (
            "the Apache licence must say it does not cover the bundled data"
        )


class TestStagingBranch:
    """Evaluation may run on a staging branch; publishing must not."""

    def _branches(self, path):
        workflow = load_workflow(path)
        trigger = workflow.get(True) or workflow.get("on")
        return trigger["push"]["branches"]

    def test_evaluation_runs_on_the_staging_branch(self):
        assert "submissions" in self._branches(EVALUATE)
        assert "main" not in self._branches(EVALUATE)

    def test_the_site_publishes_only_from_main(self):
        """A staging run must be inspectable before anyone sees it."""
        pages = WORKFLOWS / "pages.yml"
        assert self._branches(pages) == ["main"], (
            "publishing from a staging branch would defeat the point of having "
            "one: a result could reach the site before it was checked"
        )

    def test_a_results_only_commit_does_not_re_trigger_evaluation(self):
        """Committing results/ must not start another evaluation."""
        workflow = load_workflow(EVALUATE)
        trigger = workflow.get(True) or workflow.get("on")
        paths = trigger["push"]["paths"]
        assert not any(p.startswith("results") for p in paths)


class TestValidateWorkflow:
    def test_validation_needs_no_dataset(self):
        """The PR gate smoke-tests on the committed example fold."""
        workflow = load_workflow(VALIDATE)
        for step in steps_of(workflow, "validate"):
            assert not (step.get("with") or {}).get("repository")
        body = "\n".join(step.get("run", "") for step in steps_of(workflow, "validate"))
        assert "fetch_data.py" not in body

    def test_the_one_per_pull_request_rule_is_only_for_submission_prs(self):
        """The pull request that introduces the submission mechanism brings
        its baselines and worked example with it, so the limit cannot be
        unconditional."""
        workflow = load_workflow(VALIDATE)
        reject = [
            step
            for step in steps_of(workflow, "validate")
            if "github.base_ref" in step.get("if", "")
        ]
        assert reject, "the one-per-pull-request rule is gone"
        assert all(step["if"] == "github.base_ref == 'submissions'" for step in reject)

    def test_every_submission_is_validated_not_just_the_first(self):
        """Lifting the limit is only safe if all of them are still checked."""
        workflow = load_workflow(VALIDATE)
        body = "\n".join(step.get("run", "") for step in steps_of(workflow, "validate"))
        assert "validate_submission.py ${{ steps.changed.outputs.dirs }}" in body

    def test_the_example_fold_is_committed(self):
        """--smoke must run with nothing downloaded."""
        sys.path.insert(0, str(REPO_ROOT))
        sys.path.insert(0, str(REPO_ROOT / "matchmaker_eval"))
        from matchmaker_eval.folds import load_fold, missing_files

        pieces = load_fold("example", input_type="midi")
        assert pieces
        assert not missing_files(pieces, "midi")


class TestRunProvenance:
    """A published row should say which matchmaker produced it.

    The version string is "0.3.0" on every branch, so while the workflows
    install from a feature branch it cannot distinguish one run from another.
    """

    def test_the_environment_records_the_benchmark_commit(self):
        import sys as _sys

        _sys.path.insert(0, str(REPO_ROOT / "matchmaker_eval"))
        from run_submission import environment_info

        info = environment_info()
        assert info["benchmark_commit"] != "unknown"
        assert info["matchmaker"]

    def test_a_git_install_records_its_commit_and_ref(self, tmp_path, monkeypatch):
        import sys as _sys

        _sys.path.insert(0, str(REPO_ROOT / "matchmaker_eval"))
        import run_submission as R

        class FakeDist:
            def read_text(self, name):
                assert name == "direct_url.json"
                return (
                    '{"url": "https://github.com/pymatchmaker/matchmaker.git",'
                    ' "vcs_info": {"vcs": "git", "commit_id": "abc123",'
                    ' "requested_revision": "feature/clean_method_registration"}}'
                )

        monkeypatch.setattr("importlib.metadata.distribution", lambda name: FakeDist())
        found = R._matchmaker_source()
        assert found["matchmaker_commit"] == "abc123"
        assert found["matchmaker_ref"] == "feature/clean_method_registration"

    def test_a_release_install_adds_nothing(self, monkeypatch):
        import sys as _sys

        _sys.path.insert(0, str(REPO_ROOT / "matchmaker_eval"))
        import run_submission as R

        class NoDirectUrl:
            def read_text(self, name):
                return None

        monkeypatch.setattr(
            "importlib.metadata.distribution", lambda name: NoDirectUrl()
        )
        assert R._matchmaker_source() == {}
