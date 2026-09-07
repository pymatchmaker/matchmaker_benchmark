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
        fields = dict(
            line.split("=", 1) for line in done.stdout.strip().splitlines()
        )
        assert fields["owner"] and fields["repo"] and fields["branch"]

    def test_source_output_is_shell_eval_safe(self):
        """The workflow does `eval "$(... --source)"`, so KEY=VALUE only."""
        done = run_eval_script("fetch_data.py", "--source", "--fold", "eval")
        for line in done.stdout.strip().splitlines():
            key, sep, value = line.partition("=")
            assert sep == "=", f"not a KEY=VALUE line: {line!r}"
            assert key.isidentifier(), f"not a shell-safe name: {key!r}"
            assert not set(value) & set(" \t'\"$`\;&|<>()"), (
                f"value needs quoting: {value!r}"
            )

    def test_each_fold_maps_to_its_configured_branch(self):
        config = yaml.safe_load(
            (REPO_ROOT / "data" / "data_sources.yaml").read_text()
        )
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


class TestPlanJobDependencies:
    """The plan job installs only pyyaml, and now also runs fetch_data.py."""

    #: Everything the plan job can rely on: the standard library, plus what it
    #: pip-installs. matchmaker, pandas and the rest are installed later, in
    #: the evaluate job.
    ALLOWED_THIRD_PARTY = {"yaml"}

    def _toplevel_imports(self, path: Path) -> set:
        import ast

        tree = ast.parse(path.read_text())
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
        return names

    def _is_stdlib(self, name: str) -> bool:
        return name in sys.stdlib_module_names

    @pytest.mark.parametrize("module", ["fetch_data.py", "folds.py"])
    def test_importable_with_only_pyyaml(self, module):
        path = REPO_ROOT / "matchmaker_eval" / module
        third_party = {
            name
            for name in self._toplevel_imports(path)
            if not self._is_stdlib(name) and name != "matchmaker_eval"
        }
        unexpected = third_party - self.ALLOWED_THIRD_PARTY
        assert not unexpected, (
            f"{module} imports {sorted(unexpected)} at module level, which the "
            "plan job does not install"
        )

    def test_plan_installs_what_its_steps_need(self):
        steps = steps_of(load_workflow(EVALUATE), "plan")
        body = "\n".join(step.get("run", "") for step in steps)
        for package in self.ALLOWED_THIRD_PARTY:
            name = {"yaml": "pyyaml"}[package]
            assert name in body, f"the plan job must pip install {name}"


class TestEvaluateWorkflow:
    def test_plan_publishes_the_data_repo_and_branch(self):
        outputs = load_workflow(EVALUATE)["jobs"]["plan"]["outputs"]
        assert "data_repo" in outputs
        assert "data_branch" in outputs

    def test_plan_reads_the_source_from_the_config(self):
        body = "\n".join(
            step.get("run", "") for step in steps_of(load_workflow(EVALUATE), "plan")
        )
        assert "fetch_data.py --source" in body, (
            "the workflow must read the data repository from data_sources.yaml, "
            "not hardcode it"
        )

    def test_the_data_repo_is_never_hardcoded_in_the_workflow(self):
        # The address may appear in a comment; it must not appear in a `with:`
        # block, which would bypass data_sources.yaml.
        workflow = load_workflow(EVALUATE)
        for job in workflow["jobs"].values():
            for step in job.get("steps", []):
                repo = (step.get("with") or {}).get("repository", "")
                if repo:
                    assert repo.startswith("${{"), (
                        f"step '{step.get('name')}' hardcodes repository {repo!r}"
                    )

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

    def test_the_shard_preflights_its_files(self):
        body = "\n".join(
            step.get("run", "")
            for step in steps_of(load_workflow(EVALUATE), "evaluate")
        )
        assert "--verify" in body, "the shard should preflight its files"
        assert "--shard" in body

    def test_the_fetch_fallback_is_wired_to_the_preflight(self):
        """Checkout is the fast path; a gap must still be fillable."""
        steps = steps_of(load_workflow(EVALUATE), "evaluate")
        by_name = {s.get("name"): s for s in steps}

        verify = by_name["Check this shard's files are all present"]
        assert verify.get("id") == "verify"
        assert verify.get("continue-on-error") is True, (
            "the preflight must not abort the job, or the fallback never runs"
        )

        fetch = by_name["Fetch anything the checkout did not provide"]
        assert fetch["if"] == "steps.verify.outcome == 'failure'"
        assert "fetch_data.py" in fetch["run"]

        confirm = by_name["Confirm the data is complete"]
        assert confirm["if"] == "steps.verify.outcome == 'failure'"
        assert "--verify" in confirm["run"], (
            "after fetching, completeness must be re-checked without "
            "continue-on-error so an incomplete shard fails"
        )
        assert confirm.get("continue-on-error") is not True

    def test_the_steps_are_in_the_right_order(self):
        names = [
            s.get("name") or s.get("uses")
            for s in steps_of(load_workflow(EVALUATE), "evaluate")
        ]
        order = [
            "Check out the benchmark data",
            "Point the benchmark at the data",
            "Check this shard's files are all present",
            "Fetch anything the checkout did not provide",
            "Confirm the data is complete",
            "Run the shard",
        ]
        positions = [names.index(name) for name in order]
        assert positions == sorted(positions), f"steps out of order: {names}"

    def test_the_run_step_pins_blas_threads(self):
        """Submissions and references must be timed the same way."""
        steps = steps_of(load_workflow(EVALUATE), "evaluate")
        run = next(s for s in steps if s.get("name") == "Run the shard")
        assert (run.get("env") or {}).get("OMP_NUM_THREADS") == "1"

    def test_data_dir_is_set_from_the_checkout(self):
        body = "\n".join(
            step.get("run", "")
            for step in steps_of(load_workflow(EVALUATE), "evaluate")
        )
        assert "MATCHMAKER_DATA_DIR=" in body
        assert "GITHUB_ENV" in body

    def test_merge_needs_no_data(self):
        """Merging works off the shard artifacts alone."""
        steps = steps_of(load_workflow(EVALUATE), "merge")
        for step in steps:
            assert not (step.get("with") or {}).get("repository"), (
                "the merge job should not need the data repository"
            )

    def test_every_referenced_script_exists(self):
        workflow = load_workflow(EVALUATE)
        body = "\n".join(
            step.get("run", "")
            for job in workflow["jobs"].values()
            for step in job.get("steps", [])
        )
        for token in body.split():
            if token.startswith("matchmaker_eval/") and token.endswith(".py"):
                assert (REPO_ROOT / token).exists(), f"{token} does not exist"


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

    def test_a_non_main_pin_is_marked_temporary(self):
        """So the revert is not forgotten once the branch is merged."""
        refs = {ref for refs in self._pins().values() for ref in refs}
        ref = refs.pop()
        if ref == "main":
            return
        for name in self._pins():
            text = (WORKFLOWS / name).read_text()
            assert "TEMPORARY" in text, (
                f"{name} pins matchmaker to '{ref}' with no note saying it is "
                "temporary"
            )


class TestSiteAssembly:
    """One script builds the site, so preview and publish cannot diverge."""

    def test_both_workflows_use_the_same_builder(self):
        pages = (WORKFLOWS / "pages.yml").read_text()
        evaluate = EVALUATE.read_text()
        assert "build_site.py" in pages
        assert "build_site.py" in evaluate

    def test_the_staging_run_uploads_a_viewable_site(self):
        steps = steps_of(load_workflow(EVALUATE), "merge")
        uploads = [
            s
            for s in steps
            if str(s.get("uses", "")).startswith("actions/upload-artifact")
            and (s.get("with") or {}).get("name") == "leaderboard-site"
        ]
        assert uploads, (
            "a run on the staging branch must leave the page downloadable, or "
            "there is no way to look at it before publishing"
        )

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

    def test_jobs_key_off_target_not_submission(self):
        """A reference has no submission directory, so the old guard skipped it."""
        workflow = load_workflow(EVALUATE)
        for job in ("evaluate", "merge"):
            condition = workflow["jobs"][job]["if"]
            assert "outputs.target" in condition, (
                f"job '{job}' still gates on submission, so a reference run "
                "would be skipped"
            )

    def test_plan_publishes_a_target(self):
        assert "target" in load_workflow(EVALUATE)["jobs"]["plan"]["outputs"]

    def test_the_target_is_passed_unquoted_to_the_runner(self):
        """It expands to either a directory or `--method X --input-type Y`."""
        steps = steps_of(load_workflow(EVALUATE), "evaluate")
        run = next(s["run"] for s in steps if s.get("name") == "Run the shard")
        assert "${{ needs.plan.outputs.target }}" in run
        assert '"${{ needs.plan.outputs.target }}"' not in run, (
            "quoting the target would pass '--method arzt --input-type audio' "
            "as a single argument"
        )

    def test_run_submission_accepts_both_target_shapes(self):
        """The two shapes plan can emit must both be valid CLI."""
        import subprocess

        for args in (
            ["submissions/baseline-constant-tempo", "--fold", "example"],
            ["--method", "arzt", "--input-type", "audio", "--fold", "example"],
        ):
            done = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "matchmaker_eval" / "run_submission.py"),
                    *args,
                    "--help",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            assert done.returncode == 0, done.stderr


class TestReferencesWorkflow:
    """References must be measurable on the same infrastructure as submissions."""

    REFERENCES = WORKFLOWS / "evaluate-references.yml"

    def test_the_workflow_exists(self):
        assert self.REFERENCES.exists(), (
            "references evaluated on a laptop are not comparable with "
            "submissions evaluated on a runner"
        )

    def test_it_shares_the_leaderboard_concurrency_group(self):
        """Two workflows committing results/ at once would race."""
        mine = load_workflow(self.REFERENCES)["concurrency"]["group"]
        theirs = load_workflow(EVALUATE)["concurrency"]["group"]
        assert mine == theirs

    def test_it_uses_the_same_data_route_as_evaluate(self):
        workflow = load_workflow(self.REFERENCES)
        steps = steps_of(workflow, "evaluate")
        checkout = [
            s for s in steps
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
        run = next(s for s in steps if s.get("name") == "Run the shard")
        env = run.get("env") or {}
        assert env.get("OMP_NUM_THREADS") == "1", (
            "reference timings are compared with each other; leaving the "
            "thread count to the runner makes them incomparable"
        )

    def test_the_matrix_is_built_from_described_methods(self):
        body = "\n".join(
            s.get("run", "") for s in steps_of(load_workflow(self.REFERENCES), "plan")
        )
        assert "builtin_methods.yaml" in body, (
            "the set of reference rows is data/builtin_methods.yaml, not a "
            "list hard-coded in the workflow"
        )

    def test_the_merge_regroups_before_merging(self):
        body = "\n".join(
            s.get("run", "") for s in steps_of(load_workflow(self.REFERENCES), "merge")
        )
        assert "merge_references.py" in body
        assert "leaderboard.py" in body
        assert "build_site.py" in body

    def test_it_uploads_a_reviewable_site(self):
        steps = steps_of(load_workflow(self.REFERENCES), "merge")
        names = [
            (s.get("with") or {}).get("name")
            for s in steps
            if str(s.get("uses", "")).startswith("actions/upload-artifact")
        ]
        assert "leaderboard-site" in names

    def test_shard_artifact_names_match_what_the_merger_parses(self):
        """The upload name and merge_references' regex are one contract."""
        import re
        import sys as _sys

        _sys.path.insert(0, str(REPO_ROOT / "matchmaker_eval"))
        from merge_references import SHARD_DIR

        steps = steps_of(load_workflow(self.REFERENCES), "evaluate")
        upload = next(
            s for s in steps
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

    def test_evaluation_still_runs_on_main(self):
        assert "main" in self._branches(EVALUATE)

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

        monkeypatch.setattr(
            "importlib.metadata.distribution", lambda name: FakeDist()
        )
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
