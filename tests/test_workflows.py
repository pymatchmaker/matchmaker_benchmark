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
