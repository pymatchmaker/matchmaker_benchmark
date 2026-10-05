"""Loading and validating a benchmark submission.

A submission is one directory under ``submissions/``::

    submissions/alice/
        solution.py       registers a score follower (required)
        metadata.yaml     who wrote it, and how it was tuned (required)
        requirements.txt  extra pip dependencies (optional)

``solution.py`` calls ``matchmaker.register_method()`` exactly once. That is a
plain matchmaker call, not a benchmark concept: the same file works outside this
repository, and the benchmark runs the follower through ``Matchmaker`` with no
adapter in between.

    from matchmaker import register_method
    from matchmaker.base import OnlineAlignment

    class MyFollower(OnlineAlignment):
        def step(self, features):
            ...

    register_method(
        "alice",
        input_type="midi",
        build_follower=lambda mm: MyFollower(
            reference_features=mm.reference_features,
            score_positions=mm.score_positions,
            queue=mm.stream.queue,
        ),
    )

This module does the two things matchmaker has no reason to care about: reading
``metadata.yaml``, and importing a submitted file safely enough to find out what
it registered.

See ``docs/submission-api.md``, and
``HOW_TO_MAKE_CUSTOM_SCORE_FOLLOWERS.md`` in the matchmaker repository for the
follower itself.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from matchmaker.matchmaker import CUSTOM_METHODS

from matchmaker_eval.methods import builtin_methods

INPUT_TYPES = ("audio", "midi")
REQUIRED_METADATA = ("name", "authors", "input_type", "description")
DECLARATION_KEY = "eval_fold_untouched"

#: Opt-in: the follower is given the performance's estimated tempo. Off unless a
#: submission says otherwise, because the default position is that a tracker
#: knows nothing about the performance it is about to hear. A run that used it
#: is marked on the leaderboard, so the two are never compared silently.
ESTIMATED_BPM_KEY = "estimated_bpm"

BUILTIN_METHODS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "builtin_methods.yaml"
)


class SubmissionError(Exception):
    """Raised when a submission cannot be loaded, or breaks the contract."""


#: solution.py path -> (method, input_type). Registration is an import side
#: effect and matchmaker rejects a duplicate name, so a file must be executed
#: at most once per process — validate --smoke loads the same submission twice.
_LOADED = {}


def read_metadata(directory: Path) -> dict:
    """Read and validate ``metadata.yaml`` of a submission directory."""
    path = Path(directory) / "metadata.yaml"
    if not path.exists():
        raise SubmissionError(f"{path} is missing.")
    try:
        metadata = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        raise SubmissionError(f"{path} is not valid YAML: {e}") from e
    if not isinstance(metadata, dict):
        raise SubmissionError(f"{path} must contain a YAML mapping.")

    missing = [k for k in REQUIRED_METADATA if not metadata.get(k)]
    if missing:
        raise SubmissionError(f"{path} is missing required key(s): {missing}")
    if metadata["input_type"] not in INPUT_TYPES:
        raise SubmissionError(
            f"{path}: input_type must be one of {INPUT_TYPES}, "
            f"got '{metadata['input_type']}'."
        )
    if metadata.get(DECLARATION_KEY) is not True:
        raise SubmissionError(
            f"{path}: '{DECLARATION_KEY}: true' is required. It declares that the "
            "eval fold was not used to develop or tune this submission. See "
            "docs/eval-protocol.md."
        )
    authors = metadata["authors"]
    if isinstance(authors, str) or not isinstance(authors, (list, tuple)):
        raise SubmissionError(f"{path}: 'authors' must be a list.")
    declared = metadata.get(ESTIMATED_BPM_KEY, False)
    if not isinstance(declared, bool):
        raise SubmissionError(
            f"{path}: '{ESTIMATED_BPM_KEY}' must be true or false, got "
            f"{declared!r}. It declares whether the follower is given the "
            "performance's tempo; leave it out unless it is."
        )
    metadata[ESTIMATED_BPM_KEY] = declared
    return metadata


def load_solution(path: Path, module_name: Optional[str] = None) -> Tuple[str, str]:
    """Import ``solution.py`` and report the method it registered.

    Returns ``(method_name, input_type)``. Registration happens as an import
    side effect, so the registry is diffed around the import rather than
    trusting the file to say what it did.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise SubmissionError(f"{path} is missing.")
    if path in _LOADED:
        return _LOADED[path]

    before = set(CUSTOM_METHODS)
    module_name = module_name or f"matchmaker_submission_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SubmissionError(f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise SubmissionError(
            f"Importing {path} failed: {type(e).__name__}: {e}"
        ) from e

    registered = sorted(set(CUSTOM_METHODS) - before)
    if not registered:
        raise SubmissionError(
            f"{path} registered no score follower. Call "
            "matchmaker.register_method(name, input_type=..., build_follower=...) "
            "at module level."
        )
    if len(registered) > 1:
        names = [name for _, name in registered]
        raise SubmissionError(
            f"{path} registered {len(registered)} methods ({names}). "
            "A submission must register exactly one."
        )

    input_type, method = registered[0]
    _LOADED[path] = (method, input_type)
    return method, input_type


def load_submission(directory) -> Tuple[str, str, dict]:
    """Load a submission directory.

    Returns ``(method_name, input_type, metadata)``. After this call the
    follower is registered, so ``Matchmaker(method=method_name)`` builds it.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise SubmissionError(f"{directory} is not a directory.")
    metadata = read_metadata(directory)
    method, input_type = load_solution(directory / "solution.py")
    if input_type != metadata["input_type"]:
        raise SubmissionError(
            f"{directory}: metadata.yaml says input_type "
            f"'{metadata['input_type']}' but solution.py registered "
            f"'{input_type}'."
        )
    return method, input_type, metadata


def read_descriptions() -> dict:
    """``data/builtin_methods.yaml``: the prose for matchmaker's own methods.

    Only names, authors and descriptions live here. What a method *is* — its
    class, processor and defaults — comes from matchmaker's spec, so this file
    holds no configuration that could contradict a run.
    """
    return yaml.safe_load(BUILTIN_METHODS_PATH.read_text()) or {}


def undescribed_methods() -> Dict[str, List[str]]:
    """Built-in methods matchmaker has that this repo has no description for.

    They can still be evaluated with ``--method``; they just cannot be labelled
    on the leaderboard until someone writes them up. Reported rather than
    raised, so a new matchmaker method never breaks an evaluation run.
    """
    described = read_descriptions()
    missing = {}
    for input_type in ("audio", "midi"):
        known = set(described.get(input_type) or {})
        absent = [m for m in builtin_methods(input_type) if m not in known]
        if absent:
            missing[input_type] = absent
    return missing


def builtin_metadata(method: str, input_type: str) -> dict:
    """Metadata for one of matchmaker's own methods.

    Built-in methods are not submissions — they have no directory and nothing
    to declare — but they belong on the leaderboard as the reference points a
    new follower is trying to beat. This synthesises the same metadata shape
    from ``data/builtin_methods.yaml``.
    """
    available = builtin_methods(input_type)
    if method not in available:
        raise SubmissionError(
            f"'{method}' is not a built-in {input_type} method of the installed "
            f"matchmaker. Available: {available}. (A submission is run by "
            "passing its directory, not --method.)"
        )
    described = read_descriptions()
    entry = (described.get(input_type) or {}).get(method)
    if entry is None:
        raise SubmissionError(
            f"'{method}' is a matchmaker {input_type} method but has no entry in "
            f"{BUILTIN_METHODS_PATH.name}. Add a name, authors and description "
            "there to put it on the leaderboard."
        )
    return {
        "name": entry.get("name", method),
        "authors": entry.get("authors", ["matchmaker"]),
        "input_type": input_type,
        "description": entry.get("description", "").strip(),
        ESTIMATED_BPM_KEY: bool(entry.get(ESTIMATED_BPM_KEY, False)),
        "url": "https://github.com/pymatchmaker/matchmaker",
        # A built-in cannot have been tuned on the eval fold by a submitter;
        # the key is kept so every metrics record has the same shape.
        DECLARATION_KEY: True,
        "kind": "reference",
    }
