"""The wandb sweep concerns both benchmark runners share.

Sweeps are driven from the runners themselves (``test_symbolic.py --sweep``,
``test_audio.py --sweep``) rather than from a script of their own. A swept run
has to be the same code path as an ordinary one, or the sweep optimises
something the benchmark never measures -- which is how the old standalone
``sweep.py`` drifted: it grew its own paths, its own follower construction and
its own summary shape, and ended up tuning against numbers the leaderboard
does not report.

What *is* particular to sweeping -- reading settings out of the wandb config,
naming the project, logging the objective -- has no reason to exist twice, so
it lives here.
"""

import os

import wandb

from methods import default_kwargs, resolve_class_values

#: Keys a sweep config carries to steer the run, not to configure the follower.
RUNNER_KEYS = ("dataset", "method", "input_type")


def sweep_kwargs(input_type: str, method: str, wconfig) -> dict:
    """The method's defaults from matchmaker's spec, overridden by the sweep.

    A sweep config carries only strings and numbers, so a class-valued setting
    such as ``tempo_model`` arrives as a bare class name and is resolved here.
    """
    method_kwargs = default_kwargs(input_type, method)
    for key, value in wconfig.items():
        if key not in RUNNER_KEYS:
            method_kwargs[key] = value
    return resolve_class_values(method_kwargs)


def sweep_entity() -> str:
    """The wandb entity to log to.

    ``wandb agent`` sets ``WANDB_ENTITY`` from the sweep it is running, so
    honouring it lets someone outside our team run these sweeps in their own
    workspace. Ours remains the default for a run started by hand.
    """
    return os.environ.get("WANDB_ENTITY") or "matchmaker"


def sweep_project(input_type: str, method: str) -> str:
    """The wandb project to log to.

    ``wandb agent`` sets ``WANDB_PROJECT`` from the sweep it is running. It has
    to win: a sweep whose runs are created in some other project is not
    tracking them. The name below is only the fallback for a run started by
    hand, and is what the configs in sweep_config/ declare.
    """
    return os.environ.get("WANDB_PROJECT") or f"{input_type}-{method}-sweep"


def log_summary(summary_all: dict, summary_tracked: dict) -> None:
    """Report the run to wandb, with the leaderboard's metric as the objective.

    ``tracking_rate`` is logged at the top level because a sweep can only
    optimise a scalar; the full summaries go alongside it for inspection.
    """
    wandb.log(
        {
            "average": summary_all,
            "tracking_rate": summary_all.get("tracking_rate", 0),
        }
    )
    if summary_tracked.get("tracked_count", 0) > 0:
        wandb.log({"tracked_average": summary_tracked})
