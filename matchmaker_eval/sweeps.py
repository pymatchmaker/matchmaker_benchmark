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


def sweep_project(input_type: str, method: str) -> str:
    """One wandb project per method and input type, as the sweeps expect."""
    return f"{input_type}-{method}-sweep"


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
