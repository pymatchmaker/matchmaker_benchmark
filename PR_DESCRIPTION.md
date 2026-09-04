## What this is

The benchmark now reads matchmaker's method spec instead of keeping its own
copies of it, evaluates every leaderboard row on GitHub runners rather than on
whoever's laptop was free, and has a staging branch a run can land on before
anything is published.

## Changes

**Read the method spec, don't copy it.** `matchmaker_eval/methods.py` is the one
place that asks matchmaker what exists. Every runner's `--method` choices, sweep
defaults and reported sample/frame rate come from there, so a method added to
matchmaker appears here with no edit. `verify_equivalence.py` clones a method
straight from its spec, covering all eleven built-ins instead of three
hand-listed HMMs.

**Evaluate references on runners.** They had no CI path at all — `plan` derived
its target from changed submission directories, so a reference row could only be
made locally. `Evaluate references` fans out over every described method as a
(method × shard) matrix, mirroring `evaluate.yml` step for step;
`Evaluate submissions` also takes a `method`/`input_type` for a single one. Both
pin `OMP_NUM_THREADS=1` so the timing columns are comparable.

**Get the data by checkout.** The data repository's layout is the one
`MATCHMAKER_DATA_DIR` expects, so a shallow checkout of the branch a fold reads
replaces several hundred per-file downloads. `fetch_data.py --source` resolves
the branch from `data_sources.yaml`, `--verify` preflights a shard, and the
per-file fetch remains as a fallback.

**Two branches.** `submissions` evaluates and commits `results/` without
publishing; `main` publishes. A run is downloadable as the `leaderboard-site`
artifact before anyone sees it. `retract.py` withdraws a published result, with
a reason, and takes the per-piece detail file with it.

**One command for submitters.** `test_submission.py <dir>` fetches the tuning
fold, runs the follower and prints a verdict. It has no `--fold` option: not
providing a convenient way to develop against the eval fold is cheaper than
asking people not to. `test_audio.py` and `test_symbolic.py` are maintainer
tools over the metadata CSVs, and now say so.

**The parangonar trackers work** — develop's PR #57 fixes, expressed in the
method spec — and are described, so they can be leaderboard rows.

**Followers may be given the performance's tempo, visibly.** An entry opts in
with `estimated_bpm: true`; the value comes from the data repository's metadata
column of the same name; the run records it and the leaderboard marks the row
with an asterisk and a footnote. Off unless declared. `pfkorz` declares it.

**Licensing.** `resources/` ships an (n)ASAP score and a MAESTRO recording under
CC BY-NC-SA 4.0 with no attribution, inside an Apache-2.0 repository. Adds the
attribution and licence text and notes the exception in `LICENSE`.

## Defects fixed

| | |
| --- | --- |
| `latency_stats` demanded of every audio follower | it is not in the `OnlineAlignment` contract, so **no** audio submission could be scored |
| `merge_shards.py` dropped `kind` and `method` | every reference merged through CI was relabelled an anonymous submission |
| a dataset directory was trusted by its name | an upstream-shaped copy shadowed the fetch target; the cause of the 144/146 audio results |
| the CSV runners hardcoded `~/data/<upstream name>` | they ignored `MATCHMAKER_DATA_DIR` entirely, which is how they came to run against a directory nobody configured |
| `run_references.py` read a fixed metrics path | a non-eval run reported the published numbers as its own |
| `--fold <path>` crashed outside the repo | documented behaviour that raised at the last step |
| `run_references.py` printed nothing for hours | now a progress table, per-method logs, and `--watch` |
| `run_submission.py` broke on main's new tracking API | `check_tracking` lost three parameters; main could not fix a caller that does not exist there |

## Verification

- 199 tests, from none
- All ten MIDI references run with `data/`'s dataset material removed and
  `MATCHMAKER_DATA_DIR` empty — every file fetched from the data repository
- `baseline-constant-tempo-audio` evaluated on GitHub runners: 146/146 pieces,
  no failures, eight shards merged
- Every audio entry re-run from a fresh download of the data repository: 19/19
  pieces each, no failures — where the committed results had 8 to 122
- `verify_equivalence` reports identical metrics for every deterministic method,
  both input types, parangonar included

## Known issues, not fixed here

- **parangonar cannot yet make a leaderboard row.** The trackers work, but
  `note_array(include_grace_notes=True)` raises inside partitura 1.9.0 on scores
  loaded with `ignore_invisible_objects=True`, which leaves duplicate note ids
  after unfolding. 5 of 54 eval-fold scores are affected, covering 85 of 146
  pieces, so a run stays partial. Pre-existing; a fix touches note ids, which
  ground-truth matching also uses.
- **`pfkorz` is not reproducible.** It draws from a module-level `RandomState`,
  so a piece's result depends on its position in the shard — changing the shard
  count changes its numbers.
- The reference rows need regenerating on runners. The tracking rule changed in
  the main merge, so every existing number predates it.
