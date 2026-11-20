# Maintaining the benchmark

## At a glance

| When | Do | Release |
| --- | --- | --- |
| A new submission arrives | Review the PR and merge it into `submissions`; `Evaluate - Submission` runs by itself. Then run `Publish - Leaderboard` with that run ID. | minor, e.g. `v0.1.0` → `v0.2.0` |
| Only the site or docs changed | Merge into `main`. The site redeploys the latest release. | none |
| Scores change (matchmaker pin, scoring code, fold, data) | Run `Evaluate - Matchmaker` on `eval`, and `Evaluate - Submission` for every published submission. Publish all those run IDs together. | major, e.g. `v0.2.0` → `v1.0.0` |

- A release takes only the run IDs you give it; everything else is carried over from the latest release.
- After a score change, rerun everything: the leaderboard ranks only rows scored like the newest one.
- A matchmaker release does not require a new pin; move the pin only when its followers' results change.
- `evaluate.yml` runs from `submissions`, so fast-forward it to `main` after changing the pipeline.

## Reviewing a submission

Merging into `submissions` runs the submitted code straight away, so check
before merging:

1. Every `review` line in the `Check - Submission PR` log (ground truth, data directory, network).
2. `solution.py` uses only the streamed input and never opens the performance file.
3. `metadata.yaml` does not list `eval` under `tuned_on`.

## Evaluating

`Evaluate - Matchmaker` runs matchmaker's methods (`data/builtin_methods.yaml`)
on the chosen `fold`, split into parallel shards. Only `eval` reaches the
leaderboard; the others upload a `benchmark-results-<fold>` artifact.

| fold | what it runs |
| --- | --- |
| `eval` | the leaderboard, including the frozen-AMT followers in `data/amt_methods.yaml` |
| `beyond_piano` | the non-piano corpora (audio) |
| `diagnostics` | the diagnostic suites, methods in `data/diagnosis_methods.yaml` |
| `offline` | the offline reference |
| `delay` | audio methods on the valid fold at playback speed |

`Evaluate - Submission` evaluates the submissions a push to `submissions`
changed; dispatch it with a `submission` name to rerun one. Both workflows
share one concurrency group: one run at a time and one waiting, so a third
dispatch cancels the waiting one.

Artifacts expire after 30 days; publish before then. Sharding does not change
accuracy; timing columns depend on the runner.

Locally:

```bash
python matchmaker_eval/run_submission.py submissions/<name> --fold eval
python matchmaker_eval/run_submission.py --method pthmm --input-type midi --fold valid --limit 4
```

## Publishing

Run `Publish - Leaderboard` on `main` with:

- `source_runs`: the run IDs to add, separated by spaces. Later runs replace
  earlier entries of the same name.
- `release_tag`: the new version, `vMAJOR.MINOR.PATCH`. Leave both empty to
  redeploy the latest release.

Only successful runs are accepted. The release stores the run artifacts and a
`benchmark-snapshot.zip`, and the site is rebuilt from it.

### Older result versions

- **major**: a new evaluation key (scoring, data, fold or matchmaker version)
- **minor**: new entries under the same key
- **patch**: a fix to already published entries

Every release is served at `versions/<tag>/`; cite that address in a paper.
Only the newest releases and those in `PINNED_FULL_VERSIONS`
(`matchmaker_eval/build_site.py`) keep per-piece views.

### Withdrawing a result

Unpack the latest `benchmark-snapshot.zip`, point `MATCHMAKER_RESULTS_DIR` at
it, then:

```bash
python matchmaker_eval/retract.py <entry> --reason "..."
python matchmaker_eval/export_details.py
python matchmaker_eval/leaderboard.py
```

The reason is published in `retracted.json`; `--undo` restores it. Publish the
snapshot as a patch release.

## Data

`data/data_sources.yaml` names the data repository and which branch each fold
reads. Check it before a long run:

```bash
python matchmaker_eval/fetch_data.py --probe --fold eval --input-type audio
```

If the repository moves, edit `owner`, `repo` and `branch` there and probe
again. For a private repository, set the `MATCHMAKER_DATA_TOKEN` secret. A
self-hosted runner that already holds the data needs the repository variables
`BENCHMARK_RUNNER` and `MATCHMAKER_DATA_DIR`.

Changing `data/folds/eval.csv` invalidates every published number: rerun
everything and release a new major version.

## Keeping up with matchmaker

```bash
pytest tests/
python matchmaker_eval/verify_equivalence.py --method pthmm --input-type midi
```

The tests check that every method in `data/builtin_methods.yaml` still exists
in matchmaker; `verify_equivalence.py` checks that a submission is scored
exactly like a built-in method.
