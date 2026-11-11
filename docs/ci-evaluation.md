# Running the full benchmark on GitHub Actions

Open **Actions → Evaluate references → Run workflow** and select the code
branch to evaluate. Every shard and the merge use the same Matchmaker sources
from `.github/install_matchmaker.py`, including the 3 kHz SPARC implementation.
The first release under musical-beat TR must include
the complete piano evaluation; run the beyond-piano fold as well for the paper.
Full evaluations run only when explicitly dispatched. PR checks still validate
the code and bundled example without starting the full benchmark.

The installer pins `develop` at `25960ea`, which includes merged PRs #79, #80
and #81, including the Dannenberg initial-window correction. It retains the
existing bounded Multi-Fold enumeration (`678366b..0e673e2`) needed by diagnostics,
applied in a temporary CI checkout. No upstream branch or PR is changed.
Every result records the full base commit and patch commits and hashes in
`environment.matchmaker_base_commit` and `environment.matchmaker_patches`.
The full installed-source hash remains part of each run's provenance.
Once the remaining patch is merged upstream, update the installer's base and
remove it. Reviewed branch evaluations can be reused for publication as
described below.

| Run | input_type | fold | only | shards |
| --- | --- | --- | --- | --- |
| All piano references | both | eval | leave empty | 8 |
| All beyond-piano references | audio | beyond_piano | leave empty | 8 |
| MIDI OPHMM global-transition comparison | midi | ophmm_playthrough | ignored | fixed at 4 |
| Manuscript diagnostics | both | diagnostics | leave empty | 4 |
| Dannenberg perturbations only | midi | diagnostics | dannenberg | 4 |
| Offline reference, both folds | audio | offline | ignored | 8 |
| Frozen AMT reference | midi | amt_reference | ignored | 4 |

The piano fold contains 146 performances with audio and MIDI. The beyond-piano
fold contains 343 audio inputs: ChoraleBricks (191), KRAISLER (60), SWD (48), and
URMP (44). It has no performance MIDI files. Selecting `both` on that fold runs
only its available audio inputs; explicitly selecting MIDI fails during planning.

Both folds read the `eval` branch of `pymatchmaker/matchmaker-benchmark-data`,
as configured in `data/data_sources.yaml`. Each shard checks all required score,
performance, and ground-truth files before running.

## Downloading results

Once the merge finishes, download **benchmark-results-eval** or
**benchmark-results-beyond_piano** from the run's Artifacts section. This single
artifact contains every merged method directory, including `metrics.json`,
full `wp_<index>.tsv` and `gt_<index>.tsv` paths, and the score meter maps
`meter_<index>.json`. Individual `refshard-*` artifacts remain available for
diagnosing a missing or failed shard.

CI first merges into a fresh `results/ci/<fold>/` directory, so the combined
artifact contains only the current run. Piano entries also produce a
`leaderboard-site` preview of this run. Evaluation never commits result files
or updates the public website. Beyond-piano results stay in their own artifact.
A missing method or incomplete merge fails rather than reporting a complete evaluation.
For local merges without `--output`, auxiliary folds use `results/runs/<fold>/`.

## Preserving and publishing final results

1. Squash-merge the code PR into `main` without generated result changes.
2. Select successful **Evaluate references** runs for each required fold. Reuse
   reviewed pre-merge runs when their evaluation logic and data match the merged
   code. If a method changed, select its corrected rerun after the original run.
   Run missing evaluations on `main` only when needed.
3. Check the merged results and the preview. Then open **Publish leaderboard →
   Run workflow** on `main`. Set `source_runs` to the completed run IDs separated
   by spaces, in replacement order, and `release_tag` to a new version such as
   `benchmark-v1`. Later listed runs replace earlier entries for the same method.
4. The workflow preserves the results in that release, targets the publishing
   `main` commit, and deploys the website. It does not run any follower again.

The release contains the original `benchmark-results-*` artifact ZIPs, including
non-piano and auxiliary results when their run IDs are selected, a
`benchmark-snapshot.zip` for the website, and `SHA256SUMS`. The snapshot includes
`release.json` with source run IDs, evaluated commits, branches, attempts, artifact
hashes, the selected run for each leaderboard entry and the previous release.
Python, library, data and evaluation metadata remain in the original
result files. Raw paths stay in the run ZIPs; the website uses decimated paths.

Only successful, completed runs from this repository can be combined. Each
piano result must identify the commit of its source run. Publishing requires
`main`, but source runs may come from different commits or reviewed PR branches;
their original commit metadata is preserved, including after a squash merge.
The original ZIPs are all retained, including entries superseded in the snapshot.
Previously published methods are retained from the last release; methods under
older evaluation conditions remain outside the current ranking. The first
release must include the piano leaderboard. Later releases can also archive
non-piano or diagnostic results while retaining that leaderboard.

Result versions are never overwritten. Uploads are prepared as a draft and
published only after all assets have been uploaded. If an upload fails, inspect
the draft before retrying; the workflow refuses to overwrite an existing version.
Actions artifacts expire, so preserve final runs before that happens.

To redeploy an existing release, leave `source_runs` empty and optionally select
its `release_tag`. With no tag, the latest published `benchmark-*` snapshot is
used. Documentation-only updates reuse that snapshot. Before the first release,
the existing website stays unchanged. No result PR, additional branch or
additional repository is required. The old tracked `results/` files are retained
unchanged for this migration; new publication reads release assets instead.

## Adding or updating one method

After the first complete release, add the new method's description to
`data/builtin_methods.yaml` and update the pinned Matchmaker sources in
`.github/install_matchmaker.py`. Merge that code change, then run **Evaluate references**
on `main` with `only=dannenberg`, `input_type=midi` and `fold=eval`, for example.
For an audio method, run its beyond-piano fold too when needed.

Publish only those new run IDs in a new release. The website snapshot retains
the other methods and replaces the evaluated entries; the previous release
keeps their previous results. This needs no additional selection UI or result
PR. A follower-only change preserves comparison with earlier methods. A data,
fold, ground-truth or scoring change requires results under the new conditions
before they can be ranked together. See [evaluation keys](eval-protocol.md#the-evaluation-key).

## Evaluation settings

TR uses onset-anchored 30 s windows and tolerances of 1.0 musical beat for audio
and 0.1 for MIDI. The compound-meter factor is applied at each ground-truth onset,
including meter changes within a window. Position-error summaries remain in
denominator-note beats. See [the evaluation protocol](eval-protocol.md).

For a saved performance, the same TR calculation and plot can be reproduced with:

```sh
python matchmaker_eval/verify_tracking.py \
  --wp path/to/wp_1.tsv --gt path/to/gt_1.tsv \
  --meter path/to/meter_1.json --threshold 0.1 --save tracking_1.png
```

Use `--threshold 1.0` for audio. The meter map comes from the follower's actual
unfolded score, so it preserves the score-position system used by the saved paths.


## MIDI OPHMM play-through comparison

The `ophmm_playthrough` selection evaluates the same 146-piece piano fold with
score-wide transitions enabled and disabled. It calls the common benchmark
runner and saves both conditions in `ophmm-playthrough-comparison`. It does
not replace the default reference or publish either condition to the leaderboard.

```sh
python matchmaker_eval/run_submission.py --method outerhmm --input-type midi --fold eval \
  --global-transition off --output results/ophmm-off
```

Use `on` for the paired comparison. Disabling the option zeros the skip-from
probabilities before alignment; local transitions and pitch observations are
unchanged. The setting is saved in merged metadata and per-piece
records. The merge requires 146 completed performances per condition and checks
that the disabled transition probabilities remained zero.

The [diagnostic suites](diagnostics.md) retain their explicit repeat/skip
settings. Selecting a play-through setting does not change those experiments.
