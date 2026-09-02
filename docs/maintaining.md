# Maintaining the benchmark

For maintainers: reviewing submissions, running the evaluation, and the
decisions baked into this setup.

## Reviewing a submission

CI has already checked structure, metadata and that the follower runs on one
piece. **Review is the trust boundary**: after merge, the evaluation workflow
executes the submitted code on a machine that holds the datasets. Nothing
sandboxes it. Read the code.

`validate_submission.py` prints `review` lines for things worth a second look —
network access, subprocesses, `.match` file access, reaching for
`MATCHMAKER_DATA_DIR`, dynamic code execution. They are a reading aid, not a
verdict, and they are trivially avoidable by anyone acting in bad faith.

What to check by hand:

- **Is it online?** The follower must decide from what it has heard so far.
  Look for anything reading the performance file directly, or using
  `alignment_path` history in a way that implies future knowledge.
- **Does it touch ground truth?** Nothing under a dataset directory should be
  opened by a submission.
- **Are the dependencies reasonable?** Pinned, from PyPI, and plausible for the
  method described.
- **Does `metadata.yaml` match the code?** Especially `input_type` and the
  description of the method.
- **Is the runtime plausible?** A follower far slower than real time is
  interesting research but will take a long time on 146 pieces. The per-piece
  budget is 15 minutes.

When in doubt, ask in the pull request. Merging is what starts the evaluation.

## Running the evaluation

After merge, `.github/workflows/evaluate.yml` runs the eval fold on GitHub's own
runners and commits the leaderboard. Nobody has to hold the data locally.

```
plan ──► evaluate (8 parallel shards) ──► merge
  │            │                            │
  │       each shard:                  combines the shards,
which submission,   check out the data repo,      rebuilds the leaderboard,
which input type,   verify its slice is present,  commits results/
how many shards,    run it, upload results
where the data is
```

Sharding is not an optimisation, it is what makes this possible at all: a job on
a GitHub runner is capped at six hours, and audio evaluation over 146 pieces does
not fit in one. Shards are round-robin, so each gets a similar mix of datasets
and lengths, and `merge_shards.py` recomputes the pooled summary over all of
them — the merged numbers are identical to an unsharded run, which
`verify_equivalence.py` and the shard round-trip both check.

### Data

`data/data_sources.yaml` says where files come from. The data repository is laid
out exactly as `MATCHMAKER_DATA_DIR` expects, so **CI checks it out rather than
downloading files one at a time**: the `plan` job resolves the fold's branch with
`fetch_data.py --source`, each shard does one shallow `actions/checkout` of it,
points `MATCHMAKER_DATA_DIR` at that path, and runs `fetch_data.py --verify` to
fail fast if anything the fold names is absent. No cache is needed — a clone from
GitHub on a GitHub runner costs less than restoring one — and no token is needed
while the repository is public.

Set the repository variable `MATCHMAKER_DATA_DIR` to skip the checkout entirely
on a self-hosted runner that already holds the data.

`fetch_data.py` still fetches file by file, which is what a contributor who wants
one fold rather than a whole branch uses, and it remains the route for the
`url_template` escape hatch.

The address is four lines:

```yaml
owner: pymatchmaker
repo: matchmaker-benchmark-data
branch: main
layout: "{dataset}/{path}"
```

`layout` says where a fold's `<dataset>/<path>` sits inside that repository. With
the default, the repository root holds `asap/`, `batik/` and `vienna/`, and the
paths under them are exactly the ones in the fold CSVs:

```
asap/Bach/Fugue/bwv_858/VuV01M.match
batik/wav/k279_1.wav
vienna/audio/Chopin_Ballade/Chopin_op38_p01.wav
```

Check the configuration against the real repository before running anything
large — this makes a handful of HEAD requests and downloads nothing:

```bash
python matchmaker_eval/fetch_data.py --probe --fold eval --input-type audio
```

**Storage.** `storage: auto` tries `raw.githubusercontent.com` and then the Git
LFS host. The two are not interchangeable: a file committed through LFS and
fetched from `raw` comes back as a ~130-byte pointer stub rather than audio. The
fetcher detects that and says so instead of writing a corrupt file.

**Private repository.** Export a token with read access:

```bash
export MATCHMAKER_DATA_TOKEN=github_pat_...
```

In CI, add it as the repository secret `MATCHMAKER_DATA_TOKEN`; `evaluate.yml`
passes it to the fetch step. Once the data repository is public, delete the
secret and everything keeps working.

**Folds on separate branches.** Keep tuning material and evaluation material on
different branches of the data repository and map them:

```yaml
branch: main            # fallback for anything unlisted
branches:
  tuning: tuning-data
  eval: eval-data
```

Everything downstream follows automatically — `fetch_data.py`,
`run_submission.py`'s auto-fetch, `make_folds.py --from-repo` and the evaluation
workflow all resolve the branch from the fold they are working on. The workflow
reads it with `fetch_data.py --source`, so repointing a fold checks out the new
branch on the next run with no workflow edit. The locally cached metadata is
keyed by branch too, so two branches describing the same dataset cannot stand in
for each other.

**Two metadata shapes are supported.** A branch may keep one manifest inside
each dataset folder, or a single manifest at the repository root:

```
asap/metadata-asap.csv      audio,score,midi,match
                            audio/Bach_….mp3          <- dataset-relative

metadata-experiment.csv     dataset,audio,midi,score,match
                            batik,batik/audio/….mp3   <- root-relative
```

Per-dataset is tried first, the root manifest is the fallback
(`metadata-<fold>.csv`, then `metadata.csv`). Paths from either are normalised
to the dataset-relative form on read, so nothing downstream knows which was
used. That flexibility is the point of reading the repository's own metadata at
all: it can restructure without the benchmark needing an edit.

Check what a fold resolves to before running anything long:

```bash
python matchmaker_eval/fetch_data.py --probe --fold tuning --input-type midi
# repository : owner/repo@tuning-data
```

> **Branches organise the data, they do not restrict access.** On a public
> repository anyone can check out any branch. If the aim is that contributors
> genuinely cannot see the evaluation material, that needs a separate private
> repository — see the held-out fold discussion in
> [eval-protocol.md](eval-protocol.md).

**Somewhere other than GitHub.** Set `url_template` and the owner/repo/branch
fields are ignored:

```yaml
url_template: "https://huggingface.co/datasets/OWNER/NAME/resolve/main/{dataset}/{path}"
```

Audio containers are handled by the fetcher: fold CSVs name the original `.wav`
files, the data repository is expected to ship `.mp3`, and `audio_extensions`
lists what to try. Whatever it finds is what the runner loads.

For integrity, drop a `data/checksums/<dataset>.sha256` file (`sha256sum`
format, paths relative to the dataset root) and every fetched file is verified
against it.

### When the data repository moves

Renaming it, or moving it to another account, changes **four lines in one
file** and nothing else. No code, no workflow, no fold definition refers to the
repository by name.

1. Edit `owner`, `repo` and `branch` in `data/data_sources.yaml`.
2. Adjust `layout` if the new repository nests the datasets differently.
3. Run `python matchmaker_eval/fetch_data.py --probe --fold eval --input-type audio`.
4. If it went public, delete the `MATCHMAKER_DATA_TOKEN` secret.

Two things worth knowing. GitHub redirects the old URL after a rename, so
fetching keeps working for a while even if you forget — which means a stale
config can go unnoticed until the redirect lapses; the probe is how you check.
And already-downloaded files are keyed by path, not by source, so a rename does
not invalidate anyone's local copy or the CI cache.

### Runner choice

| Variable | Default | Meaning |
| --- | --- | --- |
| `BENCHMARK_RUNNER` | `ubuntu-latest` | Runner label for the shard jobs |
| `MATCHMAKER_DATA_DIR` | `~/data` | Where data is fetched to |

The same workflow works unchanged on a self-hosted runner that already holds the
datasets — the fetch step finds the files present and does nothing. Set
`BENCHMARK_RUNNER` to that runner's label if you would rather not re-fetch.

### Reference methods

The built-in methods that appear as `reference` rows are evaluated with one
command, which rebuilds the leaderboard when it finishes:

```bash
python matchmaker_eval/run_references.py --input-type midi
python matchmaker_eval/run_references.py --input-type audio
python matchmaker_eval/run_references.py --input-type both --exclude pf
```

Which methods *exist* comes from the installed matchmaker's spec
(`matchmaker/methods.yaml`); which of them this repo can *label* on the
leaderboard is `data/builtin_methods.yaml`, which holds names, citations and
descriptions and no configuration. `run_references.py` runs the described ones
and prints a note naming any matchmaker method that has no entry yet — add one
there to put it on the leaderboard. Each method runs in its own process
(`--jobs`, default 6) pinned to a single BLAS thread — measured about twice as
fast as letting BLAS thread freely, before counting the parallelism.

A method that fails, or that covers fewer than all the fold's pieces, is
reported and left off the leaderboard.

### Keeping up with matchmaker

`matchmaker_eval/methods.py` is the only place this repository reads
matchmaker's method spec, and `tests/` (plain `pytest`) checks that the two have
not drifted: that every described method still exists, that
`data/builtin_methods.yaml` carries no configuration, and that each runner's
`--method` choices come from the registry rather than a hardcoded list.

```bash
pytest tests/
python matchmaker_eval/verify_equivalence.py --method pthmm --input-type midi
```

`verify_equivalence.py` builds its clone from the spec itself, so it covers
every built-in method. A method the spec marks `deterministic: false` (a
particle filter) is reported inconclusive rather than failed — two runs of it
differ on their own.

### Staging a run before it is published

Evaluation runs on two branches; publishing runs on one.

| branch | evaluates | publishes to the site |
| --- | --- | --- |
| `submissions` | yes | **no** |
| `main` | yes | yes |

So the safe order for anything you are not sure about is:

1. merge the submission into `submissions`;
2. `evaluate.yml` runs there and commits `results/` to that branch;
3. read `results/leaderboard.json`, rerun or withdraw if it looks wrong;
4. merge `submissions` into `main`, which is what publishes.

`pages.yml` is deliberately `main`-only. Nothing on `submissions` reaches the
site, so a bad run is a branch to fix rather than a page to correct.

Note that merging `submissions` into `main` usually re-triggers evaluation on
`main`, because the merge brings the submission and code with it. That is a
re-verification on the branch that publishes, and it is cheap to let happen; a
commit that touches only `results/` does not trigger it.

### Withdrawing or replacing a published result

A leaderboard row is not a database record. It is derived from
`results/submissions/<entry>/metrics.json` every time `leaderboard.py` runs, so
results can always be corrected.

**To replace one** — a rerun after a bug fix — just evaluate again.
`run_submission.py` overwrites the same directory and the next rebuild picks up
the new numbers. Nothing needs deleting.

**To withdraw one**, use `retract.py` rather than deleting by hand: it also
removes the per-piece detail file, which the site fetches by name and which
would otherwise stay readable after the row disappeared.

```bash
python matchmaker_eval/retract.py --list        # what is published, what is not

python matchmaker_eval/retract.py pfkorz-midi \
    --reason "shared RNG makes runs irreproducible; see matchmaker#71"

python matchmaker_eval/export_details.py
python matchmaker_eval/leaderboard.py
```

The withdrawal is recorded in `results/retracted.json` with the reason and the
numbers that were published, and `leaderboard.json` carries a `retracted` list —
a result that was on the leaderboard and is not any more is itself a fact about
the leaderboard, so it is published rather than quietly erased. The run is moved
to `results/retracted/<entry>/`, so `--reason` is required and `--undo` puts it
back. `--purge` deletes instead of archiving, and cannot be undone.

`export_details.py` also removes any detail file whose run has gone, so a
hand-deleted result cannot leave one behind either.

### By hand

```bash
python matchmaker_eval/run_submission.py submissions/<name> --fold eval
python matchmaker_eval/leaderboard.py
```

Only `--fold eval` writes into `results/submissions/`, which is what the
leaderboard reads. Any other fold writes to `results/runs/<fold>/`, so a tuning
or smoke run can never overwrite a published record.

Or trigger the workflow manually (`workflow_dispatch`) with a submission name,
fold and shard count. Only complete `eval`-fold runs reach the leaderboard;
partial runs (a single shard, or `--limit`) are dropped automatically.

## Spot-checking for eval-fold tuning

The eval fold is public, so the declaration in `metadata.yaml` cannot be
verified at submission time — see
[eval-protocol.md](eval-protocol.md#why-this-is-on-the-honour-system). What
periodic checking looks like in practice:

```bash
python matchmaker_eval/run_submission.py submissions/<name> --fold tuning
```

Compare the tuning-fold and eval-fold numbers. A follower that does markedly
better on eval than on tuning is worth a conversation — but the folds differ in
size and difficulty, so this is a weak signal and never a verdict on its own.
Combine it with a reading of the code and, where a paper or repository exists,
with what it says about training data.

If a submission is removed, record why in its `metadata.yaml` under `notes` and
keep the directory, so the history stays readable.

## Sweeps and Weights & Biases

Sweeps (`sweep_config/`, `matchmaker_eval/sweep.py`) are how the built-in
methods are tuned. They run on the **tuning** fold and are kept in this
repository so the tuning procedure is public and reproducible.

The wandb workspace is not public. That is deliberate and it costs nothing:
sweeps log hyper-parameters and metrics for internal runs, and everything needed
to reproduce a sweep — the config, the code, the fold — is in the repository. A
contributor who wants to sweep points the same configs at their own workspace:

```bash
wandb sweep --entity <your-entity> --project <your-project> sweep_config/hmm-minimalistic.yaml
wandb agent <your-entity>/<your-project>/<sweep_id>
```

Sweeps and the leaderboard deliberately do not share a driver: sweeps optimise
on the tuning fold, `run_submission.py` measures on the eval fold. They do share
the evaluation functions in `eval.py` and `utils.py`, so a metric means the same
thing in both.

## One benchmark, two entry points

`test_audio.py` and `test_symbolic.py` evaluate the built-in methods;
`run_submission.py` evaluates community submissions. They are not two
benchmarks. A submission registers its follower through
`matchmaker.register_method()`, so by the time it is evaluated it is an ordinary
method name and every path ends in the same `Matchmaker(method=...)` call and
the same `run_evaluation` (in `eval.py`), `check_tracking` (in
`verify_tracking.py`) and `compute_event_pooled_summary` (in `utils.py`).

The only thing `run_submission.py` adds is the fold-based piece list and the
`metrics.json` record the leaderboard reads.

`verify_equivalence.py` proves this instead of asserting it: it re-registers a
built-in follower under a new name through the public registration API, runs
both, and compares every metric.

```bash
python matchmaker_eval/verify_equivalence.py --method pthmm --fold tuning --limit 4
```

Run it after touching the evaluation code.

## Changing the eval fold

Regenerating `data/folds/eval.csv` **invalidates every published number**. Each
`metrics.json` records the fold's SHA-256 for exactly this reason, so an
accidental change is detectable rather than silent.

If the fold has to change (a dataset is corrected, a piece is withdrawn):

1. Change `matchmaker_eval/make_folds.py`, regenerate, and commit the diff on
   its own, with the reason in the commit message.
2. Re-run every submission on the new fold.
3. Say so on the leaderboard page — a number from before the change is not
   comparable to one from after.

`make_folds.py --check` runs in CI on every pull request and fails if the
committed folds do not satisfy the disjointness property.

## Publishing

`.github/workflows/pages.yml` builds the site from `docs/` plus a copy of
`results/leaderboard.json` and `results/details/`, and deploys it to GitHub
Pages. It runs whenever the leaderboard changes.

### What the site can show

`results/leaderboard.json` is the index: one row per entry, small enough to load
instantly however many entries accumulate. Everything heavier lives in
`results/details/<entry>.json`, which the page fetches only when someone opens
that row:

* a per-dataset breakdown, pooled the same way the headline numbers are, so an
  asap number means what the number above it means;
* one row per piece, with its metrics and its tracked/lost verdict;
* the alignment path and the ground truth for every piece.

```bash
python matchmaker_eval/export_details.py
```

runs after every evaluation (`run_references.py` and the merge job both call
it). It only exports complete runs on the leaderboard fold — a partial run has
no published row for the detail to hang off.

**No plot images are stored.** The alignment pictures are drawn in the browser
from those coordinates, so re-styling them is a change to `docs/index.html` and
never a re-run of the benchmark.

That is affordable only because the paths are decimated. A full audio run holds
about 745,000 path points — roughly 15 MB of JSON per entry. At 300 points per
piece the shape survives at any size a browser will draw it, and an entry costs
about 1 MB, or 300 KB over the wire once Pages gzips it. `PATH_POINTS` in
`export_details.py` is the dial; `--no-paths` drops them entirely.

A Hugging Face Space is a reasonable second surface — sorting and filtering are
easier there than in a static page, and it reaches a different audience. It
needs nothing new from this repository: a Space can read
`results/leaderboard.json` straight from the raw GitHub URL, so it stays a
consumer of the same file and cannot drift from it. Add a push step here only if
a Space needs to be told to refresh.
