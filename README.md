# Matchmaker Benchmark

An open leaderboard for **real-time score following**: how well can a machine
follow a live performer through a score?

**[→ Leaderboard](https://pymatchmaker.github.io/matchmaker_benchmark/)**

Write one Python file, open a pull request, and after review your score follower
is evaluated on 146 fixed performances and published. **You never download any
data** — the evaluation runs on GitHub's runners, which fetch what they need.

| | |
| --- | --- |
| **Evaluation set** | 146 performances (ASAP · Batik · Vienna 4x22), audio and MIDI |
| **Data** | [matchmaker-benchmark-data](https://github.com/pymatchmaker/matchmaker-benchmark-data), fetched on demand |
| **Library** | [matchmaker](https://github.com/pymatchmaker/matchmaker) |
| **Metrics** | beat error, ms error, tracking rate, real-time factor |

## Submit a score follower

```bash
git clone https://github.com/pymatchmaker/matchmaker_benchmark.git
cd matchmaker_benchmark
conda env create -f environment.yml && conda activate matchmaker-benchmark
pip install "pymatchmaker @ git+https://github.com/pymatchmaker/matchmaker.git@main"

cp -r submissions/_template submissions/your-name
```

Edit `solution.py`. The whole required surface is a follower and one call:

```python
from matchmaker import register_method
from matchmaker.base import OnlineAlignment

class MyFollower(OnlineAlignment):
    def step(self, features) -> None:
        # one note or audio frame in; update self.current_index
        ...

register_method(
    "your-name",
    input_type="midi",                       # or "audio"
    build_follower=lambda mm: MyFollower(
        score_positions=mm.score_positions,
        queue=mm.stream.queue,
        reference_features=mm.reference_features,
    ),
    default_kwargs={"processor": "pitch", "piano_range": True},
)
```

`register_method` is a plain matchmaker function, so your submission is an
ordinary matchmaker plugin — the same file works outside this repository.

Check it in about thirty seconds, with no dataset:

```bash
python matchmaker_eval/validate_submission.py submissions/your-name --smoke
```

Then open a pull request. CI runs that same check; a maintainer reviews the
code; after merge it is evaluated and appears on the leaderboard.

| Read next | |
| --- | --- |
| [docs/submitting.md](docs/submitting.md) | The full walkthrough |
| [docs/submission-api.md](docs/submission-api.md) | What `mm` carries, audio submissions, the contract |
| [docs/eval-protocol.md](docs/eval-protocol.md) | Folds, metrics, ranking, and what "do not tune on the eval fold" means |
| [docs/maintaining.md](docs/maintaining.md) | Review, runners, data sources, publishing |

## How it works

```
 you              GitHub                    maintainer          GitHub Actions
  │                 │                            │                    │
  │  pull request   │                            │                    │
  ├────────────────►│  structure + contract      │                    │
  │                 ├──► smoke test on the ──────┤                    │
  │                 │    committed example       │ reads the code     │
  │                 │    (no data needed)        │ and merges         │
  │                 │                            ├───────────────────►│
  │                 │                            │        fetches its own data,
  │                 │                            │        runs 146 pieces in
  │                 │                            │        parallel shards,
  │                 │                            │        commits the leaderboard
  │                 │                            │                    │
  └──────────────── leaderboard on GitHub Pages ◄─────────────────────┘
```

Review is the trust boundary: CI executes submitted code, it does not sandbox
it. Everything after merge is automatic.

## Folds

The benchmark is defined by frozen lists in `data/folds/`, committed so that any
change to what it measures arrives as a reviewable diff.

| Fold | Pieces | Data branch | Use |
| --- | ---: | --- | --- |
| `example` | 1 | committed in `resources/` | smoke tests; no download |
| `valid` | 20 | [`valid`](https://github.com/pymatchmaker/matchmaker-benchmark-data/tree/valid) | development and sweeps |
| `eval` | 146 | [`eval`](https://github.com/pymatchmaker/matchmaker-benchmark-data/tree/eval) | the leaderboard, and nothing else |
| `beyond_piano` | 343 | [`eval`](https://github.com/pymatchmaker/matchmaker-benchmark-data/tree/eval) | audio-only test corpora (ChoraleBricks, KRAISLER, URMP, Winterreise); not on the leaderboard |

No performance appears in both `valid` and `eval`; CI checks it on every pull
request (`make_folds.py --check`).

## Data

Audio and MIDI live in the public
[matchmaker-benchmark-data](https://github.com/pymatchmaker/matchmaker-benchmark-data)
repository, laid out exactly as the benchmark expects. There are two ways in,
and the address for both is four lines in
[data/data_sources.yaml](data/data_sources.yaml).

**Check it out.** The repository's layout *is* `MATCHMAKER_DATA_DIR`, so a
checkout of the branch a fold reads needs no further step. This is what CI does
— one shallow clone, no file-by-file downloads:

```bash
git clone --depth 1 --branch eval \
    https://github.com/pymatchmaker/matchmaker-benchmark-data.git ~/benchmark-data/eval
export MATCHMAKER_DATA_DIR=~/benchmark-data/eval

# confirm the fold is complete before running anything
python matchmaker_eval/fetch_data.py --verify --fold eval --input-type audio
```

`fetch_data.py --source --fold <fold>` prints which repository and branch that
fold reads, so you never have to look it up.

**Or fetch file by file.** For a single fold, or when a full branch is more than
you want, `fetch_data.py` downloads only the files a run touches:

```bash
# only the 20 validation performances, from the `valid` branch
python matchmaker_eval/fetch_data.py --fold valid --input-type midi

# check the configuration without downloading a corpus
python matchmaker_eval/fetch_data.py --probe --fold eval --input-type audio
```

`run_submission.py` fetches what it needs on its own, so even that is optional.
Files already under `MATCHMAKER_DATA_DIR` (default `~/data`) are used as-is, so
a local copy of the corpora still works.

## Running things

```bash
# one submission on a fold
python matchmaker_eval/run_submission.py submissions/<name> --fold valid

# the built-in methods, as leaderboard reference rows, then republish
python matchmaker_eval/run_references.py --input-type midi
python matchmaker_eval/run_references.py --input-type audio

# split a fold across parallel runs, then recombine
python matchmaker_eval/run_submission.py submissions/<name> \
    --shard 0 --num-shards 8 --output results/shards/shard-0
python matchmaker_eval/merge_shards.py results/shards

# rebuild the published ranking and the per-piece detail behind it
python matchmaker_eval/leaderboard.py
python matchmaker_eval/export_details.py
```

The original per-dataset runners for the built-in methods are unchanged:

```bash
python matchmaker_eval/test_audio.py    --dataset asap --method arzt
python matchmaker_eval/test_symbolic.py --dataset asap --method hmm
```

Submissions and built-in methods go through the *same* `Matchmaker` call and the
*same* evaluation functions — `run_evaluation`, `check_tracking`,
`compute_event_pooled_summary`. `verify_equivalence.py` proves it by running one
built-in method through both paths and comparing every metric.

### Where method knowledge lives

Which methods exist, which class each one is, which processor it runs with and
what its defaults are all come from matchmaker's own spec,
`matchmaker/methods.yaml` (see `HOW_TO_MAKE_CUSTOM_SCORE_FOLLOWERS.md` in the
matchmaker repository). This repository keeps no second copy of any of it:
`matchmaker_eval/methods.py` is the only place that asks, and every runner's
`--method` choices, sweep defaults and reported sample/frame rate come from
there. A method added to matchmaker therefore shows up in the runners with no
edit here.

The one method-related file this repo does own is
`data/builtin_methods.yaml`, and it holds **prose only** — the name, authors and
description shown on the leaderboard. A method matchmaker has that nobody has
described yet is reported by `run_references.py` and can still be evaluated with
`--method`; it just has no leaderboard label. `tests/` pins both halves of that
contract.

## Evaluation protocol

- **Primary metric**: tracking rate — the share of pieces followed to the end.
- **Tracking decision**: onset-wise forward windows of 30 seconds, evaluated
  from the first annotated performance onset through the first window that
  reaches the final onset. A window's statistic is the median absolute beat
  error, and a piece fails when any evaluated window exceeds the modality
  threshold — 1.0 beat for audio, 0.5 for MIDI.
- **Precision metric**: beat error (performance → score), pooled over the
  pieces that method actually tracked.
- **Secondary precision metric**: ms error (score → performance), same pooling.
- **Aggregation**: precision metrics are event-pooled across tracked pieces.
- **Ranking**: tracking rate first, then median beat error.

Full detail, including fold hygiene and the anti-tuning policy, in
[docs/eval-protocol.md](docs/eval-protocol.md).

## Project structure

```
matchmaker_eval/
  eval.py                 — single-piece alignment functions (audio + symbolic)
  test_audio.py           — audio runner for the built-in methods
  test_symbolic.py        — symbolic runner for the built-in methods
  submission.py           — loads submissions/<name>/ and validates metadata
  run_submission.py       — evaluate one submission on one fold -> metrics.json
  run_references.py       — evaluate every built-in method, then republish
  validate_submission.py  — pull-request checks: structure, contract, smoke run
  merge_shards.py         — recombine parallel shards into one metrics record
  leaderboard.py          — metrics.json files -> leaderboard.json / .csv,
                            plus the same ranked per dataset
  test_submission.py      — what a submitter runs: their follower, validation fold
  retract.py              — withdraw a published result, with a reason
  merge_references.py     — regroup and merge a sharded multi-method run
  export_details.py       — every pooled metric per dataset, per-piece
                            detail and decimated alignment paths
  folds.py, make_folds.py — fold definitions and their integrity checks
  fetch_data.py           — locate a fold's data (--source), check it is all
                            present (--verify), or download just what it needs
  verify_equivalence.py   — proves submissions are scored like built-in methods
  methods.py              — the one seam onto matchmaker's method spec
  utils.py, verify_tracking.py, sweep.py
data/
  folds/                  — frozen fold definitions (example, valid, eval)
  data_sources.yaml       — where performance data is fetched from
  builtin_methods.yaml    — leaderboard prose for the reference methods
                            (names and citations only; no configuration)
submissions/
  _template/              — copy-me skeleton
  baseline-constant-tempo/  — the no-information floor
  example-pitch-matcher/    — a worked example that actually listens
tests/                    — registry/benchmark consistency checks (pytest)
resources/                — the one example performance, CC BY-NC-SA 4.0
                            (see resources/ATTRIBUTION.md — not Apache-2.0)
results/
  leaderboard.json/.csv   — the published ranking, with per-dataset columns
  leaderboard-<ds>.json/.csv — the same entries ranked on one dataset
  retracted.json          — results withdrawn from it, and why
  details/<name>.json     — per-dataset, per-piece and alignment-path detail
  submissions/<name>/     — per-submission metrics.json
docs/                     — contributor and maintainer documentation, and the
                            GitHub Pages leaderboard
.github/workflows/
  validate-submission.yml — pull-request checks (no secrets, read-only token)
  evaluate.yml            — post-merge evaluation in parallel shards
  pages.yml               — publishes the leaderboard
```

## Acknowledgments

This work has been supported by the Austrian Science Fund (FWF), grant agreement PAT 8820923 ("Rach3: A Computational Approach to Study Piano Rehearsals"). Additionally, this work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. NRF-2023R1A2C3007605).
