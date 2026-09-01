# Submitting a score follower

The benchmark takes submissions as pull requests. You add one directory, CI
checks it, a maintainer reviews it, and after merge it is evaluated and appears
on the leaderboard.

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
  │                 │                            │        runs the eval fold in
  │                 │                            │        parallel shards,
  │                 │                            │        commits the leaderboard
  │                 │                            │                    │
  └──────────────── leaderboard on GitHub Pages ◄─────────────────────┘
```

## 1. Set up

```bash
git clone https://github.com/pymatchmaker/matchmaker_benchmark.git
cd matchmaker_benchmark

conda env create -f environment.yml
conda activate matchmaker-benchmark
pip install "pymatchmaker @ git+https://github.com/pymatchmaker/matchmaker.git@main"
```

You do not need to download any dataset to write and test a submission — the
`example` fold is committed under `resources/`.

## 2. Write the follower

```bash
cp -r submissions/_template submissions/your-name
```

`solution.py` writes a score follower and registers it with matchmaker. That is
all the benchmark needs — there is no benchmark-specific base class, and the
same file works with plain matchmaker outside this repository:

```python
from matchmaker import register_method
from matchmaker.base import OnlineAlignment

class MyFollower(OnlineAlignment):
    def step(self, features) -> None:
        # one observation in; update self.current_index
        ...

register_method(
    "your-name",
    input_type="midi",
    build_follower=lambda mm: MyFollower(
        score_positions=mm.score_positions,
        queue=mm.stream.queue,
        reference_features=mm.reference_features,
    ),
    default_kwargs={"processor": "pitch", "piano_range": True},
)
```

Your follower is then built by the same `Matchmaker` call the built-in methods
use, and scored by the same evaluation code — no separate path, and nothing to
keep in sync.

The full contract — what `mm` carries, how to change the input features, how to
build audio reference features — is in
[submission-api.md](submission-api.md). For the follower itself, see
[HOW_TO_MAKE_CUSTOM_SCORE_FOLLOWERS.md](https://github.com/pymatchmaker/matchmaker/blob/main/HOW_TO_MAKE_CUSTOM_SCORE_FOLLOWERS.md)
in the matchmaker repository.

## 3. Fill in metadata.yaml

```yaml
name: My Score Follower
authors: [Your Name]
input_type: midi
description: >
  What it does and what is new about it.
eval_fold_untouched: true
```

`eval_fold_untouched` is a declaration, not a formality: it says the eval fold
played no part in developing, tuning, selecting or validating your follower.
Read [eval-protocol.md](eval-protocol.md#tuning-hygiene) before you set it —
the indirect forms of eval-fold tuning are easy to fall into.

Pin any extra dependencies in `requirements.txt`, exact versions.

## 4. Check it locally

```bash
python matchmaker_eval/validate_submission.py submissions/your-name --smoke
```

This is what CI runs on your pull request: structure, metadata, the contract,
and one piece end to end. It proves your follower *runs* — not that it is good.

### Do I need the datasets?

**No.** The smoke test above uses the piece committed under `resources/`, and
after merge the evaluation runs on GitHub's runners, which fetch the data
themselves. You can write, test and submit a follower without downloading
anything.

If you do want to measure yourself before submitting, the tuning fold is the
one to use. `fetch_data.py` pulls **only the files that fold needs** — never the
evaluation material:

```bash
export MATCHMAKER_DATA_DIR=~/data     # default
python matchmaker_eval/fetch_data.py --fold tuning --input-type midi
python matchmaker_eval/run_submission.py submissions/your-name --fold tuning
```

That is roughly 20 performances rather than 146, and `run_submission.py` fetches
what it needs on its own anyway, so the explicit fetch is optional. Add
`--input-type audio` for the recordings.

Do not run the eval fold yourself. There is nothing stopping you — the data is
public — but the point of the declaration is that you did not.

## 5. Open the pull request

One submission per pull request; CI rejects a pull request that touches two.
Fill in the checklist in the template.

CI runs on `pull_request`, so it has no repository secrets and a read-only
token — it cannot write to the leaderboard. If you are a first-time contributor,
GitHub will hold the workflow until a maintainer approves the run.

## 6. After merge

The evaluation workflow runs your submission on the eval fold and commits the
updated leaderboard. It splits the 146 pieces across parallel jobs, each of
which fetches only its own share of the data, so the whole thing finishes in
well under an hour even though a single job could not. The merged result is
identical to an unsharded run.

The full per-piece output (alignment paths, per-piece metrics) is attached to
the workflow run as an artifact for 30 days; the summary record stays in
`results/submissions/your-name/metrics.json`.

The leaderboard is published at
[pymatchmaker.github.io/matchmaker_benchmark](https://pymatchmaker.github.io/matchmaker_benchmark/).

## Updating a submission

Open a new pull request against the same directory. The evaluation re-runs and
the leaderboard row is replaced. If the change is substantial enough that it is
really a different method, submit it under a new name so both can be compared.

## Common problems

| Symptom | Cause |
| --- | --- |
| `registered no score follower` | `register_method(...)` must run at module level in `solution.py`, not inside a function or an `if __name__` block. |
| `registered N methods` | Call `register_method` exactly once per submission. |
| `'x' is a built-in midi method` | Pick a name that is not already a matchmaker method; your directory name is the right choice. |
| `returned a follower without a queue` | Pass `queue=mm.stream.queue` — the stream feeds the follower through it. |
| `...without score_positions` | Pass `score_positions=mm.score_positions`; the base class uses them to know when the score has ended. |
| `the follower produced an empty alignment path` | `step()` raised on the first observation, or `is_still_following()` was already false. |
| The run never finishes | `step()` is too slow, or `current_index` never advances so the follower never reaches the end of the score. Each piece has a 15-minute budget. |
