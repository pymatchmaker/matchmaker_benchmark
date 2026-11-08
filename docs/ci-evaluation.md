# Running the full benchmark on GitHub Actions

Open **Actions → Evaluate references → Run workflow** and select the code
branch to evaluate. Every shard and the merge use the same pinned Matchmaker
commit, including the 3 kHz SPARC implementation.

| Run | input_type | fold | only | shards |
| --- | --- | --- | --- | --- |
| All piano references | both | eval | leave empty | 8 |
| All beyond-piano references | audio | beyond_piano | leave empty | 8 |
| MIDI OPHMM global-transition comparison | midi | ophmm_playthrough | ignored | fixed at 4 |
| Manuscript diagnostics | both | diagnostics | ignored | 8 |

The piano fold contains 146 performances with audio and MIDI. The beyond-piano
fold contains 343 audio inputs: ChoraleBricks (191), KRAISLER (60), SWD (48), and
URMP (44). It has no performance MIDI files. Selecting `both` on that fold runs
only its available audio inputs; explicitly selecting MIDI fails during planning.

Both folds read the `eval` branch of `pymatchmaker/matchmaker-benchmark-data`,
as configured in `data/data_sources.yaml`. Each shard checks all required score,
performance, and ground-truth files before running.

The runner preserves the source files. If an XML score contains integer voice
identifiers outside Partitura's 32-bit range, it assigns unique small identifiers
in a temporary score copy before loading it. This preserves voice membership,
notes, and timing, and prevents NumPy 2 from rejecting oversized export labels.
Affected piece records include `score_voice_id_remapping`; the normalization
code is included in the evaluation key.

## Downloading results

Once the merge finishes, download **benchmark-results-eval** or
**benchmark-results-beyond_piano** from the run's Artifacts section. This single
artifact contains every merged method directory, including `metrics.json`,
full `wp_<index>.tsv` and `gt_<index>.tsv` paths, and the score meter maps
`meter_<index>.json`. Individual `refshard-*` artifacts remain available for
diagnosing a missing or failed shard.

CI first merges into a fresh `results/ci/<fold>/` directory, so the combined
artifact contains only the current run. Completed piano entries are then copied
to `results/submissions/` for the leaderboard. Beyond-piano results stay in their
artifact and do not overwrite piano records or rebuild the piano leaderboard.
A missing method or incomplete merge fails rather than reporting a complete evaluation.
For local merges without `--output`, auxiliary folds use `results/runs/<fold>/`.

## Evaluation settings

TR uses onset-anchored 30 s windows and tolerances of 1.0 musical beat for audio
and 0.1 for MIDI. The compound-meter factor is applied at each reference onset,
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
score-wide transitions enabled and disabled. It calls the existing symbolic
runner and saves both conditions in `ophmm-playthrough-comparison`. It does
not replace the default reference or publish either condition to the leaderboard.

```sh
python matchmaker_eval/test_symbolic.py --method outerhmm --fold eval \
  --global-transition off --no-plots --output results/ophmm-off
```

Use `on` for the paired comparison. Disabling the option zeros the skip-from
probabilities before alignment; local transitions and pitch observations are
unchanged. The setting is saved in `config.yaml`, merged metadata and per-piece
records. The merge requires 146 completed performances per condition and checks
that the disabled transition probabilities remained zero.

The [diagnostic suites](diagnostics.md) retain their explicit repeat/skip
settings. Selecting a play-through setting does not change those experiments.
