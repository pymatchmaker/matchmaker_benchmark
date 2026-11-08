# Running the manuscript diagnostics

In Actions, choose **Evaluate references**, set `fold` to `diagnostics`, and
use four shards. Select the branch containing this workflow. No leaderboard
files are changed by this run. `benchmark-results-diagnosis` contains all raw
paths, ground truth, per-case metrics, logs, input hashes and a merged summary.
The merge fails if any case or shard is missing, duplicated or failed.

For non-piano evaluation use the same workflow with `fold=beyond_piano`,
`input_type=audio`. Its artifact is `benchmark-results-beyond_piano`.

Diagnostic inputs are pinned in `data/diagnostics.json`. The runner verifies
the manifest hash and each selected input, including losslessly encoded audio.
The source data release documents the original generator scripts and licenses.
No regeneration with an unrecorded soundfont occurs in CI.

The diagnostic configurations are in `matchmaker_eval/diagnostics.py`:

- Perturbations: four audio and six MIDI followers on 41/34 conditions.
- Repeat examples: Arzt, Multi-Fold Arzt and OPHMM on two K.282 excerpts.
- Folded Batik: Dixon, Arzt, SKF, OPHMM and Multi-Fold Arzt on 24 test movements.
- Global skip: symbolic OPHMM on two K.279 conditions, changing only its
  skip-from distribution to disable the global transition.

Audio OPHMM uses `s_j=1e-100` for the synthetic diagnostic suites, where jumps
are part of the experiment. The folded-Batik OPHMM configuration uses `s_j=1e-5`, as in the original
folded-score diagnostic protocol. The play-through baseline registry uses zero.
Nominal tempo is provided only for the two K.282 illustrative cases (60 BPM).
The synthetic perturbations and recorded folded cases receive the normal
score-based tempo initialization, not the performance's estimated tempo.

All suites use the same `run_evaluation`, causal onset sampling, 3 kHz SPARC
implementation, and musical-beat tracking criterion as the main benchmark.
Accuracy remains in score quarter-note beats. The input manifest specifies
whether the score is unfolded. Repeat ground truth retains backward jumps.
The perturbation summary averages per-condition MeanAE over tracked cases;
raw per-case paths also support pooled accuracy and common-tracked analyses.
Timing measured on these runners is retained separately from previously
reported RTF. A new runner measurement must not silently replace an earlier
measurement or be selected because it is smaller.

Local reproduction:

```sh
python matchmaker_eval/diagnostics.py run \
  --config perturbations-audio-arzt --shard 0 --shards 4 --output results/diagnostics/arzt-0
```

Use the library revision installed by the workflow. Artifacts record both
library and benchmark commits and the GitHub run ID. A raw-path re-score from
an old local experiment is not equivalent to a CI execution of a follower.
