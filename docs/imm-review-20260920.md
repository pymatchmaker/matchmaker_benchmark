# IMM implementation review and diagnostic evaluation — 2026-09-20

## Current result: final SoftOLTW + IMM

The main manuscript comparison is SoftOLTW versus the final combined method. Earlier IMM checkpoints below are development history, not additional methods required in the manuscript.

| Full146, event-pooled tracked-only accuracy | TR | Beat MAE | MedAE | ≤0.5b | ≤1b |
|---|---:|---:|---:|---:|---:|
| SoftOLTW | 86.99% (127/146) | 0.2938 | 0.1000 | 86.57% | 94.33% |
| **SoftOLTW + IMM** | **91.78% (134/146)** | **0.2680** | **0.0840** | **88.13%** | **95.00%** |

Dataset successes are ASAP 30/32, Batik 26/30, Vienna 78/84. Relative to SoftOLTW there are nine gains and two losses, a net gain of seven. The two losses are Vienna Schubert D783 no.15 p13 and p21. On their common 125 tracked pieces, baseline/final MAE is 0.2900/0.2589, MedAE 0.1000/0.0828, ≤0.5b 86.68/88.44%, and ≤1b 94.41/95.21%. The goal of at least 133 tracked pieces is achieved. This is practical benchmark improvement, not a statistical significance claim.

The selected validation20 result is 19/20, MAE 0.1994, MedAE 0.0731, ≤0.5b 90.30%, ≤1b 95.92%; baseline is 19/20, 0.2306, 0.1000, 89.01%, 95.64%. Validation and evaluation share no identical `(dataset, audio_performance)` pairs. No new scalar parameter was swept. Architecture selection still constitutes model selection, and the previously inspected full set is not an unseen test.

The final implementation adds one Gaussian position/tempo estimate per DP position candidate. Incoming path branches use the same constant-velocity dynamics, Gaussian predictive transition costs, and Kalman updates, followed by moment matching including between-mean covariance. Traversed score-frame acoustic costs and path-length normalization are retained. Process noise is calibrated to the existing observation variance over one nominal beat. Existing step size, gamma, observation variance and score time scale are reused; these modeling choices acquire new roles and are not parameter-free.

This is one bounded-step DP lattice with path-conditioned Kalman mixtures, followed by the existing correlated-error CV/CA/ZV IMM. The DP's Kalman mixtures combine path histories; the output IMM combines different motion models. The output IMM posterior is not fed into the lattice transition calculation. Thus the full improvement cannot be attributed to the output IMM alone. The changed DP recurrence is a motion-regularized DTW approximation, not the original unchanged SoftOLTW recurrence or an exact Bayesian posterior.

A validation ablation removing only the Kalman transition cost, while retaining DP geometry and output IMM, obtained 19/20, MAE 0.2069, MedAE 0.0744, ≤0.5b 89.99%, ≤1b 95.84%. This supports an additional contribution from the transition prior on validation; it does not isolate the output IMM or establish full-set attribution.

Implementation commit: `06e1ae3`. Production defaults exactly matched the frozen candidate over 720 synthetic steps; 55 implementation tests passed. Full-run artifacts are `output/imm_path_dtw_full_146_20260920/{summary_tracked,summary_common_tracked,comparison_tracked,verification}.json`, with frozen selection and source snapshots in `protocol/`. Every audio run uses `matchmaker_eval/test_audio.py --no-plots`.

Mean millisecond error also falls (148.4645 to 146.6399), but median millisecond error rises (35.4238 to 46.8750); not every metric improves. The fresh reproduction under `output/paper_reproduction_146_20260920` completed both SoftOLTW variants. At the user's request, the remaining five audio reruns were cancelled. Their manuscript accuracy cells have not been newly verified. The two-method common subset contains 125 pieces. SPARC uses each method's own tracked pieces, averaging 30-second windows within each piece and then across pieces.

## Refactoring verification

Implementation commit `225038a` separates acoustic SoftOLTW, the path-conditioned Kalman lattice, and output IMM into their respective modules. The baseline constructs neither Kalman component. Hierarchical alignment retains composition because it owns multiple followers. Unused experimental paths and diagnostics were removed without changing numerical parameters.

All 56 implementation tests pass. A 1,588-step comparison across baseline, output-IMM-only, final combined, and hierarchical configurations produced exactly equal states and paths. The final refactored validation run reproduces all 20 saved paths byte for byte. The full baseline rerun reproduces all 146 saved paths byte for byte, retaining 127 tracked pieces and all accuracy values above. Its measured all-piece RTF is 0.0456411. Artifacts are in `output/imm_refactor_verification_20260920`, including `protocol/final_verification.json`.

## SPARC correction

The legacy report helper removed mean speed before spectrum normalization, assigned constant-speed windows an arbitrary -100, and normalized frequency differences by the fixed cutoff rather than the selected spectral span. Removing DC makes relative jitter magnitude disappear after peak normalization. The manuscript also mixed values obtained under different definitions. These errors explain the apparent reversal; no tracker was changed to obtain the corrected smoothness values.

The existing report helper now follows the [authors' SPARC implementation](https://github.com/siva82kb/SPARC/blob/master/scripts/smoothness.py), with the definition in [Balasubramanian et al. (2015)](https://doi.org/10.1186/s12984-015-0090-9): retain the speed spectrum's DC component, normalize its magnitude, select support below 10 Hz with relative amplitude threshold 0.05, and normalize arc length by that support's frequency span. Speed is the absolute derivative of position resampled at 50 Hz. Windows last 30 seconds with a 15-second hop; short pieces use their full duration. Zero-speed windows have undefined normalized spectra and are excluded. No parameters were tuned. Three reporting tests pass, including the authors' Gaussian example, amplitude scaling, jitter magnitude, and native event pooling.

| Method | Own tracked pieces | Corrected SPARC |
|---|---:|---:|
| Arzt | 104 | -12.4359 |
| Arzt with Tempo | 108 | -12.5126 |
| OPHMM | 78 | -27.7876 |
| SoftOLTW | 127 | -11.8547 |
| **SoftOLTW + IMM** | 134 | **-5.4245** |

The saved Dixon and SKF runs have 99 and 88 successes, whereas the manuscript rows have 101 and 89. Their recomputed SPARC values therefore cannot populate those manuscript rows; those cells are left blank (`--`). No additional audio methods were rerun. Detailed provenance and values are in `output/paper_reproduction_146_20260920/sparc_corrected_audit.json`; the corrected helper and tests are preserved in `protocol/sparc/`. RTF values are never bold in the manuscript.

## Historical 130-piece checkpoint

The following sections describe the earlier frozen checkpoint and its contemporaneous conclusions; references to its final implementation apply only to that checkpoint.

## Outcome and reporting convention

Accuracy is **event-pooled across tracked pieces**, taken from `summary_tracked.json`. Tracking count uses every evaluated piece. Dataset means are not averaged to form the 146-piece result. Different methods may track different subsets; count and accuracy must be read together.

The requested beat-domain target is achieved by the **augmented-state IMM with correlated observation error**: 130/146 tracked, versus SoftOLTW's 127, with lower tracked MAE and higher coverage within both 0.5 and 1.0 beats. Earlier unsuccessful implementations are retained below as experiment history; the earlier three-state model is superseded.

| Final full evaluation | TR | Tracked beat MAE | MedAE | ≤0.5 beat | ≤1 beat |
|---|---:|---:|---:|---:|---:|
| SoftOLTW | 86.99% (127/146) | 0.2938 | 0.1000 | 86.57% | 94.33% |
| Single KF + same correlated observation model | 86.99% (127/146) | 0.2859 | 0.0885 | 87.15% | 94.40% |
| **IMM — correlated observation error** | **89.04% (130/146)** | **0.2857** | **0.0873** | **87.22%** | **94.41%** |

All 127 baseline successes remain tracked. Three Vienna Chopin Op.38 performances, p04, p06, and p09 (indices 25, 27, 30), become tracked. Dataset counts are ASAP 30/32, Batik 25/30, Vienna 75/84. On the common 127 tracked pieces, IMM MAE is 0.2805, MedAE 0.0866, ≤0.5b coverage 87.37%, and ≤1b coverage 94.52%; improvements are not an easier-subset artifact.

The single KF comparison uses the same augmented observation model, with continuous white acceleration calibrated to position variance R over one beat. It is deliberately stronger than the zero-process-noise CV mode alone. It recovers Vienna indices 25 and 30 but loses 24 and 47, giving 127/146. This is a comparison against a reasonable single-dynamics alternative, not a pure deletion-of-modes ablation with every dynamics matrix unchanged.

On the 125 pieces tracked by all three methods:

| Method | Common tracked MAE | MedAE | ≤0.5 beat | ≤1 beat |
|---|---:|---:|---:|---:|
| SoftOLTW | 0.2914 | 0.1000 | 86.68% | 94.41% |
| Single KF + correlated observation error | 0.2819 | 0.0881 | 87.25% | 94.48% |
| **IMM — correlated observation error** | **0.2781** | **0.0863** | **87.47%** | **94.60%** |

The matched-comparator record is `output/imm_colored_full_146_20260920/comparison_with_single_kf.json`; the single KF source and results are under `output/imm_colored_cv_full_146_20260920`. Its validation result was frozen before full evaluation. These differences do not establish universal or statistically significant superiority over single Kalman filters.

The evidence is `output/imm_colored_full_146_20260920/{summary_tracked,summary_common_tracked,comparison_tracked}.json`, with source and validation selection frozen in `protocol/` before evaluation. The final model was selected on validation: 19/20 tracked, MAE 0.2155, MedAE 0.0757, ≤0.5b 89.67%, ≤1b 95.65%. Baseline validation is 19/20, 0.2306, 0.1000, 89.01%, and 95.64%. The 1b validation gain is tiny; no statistical significance is claimed.

The target concerns beat-domain metrics. It does not imply improvement of every metric: tracked score-to-performance mean error in milliseconds is 155.6334 versus 148.4645 for baseline, and ≤2b coverage is 97.82% versus 97.95%. These limitations must not be hidden behind a general claim that all alignment measures improve. Tracking success uses the unchanged benchmark criterion and does not mean every event has error below one beat.

## Final implementation

The canonical IMM recursion is retained. The state is `[position, velocity, acceleration, observation_error]`; all CV/CA/ZV modes share the observation `z = position + observation_error + white_noise`. The added error state is first-order Gauss–Markov, so a persistent DP localization error is not automatically interpreted as repeated independent evidence of changed tempo.

For a score ambiguity span of L reference frames, bias variance is `L²/12`, correlation is `exp(-1/L)`, and driving noise variance is `L²/12 * (1-rho²)`. These are explicit score-based assumptions, not a fitted per-piece rule. Existing white observation variance R=5 and the previously defined dynamics time scales are retained. No numeric parameters were selected from full146. Repeated onsets are neither deleted from the acoustic input nor specially detected by this model.

The default acoustic SoftOLTW path is unchanged; this is an augmented-state output estimator, not evidence that the DP itself has reacquired a lost path. The score-dependent persistent-error model is what distinguishes this successful candidate from the earlier independent-observation variants. Theory, alternatives and primary-source links are in [the literature review](imm-literature-review-20260920.md).

Final code changes are restricted to `matchmaker/dp/oltw_soft.py` and its relevant tests. Experiments use isolated package copies; rejected variants are not added as production branches. The SKF unit correction was evaluated separately and was not applied to SKF production code. No manuscript results have been silently rewritten.

## Earlier full evaluations

| Completed full evaluation | Tracked | Tracked beat MAE | ≤0.5 beat | ≤1 beat |
|---|---:|---:|---:|---:|
| SoftOLTW | 127/146 | 0.2938 | 86.57% | 94.33% |
| First repair: legacy IMM output filter | 126/146 | 0.2906 | 85.87% | 94.10% |

On the 126 common tracked pieces, baseline MAE is 0.2921 and the first repair is 0.2906, but both tolerance coverages worsen. Thus the small MAE reduction is not solely a membership artifact and still does not establish overall superiority.

These results are in `output/imm_full_146_20260920/{no_imm,imm}/summary_tracked.json`. Earlier all-piece MAE reporting is superseded by this table.

## Protocol and limits

All audio runs use the existing `matchmaker_eval/test_audio.py --no-plots`. The runner gained multiprocessing, one-based row selection, JSON method overrides, explicit output directories, and a completeness check. Incomplete runs fail instead of silently producing a successful summary. Concurrent-run RTF is not a controlled runtime comparison.

Architecture comparisons and any selection use only `data/metadata-validation.csv` (20 performances). Validation and evaluation have no identical audio paths. No numeric grid sweep was performed, but multiple architecture variants were tried; this is model selection, not a claim of zero tuning. The original observation variance of 5 reference-frame² was retained, and remains a modeling choice. Score-derived time scales and noise calibration are also assumptions, not uniquely implied by IMM theory.

At the user's request, full-set Vienna baseline failures were inspected before the later redesign. Therefore the later 146-piece evaluation is a diagnostic repeat, **not a pristine unseen test**. No full-set scores are used to select numeric values or choose the final variant. Source snapshots and frozen settings precede each full run in its `protocol` directory.

## Earlier three-state model

State units are reference frames, reference frames/second, and reference frames/second². The filter implements the usual IMM interaction, per-model prediction/update, innovation-likelihood mode update, and posterior moment matching.

- CV propagates constant velocity without process noise. CA models continuous white jerk. ZV holds position. CV has no artificial minimum speed.
- The CA jerk spectral density is `20 R / T^5`, where `T` is the notated beat duration: integrated position process variance over one beat is `R`. The factors in the covariance matrix are integration coefficients, not fitted constants.
- Mode transitions use the exact matrix exponential of a continuous-time generator. CV residence time is the median notated measure duration; CA and ZV use one beat. Eligible exit destinations have equal prior probability. These musical time scales replace hand-entered per-frame transition probabilities.
- Global rests and fermatas make ZV eligible; they never force a pause. A written rest still consumes score time. There is no audio-chroma threshold overriding IMM mode probabilities. The baseline's initial silence gate is unchanged.
- Gaussian log likelihoods avoid an arbitrary probability floor. Impossible modes retain exactly zero probability. Joseph covariance updates preserve covariance numerics.
- There is one posterior output, without blending the same observation into it a second time.
- `tempo_model="cv"` uses the same state units, observations, and strict CV model as a removal-of-modes ablation. It tracks only 7/20 validation pieces, so the full comparison also uses the previously validated white-acceleration single KF as a stronger comparator. That source is frozen separately as `protocol/oltw_soft_cv.py`; its position process variance over one beat is also R. This stronger comparator is not a pure removal-of-modes experiment. `score_observation="fixed"` removes only score-dependent observation variance.

The reference for the canonical recursion was the [FilterPy IMM implementation](https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/IMM.html). The score adaptation and the noise/time-scale assumptions above are our design choices.

### Score-predicted uncertainty

The score is partitioned into maximal adjacent spans with identical active pitch-class support. Repeated attacks of the same pitch-class set remain in one span. If its length is `L` reference frames, harmonic identity alone leaves uniform phase uncertainty of `L²/12`. At the predicted position the observation model uses `R_t = R + L²/12`.

This increases reliance on tempo prediction in harmonically stationary passages without a learned threshold or new scalar weight. It is only a proxy: chroma amplitudes, attacks, decay and voicing can contain information beyond pitch-class support. It does not make the recurrent DP endpoint an unbiased, independent Gaussian measurement. Repeated biased endpoint observations remain a major limitation.

The acoustic observation retains SoftOLTW's monotonic/reachable-step constraint. Unrestricted DP minima can jump between repetitions; feeding them back shifted the corridor catastrophically on validation. The default therefore leaves the acoustic DP unchanged. The failed experimental feedback branch was removed from the final implementation. The final implementation is a tempo posterior over the acoustic path; it does not demonstrate the manuscript's closed-loop recovery claim.

## Earlier validation experiments

All rows below have 20 performances; accuracy includes their 19 tracked pieces. The complete architecture comparison is `output/imm_validation_20260920/validation_tracked.tsv`; source snapshots are retained with experiments so unsuccessful trials are visible.

| Model | Tracked | Tracked beat MAE | ≤0.5 beat | ≤1 beat |
|---|---:|---:|---:|---:|
| SoftOLTW | 19/20 | 0.2306 | 89.01% | 95.64% |
| Legacy repaired output filter | 19/20 | 0.2244 | 88.41% | 95.23% |
| Standard IMM, white-acceleration CV, score variance | 19/20 | 0.2285 | 88.56% | 95.20% |
| Single CV KF with white acceleration, score variance | 19/20 | 0.2294 | 88.55% | 95.30% |
| Selected standard IMM, constant CV, score variance | 19/20 | 0.2274 | 88.69% | 95.23% |
| Correlation-scaled observation variant, constant CV | 19/20 | 0.2285 | 89.14% | 95.45% |

The constant-CV/score-variance variant was selected among standard score-uncertainty models for its lower MAE and simpler observation model. This is not the best legacy MAE and is not uniformly better than the baseline. Direct predictive corridor gating, event-only observations, local curvature uncertainty, and acoustic candidate association did not establish an improvement. Their unsuccessful validation results must not be presented as successful IMM ablations.

## Vienna failure diagnosis

Baseline tracking by composition: Chopin Op.10 No.3 21/21, Chopin Op.38 11/21, Mozart K331 movement 1 21/21, Schubert D783 No.15 19/21. Thus 10 of Vienna's 12 failures are Op.38.

Op.38 evaluation performances p04 and p18 both stall around score beat 5.8667 for approximately 3.87 s and 3.13 s, respectively, before larger lag accumulates. The validation Op.38 p20 also exhibits severe lag. The diagnostic figure is `output/imm_full_146_20260920/diagnostics/vienna_failure_modes.png`.

This is not just noisy instantaneous tempo: the acoustic endpoint itself stalls, and filtering repeated stalled endpoints teaches the model a near-zero velocity. Adding CA does not create acoustic evidence for the correct location. Score uncertainty can weaken these observations, but a filter after an already lost path has limited recovery ability. Our unsuccessful validation feedback experiments do not justify claiming this problem is solved.

## Manuscript discrepancies

Reviewed file: `/Users/jiyun/workspace/overleaf/ICASSP2027_SoftOLTW/Template.tex`. The manuscript has not been rewritten to claim unsupported improvements.

1. The historical `benchmark_full_146_imm_restored` archive tracks 101/146 with IMM and 127/146 without it, not the draft's 135 and 120. That archive uses a different replay runner and is not interchangeable with these event-pooled metrics.
2. The text specifies cosine distance; tested defaults use Manhattan distance.
3. The displayed softmin equation adds local distance both outside the log sum and inside the candidates; the implementation counts the weighted candidate distance once.
4. The old per-frame transition diagonals do not match the stated mean sojourn times. The final implementation uses a continuous-time generator with explicitly documented musical time scales.
5. The draft describes a closed-loop corridor/step-penalty method. The validated default output filter does not implement that claimed behavior.
6. Claims that multiple dynamics necessarily outperform a single Kalman filter require the matched CV ablation. They cannot be inferred from model flexibility alone.

## Reproduction

The command below now uses the final correlated-observation implementation in the working repository. To reproduce an earlier experiment, use that experiment's frozen source rather than the current working source.

From `/Users/jiyun/workspace/matchmaker-benchmark`:

```sh
export PYTHONPATH=/Users/jiyun/workspace/matchmaker-laurenceyoon
export NUMBA_CACHE_DIR=/tmp/imm-numba
export MPLCONFIGDIR=/tmp/imm-mpl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python matchmaker_eval/test_audio.py --dataset valid --method softoltw --kwargs '{"score_observation":"variance","tempo_model":"imm"}' --workers 4 --no-plots --output-dir output/imm_reproduction_valid
```

For the full evaluation, use `asap`, `batik`, and `vienna` with separate output directories and fixed settings. For the strict single-model ablation use `"tempo_model":"cv"`. To reproduce the stronger white-acceleration CV comparator, place `protocol/oltw_soft_cv.py` in an isolated copy of the package and use that copy on `PYTHONPATH` with `"tempo_model":"cv"`; baseline is `--method softoltw_no_imm`. The final IMM record is `output/imm_colored_full_146_20260920/protocol/frozen_selection.json`. The matched colored-observation single KF comparator is frozen separately in `output/imm_colored_cv_full_146_20260920/protocol/oltw_soft.py`. It adds white-acceleration noise to CV, calibrated to the same beat-horizon position variance; it is a stronger comparator, not a pure removal-of-modes ablation.

## Verification

The final IMM/SoftOLTW and hierarchical integration tests pass (22). The benchmark runner tests previously passed (6), and the runner has not changed since. Checks include continuous-time subdivision, reference IMM and augmented Kalman recursions, stationary colored-error variance, covariance stability, mode probabilities, and unchanged acoustic paths. Final working-source validation reproduction under `output/imm_literature_validation_20260920/final_reproduction` exactly matches the selected metrics. After removing the unused failed feedback branch, a 720-step comparison with frozen full-run source gives identical states, covariances, mode probabilities and acoustic positions. Final source hashes and goal checks are in `output/imm_colored_full_146_20260920/verification.json`.

## Earlier three-state full evaluation (superseded)

All three datasets completed without evaluation exceptions. Every run used `test_audio.py --no-plots`; no settings were changed in response to the full results.

| Method | Tracked | Tracked beat MAE | ≤0.5 beat | ≤1 beat |
|---|---:|---:|---:|---:|
| SoftOLTW | 127/146 | 0.2938 | 86.57% | 94.33% |
| Single CV with white acceleration + score variance | 126/146 | 0.2958 | 86.27% | 93.98% |
| Standard IMM + score variance | 124/146 | 0.2901 | 86.47% | 94.09% |

On the **124 pieces tracked by all three methods**, membership is fixed:

| Method | Common tracked beat MAE | ≤0.5 beat | ≤1 beat |
|---|---:|---:|---:|
| SoftOLTW | 0.2876 | 86.75% | 94.46% |
| Single CV with white acceleration + score variance | 0.2918 | 86.38% | 94.06% |
| Standard IMM + score variance | 0.2901 | 86.47% | 94.09% |

IMM is slightly more accurate than the adaptive single CV on these common pieces, but both underperform SoftOLTW. IMM also loses two more tracked pieces than the single CV. Its lower own-tracked MAE than baseline must not be treated as an accuracy gain on the same recordings.

| Method | ASAP | Batik | Vienna |
|---|---:|---:|---:|
| SoftOLTW | 30/32 | 25/30 | 72/84 |
| Single CV with white acceleration + score variance | 30/32 | 25/30 | 71/84 |
| Standard IMM + score variance | 30/32 | 25/30 | 69/84 |

The standard IMM recovers none of the baseline failures and loses three Vienna performances: Op.38 p10 (worst 30-second median error 0.9667→1.0331 beats), Schubert p13 (0.9000→1.0263), and Schubert p21 (1.0000→1.0118). The existing success threshold was not changed. These marginal failures are still failures under the prescribed metric.

Full comparison: `output/imm_standard_full_146_20260920/comparison_tracked.json`. Piece changes: `tracking_changes.tsv`. The baseline was preferable to this earlier three-state model. The later augmented observation-error model at the top of this report supersedes this conclusion for the current implementation.

## Completed Table 1 aggregation

All seven methods were re-evaluated from saved trajectories using the current canonical ground truth, existing `check_tracking`, and `evaluate_native` / `compute_event_pooled_summary`. No audio reruns or standalone evaluation scripts were needed. The common-tracked intersection contains 46 performances. All original Table 1 tracking counts are preserved.

Dixon and SKF trajectories matching the manuscript 101/89 successes were found in `results/submissions/{dixon,skf}-audio`; the preceding audit had missed this location. Two ASAP ground-truth files (indices 17 and 18) differ from the canonical annotations in those submissions. Both pieces fail for both methods under either annotation set, so tracked-only results are unaffected. Arzt / Arzt-with-Tempo use `output/arzt_vs_tempo_146`, OPHMM uses `output/linear_reduced146_20260906`, and the two SoftOLTW variants use their verified full runs. Existing Arzt all-tracked paper values used a different aggregation and are replaced by native event-pooled values.

| Method | TR | Tracked MAE | MedAE | ≤0.5b | ≤1b | Common MAE (46) | Common ≤1b | SPARC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arzt | 71.23% (104/146) | 0.3367 | 0.1000 | 85.25% | 93.22% | 0.2375 | 95.95% | -12.4359 |
| arzt_tempo | 73.97% (108/146) | 0.3434 | 0.1000 | 85.10% | 93.00% | 0.2373 | 95.86% | -12.5126 |
| dixon | 69.18% (101/146) | 0.3956 | 0.1317 | 82.19% | 91.94% | 0.3373 | 93.42% | -17.9970 |
| outerhmm | 53.42% (78/146) | 0.4825 | 0.2500 | 78.86% | 91.22% | 0.4389 | 92.29% | -27.7876 |
| skf | 60.96% (89/146) | 0.2476 | 0.1243 | 89.67% | 96.56% | 0.1867 | 97.98% | -15.0873 |
| softoltw_no_imm | 86.99% (127/146) | 0.2938 | 0.1000 | 86.57% | 94.33% | 0.2039 | 97.08% | -11.8547 |
| softoltw | 91.78% (134/146) | 0.2680 | 0.0840 | 88.13% | 95.00% | 0.1779 | 97.42% | -5.4245 |

Full reports, symlinks to source paths, and per-path hashes are saved under `output/table1_complete_20260920`. Timing protocols differ among historical sources; existing manuscript RTF cells are retained and are not claimed to have been reproduced on a common hardware setup. RTF values remain unbolded. The manuscript contains only the evaluation definition and results, not this provenance history.
