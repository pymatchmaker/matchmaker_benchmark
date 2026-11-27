# Transcribe twice and follow

Team Björk Guðmundsdóttir (Matthias Nagl, Klaus Lex, Kevin Eberl), Music
Informatics WS26, JKU Linz.

An audio score follower in two steps, like our first submission
(`bjork-gudmundsdottir`): piano transcription turns the audio into notes while
it arrives, and a symbolic follower aligns those notes to the score. What is
new is that two transcribers run side by side. The slower one decides where
in the score the performer is, the faster one decides when the next chord
comes.

## Why two transcribers

A transcriber that looks 5 frames ahead finds notes reliably, but it knows a
note only about 166 ms after it is played. Our first submission bridges that
delay by moving on at the followed tempo between notes. A transcriber without
lookahead knows a note after 46 ms, but when we followed its notes alone, the
follower lost some validation pieces in the first seconds: it ran ahead on
the first few chords, and a follower that only moves forward cannot undo
that. Here the fast notes never move the follower itself. They only time the
chords that come after the slow follower's last chord.

## How it works

1. **Transcription** (`DualProcessor` in `solution.py`). One log-filtered
   spectrogram (4096-sample window, hop 882, 50 frames per second, the same
   features as madmom's note CNN) is computed one frame at a time and fed to
   two networks of our own transcriber (`transcriber_b/stream.py`, numpy).
   Both have the same architecture: 5 blocks of 3x3 convolutions, a
   one-directional GRU with 256 units, and sigmoid heads for onset, sounding
   note and offset of the 88 keys (1.5 million parameters).
   - `anchor.npz`: each convolution block sees one future frame, so 5
     frames of lookahead. A note starts where the onset activation of a key
     is at least 0.5, higher than one frame earlier and at least as high as
     one frame later, and the key has not started a note in the last 50 ms.
     Onsets are final 166 ms after their frame.
   - `fast.npz`: no lookahead. The same rule without the condition on the
     next frame. Onsets are final 46 ms after their frame (half the window).

   Every onset keeps the time of the frame it belongs to.
2. **Following where** (`TranscriptionFollower`, unchanged from
   `bjork-gudmundsdottir`). The anchor's onsets within 35 ms of the first
   note of a chord form one chord, as in Matchmaker's `ChordOnsetProcessor`.
   Each chord goes to Matchmaker's `OnlineTimeWarpingArztEvent` with the
   settings of Matchmaker's MIDI `arzt` follower.
3. **Following when** (`DualFollower`). The fast onsets are grouped into
   chords at their first note (35 ms). Every fast chord that starts more than
   35 ms after the anchor's last chord moves the reported position one score
   chord further than the anchor's last chord. When the anchor takes its next
   chord, the count starts again from there, so a wrong fast note only
   matters until then.
4. **Reported position.** From the last chord (fast or anchor), the position
   moves on at the tempo the anchor followed over the last 4 seconds (at most
   8 beats per second, for at most 1.5 seconds after the chord), as in
   `bjork-gudmundsdottir`.

## Validation fold

One run of the benchmark's own runner on the 20 validation performances,
seed 0 pair (the pair we fixed before running it):

| | Tracked | Mean of per-piece median beat errors | Within 0.1 beat | Within 0.05 beat |
|---|---|---|---|---|
| `bjork-gudmundsdottir` | 20/20 | 0.085 | 0.667 | 0.470 |
| this submission | 20/20 | 0.023 | 0.792 | 0.733 |

Real-time factor about 0.13 (two pieces at a time on a 2-core machine). The rule that combines the two transcribers was designed after we had seen
validation results of the fast transcriber alone, so these numbers are not an
unbiased estimate. No setting was tuned on the validation fold: the onset
thresholds come from the MAESTRO validation split, everything else from
`bjork-gudmundsdottir`. The output jumps from chord to chord more than our
first submission's, so it is less smooth.

## Training data

Both networks were trained by us on the MAESTRO v3.0.0 train split only. The
MAESTRO validation split chose the checkpoint (best onset F1) and the onset
threshold (0.5 for both). ASAP's audio are MAESTRO recordings: all 42 ASAP
recordings of the benchmark folds (10 valid, 32 eval) are in MAESTRO's test
split, listed in `asap_in_maestro.csv` with their MAESTRO file names. That
split was never prepared or read, and our data preparation stops if any of
the 42 appears in the train or validation split. We used MAESTRO with the
agreement of the course instructor. Batik and Vienna 4x22 are not part of
MAESTRO.

## Files

| File | What it is | Licence |
|---|---|---|
| `solution.py` | Processor, both followers and registration | Apache-2.0 |
| `transcriber_b/` | Our transcriber's spectrogram and frame-by-frame numpy network | Apache-2.0 |
| `anchor.npz` | Weights of the network with lookahead, trained by us | CC BY-NC-SA 4.0 |
| `fast.npz` | Weights of the network without lookahead, trained by us | CC BY-NC-SA 4.0 |
| `settings.json` | Both onset thresholds and the weights' sha256 | |
| `asap_in_maestro.csv` | The 42 benchmark recordings in MAESTRO's test split | |
| `requirements.txt` | No dependencies beyond the benchmark environment | |

The weights carry CC BY-NC-SA 4.0 because MAESTRO, their training data, is
licensed under it.

## References

- C. Hawthorne et al. Enabling Factorized Piano Music Modeling and Generation
  with the MAESTRO Dataset. ICLR 2019.
- C. Hawthorne et al. Onsets and Frames: Dual-Objective Piano Transcription.
  ISMIR 2018. (Our sounding-note head also gets the onset probabilities, as
  there.)
- R. Kelz, S. Böck and G. Widmer. Deep Polyphonic ADSR Piano Note
  Transcription. ICASSP 2019. (The spectrogram features.)
