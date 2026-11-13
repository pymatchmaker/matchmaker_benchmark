# Transcribe and follow

Team Björk Guðmundsdóttir (Matthias Nagl, Klaus Lex, Kevin Eberl), Music
Informatics WS26, JKU Linz.

An audio score follower in two steps. A piano transcriber turns the audio
into notes while it arrives, and a symbolic follower aligns those notes to
the score.

## How it works

1. **Transcription** (`TranscriptionProcessor` in `solution.py`). madmom's
   piano note CNN (Kelz, Böck and Widmer, 2019) gives onset activations for
   the 88 keys at 50 frames per second. madmom computes them for a whole
   recording at once, so `stream_cnn.py` reimplements the network in numpy
   and computes it one spectrogram frame at a time. On the validation fold
   its output differs from madmom's by at most 6.2e-6. A note starts where
   the onset activation of a key is at least 0.5, higher than one frame
   earlier and at least as high as one frame later, and the key has not
   started a note in the last 50 ms. Every onset keeps the time of the frame
   it belongs to.
2. **Following** (`TranscriptionFollower`). Onsets within 35 ms of the first
   note of a chord form one chord, as in Matchmaker's `ChordOnsetProcessor`.
   Each chord goes to Matchmaker's `OnlineTimeWarpingArztEvent` with the
   settings of Matchmaker's MIDI `arzt` follower.
3. **Reported position.** The network needs 5 frames of future context, so a
   note is known about 170 ms after it is played. The follower reports the
   position of the last chord, moved on at the tempo it followed over the
   last 4 seconds, from that chord's own onset time (at most 8 beats per
   second, for at most 1.5 seconds after the chord).

## Files

| File | What it is | Licence |
|---|---|---|
| `solution.py` | Processor, follower and registration | Apache-2.0 |
| `stream_cnn.py` | Frame-by-frame numpy version of madmom's note CNN | Apache-2.0 |
| `notes_cnn_2019.npz` | madmom's model weights, window and filterbank | CC BY-NC-SA 4.0 |
| `requirements.txt` | No dependencies beyond the benchmark environment | |

## Attribution for `notes_cnn_2019.npz`

The weights are madmom's piano note CNN, `notes/2019/notes_cnn.pkl` in
[CPJKU/madmom_models](https://github.com/CPJKU/madmom_models), by Rainer Kelz,
Sebastian Böck and Gerhard Widmer. They are licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), and so
is this file.

Changes: we exported the network weights, together with the spectrogram's
window and filterbank from madmom, into one numpy `.npz` file, so the
follower needs nothing but numpy. The values are unchanged.

## References

- R. Kelz, S. Böck and G. Widmer. Deep Polyphonic ADSR Piano Note
  Transcription. ICASSP 2019, pp. 246 to 250.
- S. Böck, F. Korzeniowski, J. Schlüter, F. Krebs and G. Widmer. madmom: a
  new Python Audio and Music Signal Processing Library. ACM Multimedia 2016.
