"""Team Björk Guðmundsdóttir (Music Informatics WS26, JKU Linz): transcribe twice, then follow.

Our first submission (bjork-gudmundsdottir) follows the notes of one
streaming transcriber that needs 5 frames of future context, so it knows a
note about 170 ms after it is played. This one adds a second transcriber
without that lookahead and uses its notes only for timing:

1. Processor: one spectrogram frame at a time (transcriber_b/features.py,
   frame k centred on sample 882 k, 50 frames per second), fed to two
   networks of our own transcriber (transcriber_b/stream.py, numpy):
   - anchor.npz: the network with 5 frames of lookahead. A note starts where
     the onset activation of a key is at least the threshold, higher than
     one frame earlier and at least as high as one frame later, and the key
     has not started a note in the last 50 ms. Onsets are final 166 ms after
     their frame.
   - fast.npz: the network without lookahead. The same rule without the
     condition on the next frame. Onsets are final 46 ms after their frame
     (half the 4096-sample window).
   Every onset keeps the time of the frame it belongs to.
2. Follower: the follower of bjork-gudmundsdottir, on the anchor's onsets
   alone, decides the score position (chords of 35 ms into Matchmaker's
   event-level online time warping, moved on at the followed tempo). The
   fast onsets only time the chords after it: every fast chord (grouped at
   its first note) that starts more than 35 ms after the anchor's last chord
   moves the output one score chord further than the anchor's last chord,
   and the position moves on from that fast chord. When the anchor takes its
   next chord, the count starts again from there, so the fast notes can
   never move the anchor itself.

Files next to this one (see README.md):
    transcriber_b/   numpy copies of our transcriber's features.py and stream.py
    anchor.npz       weights of the network with lookahead (trained by us)
    fast.npz         weights of the network without lookahead (trained by us)
    settings.json    both onset thresholds (chosen on the MAESTRO validation split) and sha256s
"""

import bisect
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from matchmaker import register_method
from matchmaker.base import OnlineAlignment
from matchmaker.dp import OnlineTimeWarpingArztEvent
from matchmaker.features.midi import onset_pianoroll
from matchmaker.features.processor import Processor

HERE = Path(__file__).resolve().parent


def _transcriber_stream():
    """transcriber_b.stream from the copy next to this file (the benchmark does not put it on sys.path)."""
    if "transcriber_b" not in sys.modules:
        package = HERE / "transcriber_b"
        spec = importlib.util.spec_from_file_location(
            "transcriber_b", package / "__init__.py", submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["transcriber_b"] = module
        spec.loader.exec_module(module)
    return importlib.import_module("transcriber_b.stream")


tb = _transcriber_stream()
features = importlib.import_module("transcriber_b.features")
SETTINGS = json.loads((HERE / "settings.json").read_text())

MIN_GAP_FRAMES = 2  # 50 ms at 50 frames per second, as in bjork-gudmundsdottir
CHORD_THRESHOLD = 0.035  # seconds, Matchmaker's ChordOnsetProcessor
LOWEST_KEY = 21  # MIDI pitch of the first of the 88 keys


class OnsetStream:
    """One network and its causal onset rule.

    peak=True: frame t is decided once frame t + 1 is known (needs a peak).
    peak=False: frame t is decided at once (a rising activation is enough).
    """

    def __init__(self, network, threshold, peak):
        self.network, self.threshold, self.peak = network, threshold, peak
        self.trace = None  # set to a list to record the onset activations (checks only)
        self.reset()

    def reset(self):
        self.network.reset()
        self.act_index = -1  # index of the newest activation frame
        self.before = None  # onset activations one frame before the frame being decided
        self.now = None  # onset activations of the newest frame (peak rule: the frame being decided)
        self.last_onset = np.full(88, -(10**9))

    @property
    def horizon(self):
        """Onsets up to this time (seconds) are final."""
        return (self.act_index - (1 if self.peak else 0)) / features.FPS

    def push(self, spectrum):
        """One spectrogram frame in, the onsets it decides out [(time, pitch)]."""
        act = self.network.push(spectrum)
        if act is None:
            return []
        self.act_index += 1
        onset = act[:, 1]
        if self.trace is not None:
            self.trace.append(onset.copy())
        if self.peak:
            found = []
            if self.before is not None:
                found = self._decide(self.act_index - 1, self.now, self.before, self.now >= onset)
            self.before, self.now = self.now, onset
            return found
        found = []
        if self.now is not None:
            found = self._decide(self.act_index, onset, self.now, np.ones(88, bool))
        self.now = onset
        return found

    def _decide(self, t, current, previous, extra):
        keys = np.flatnonzero(
            (current >= self.threshold) & (current > previous) & extra & (t - self.last_onset >= MIN_GAP_FRAMES)
        )
        self.last_onset[keys] = t
        return [(t / features.FPS, float(key + LOWEST_KEY)) for key in keys]


class DualProcessor(Processor):
    """Audio in, two onset streams out.

    Returns, for every block of audio, (anchor rows, fast rows): arrays whose
    first row is (horizon, -1) and whose other rows are (onset time, MIDI
    pitch) of the onsets found in this block.
    """

    def __init__(self, sample_rate, hop_length, anchor_threshold, fast_threshold):
        super().__init__()
        if sample_rate != features.SAMPLE_RATE:
            raise ValueError(f"the transcriber needs {features.SAMPLE_RATE} Hz audio, got {sample_rate}")
        self.hop_length = hop_length
        # The stream hands over n_fft samples per block: the newest hop and the
        # one before it. Only the newest hop is new, so only that is used.
        self.n_fft = 2 * hop_length
        self.spectrogram, anchor, _ = tb.load(HERE / "anchor.npz")
        _, fast, _ = tb.load(HERE / "fast.npz")  # the spectrogram does not depend on the weights
        self.anchor = OnsetStream(anchor, anchor_threshold, peak=True)
        self.fast = OnsetStream(fast, fast_threshold, peak=False)
        self.reset()

    def reset(self):
        self.anchor.reset()
        self.fast.reset()
        half = features.FRAME // 2
        # Frame k is centred on sample HOP * k, so frame 0 needs half a frame
        # of zeros before the recording starts (features.frames_of does the same).
        self.audio = np.zeros(half, np.float32)
        self.audio_start = -half  # sample index of self.audio[0]
        self.received = 0
        self.next_frame = 0

    def __call__(self, data):
        samples, perf_time = data
        new = np.asarray(samples[-self.hop_length :], dtype=np.float32)
        self.audio = np.concatenate((self.audio, new))
        self.received += len(new)

        anchor_onsets, fast_onsets = [], []
        half = features.FRAME // 2
        while self.next_frame * features.HOP + half <= self.received:
            start = self.next_frame * features.HOP - half - self.audio_start
            spectrum = self.spectrogram.frame(self.audio[start : start + features.FRAME])
            self.next_frame += 1
            anchor_onsets.extend(self.anchor.push(spectrum))
            fast_onsets.extend(self.fast.push(spectrum))

        keep_from = self.next_frame * features.HOP - half  # first sample the next frame needs
        if keep_from > self.audio_start:
            self.audio = self.audio[keep_from - self.audio_start :]
            self.audio_start = keep_from

        anchor = np.asarray([(self.anchor.horizon, -1.0)] + anchor_onsets, dtype=np.float64)
        fast = np.asarray([(self.fast.horizon, -1.0)] + fast_onsets, dtype=np.float64)
        return (anchor, fast), perf_time


class TranscriptionFollower(OnlineAlignment):
    """The follower of our first submission, bjork-gudmundsdottir, unchanged."""

    def __init__(self, score_note_array, queue, config):
        features_, positions = onset_pianoroll(score_note_array, onset_key="onset_beat", piano_range=True)
        super().__init__(features_, positions, queue)
        # The same follower and settings as Matchmaker's MIDI `arzt`
        # (window of 30 events, 5 at the start, step size 5, cosine distance).
        self.inner = OnlineTimeWarpingArztEvent(reference_features=features_, score_positions=positions)
        self.tempo_window = config["tempo_window"]
        self.extrapolation_cap = config["extrapolation_cap"]
        self.max_speed = config["max_speed"]
        self.chord_notes = []
        self.chord_time = None
        self.chord_times = []  # onset time of every chord given to the follower
        self.chord_beats = []  # the follower's position after that chord
        self.beat = float(positions[0])
        self.latency_stats = {"total_latency": 0.0, "total_frames": 0}
        self.calls = 0

    def _flush_chord(self):
        vector = np.zeros(88, dtype=np.float32)
        for pitch in self.chord_notes:
            key = pitch - LOWEST_KEY
            if 0 <= key < 88:
                vector[key] = 1.0
        self.inner.step(vector)
        self.chord_times.append(self.chord_time)
        self.chord_beats.append(float(self.inner.get_current_position()))
        self.chord_notes, self.chord_time = [], None

    def _beat_at(self, t):
        """The follower's position at time t, as a step function over chords."""
        k = int(np.searchsorted(self.chord_times, t, side="right")) - 1
        return self.chord_beats[max(k, 0)]

    def _speed(self):
        t_k, b_k = self.chord_times[-1], self.chord_beats[-1]
        span = min(self.tempo_window, t_k - self.chord_times[0])
        if span <= 0:
            return 0.0
        return float(np.clip((b_k - self._beat_at(t_k - span)) / span, 0.0, self.max_speed))

    def step(self, features_) -> None:
        started = time.perf_counter()
        horizon = float(features_[0, 0])
        for onset, pitch in features_[1:]:
            if self.chord_time is not None and onset - self.chord_time > CHORD_THRESHOLD:
                self._flush_chord()
            self.chord_notes.append(int(pitch))
            if self.chord_time is None:
                self.chord_time = float(onset)
        if self.chord_time is not None and horizon - self.chord_time > CHORD_THRESHOLD:
            self._flush_chord()

        if self.chord_times:
            t_k, b_k = self.chord_times[-1], self.chord_beats[-1]
            elapsed = min(max(self.current_perf_time - t_k, 0.0), self.extrapolation_cap)
            self.beat = b_k + self._speed() * elapsed
            self.current_index = max(int(np.searchsorted(self.score_positions, b_k, side="right")) - 1, 0)

        self.calls += 1
        self.latency_stats["total_latency"] += time.perf_counter() - started
        self.latency_stats["total_frames"] = self.calls

    def get_current_position(self) -> float:
        return self.beat


class DualFollower(TranscriptionFollower):
    """The follower above on the anchor's onsets, the fast onsets time the chords after it."""

    def __init__(self, score_note_array, queue, config):
        super().__init__(score_note_array, queue, config)
        self.fast_times = []  # start of every fast chord (grouped at its first note)

    def step(self, features_) -> None:
        anchor, fast = features_
        super().step(anchor)
        started = time.perf_counter()
        for onset in fast[1:, 0]:
            if not self.fast_times or onset - self.fast_times[-1] > CHORD_THRESHOLD:
                self.fast_times.append(float(onset))
        if self.chord_times:
            t_k, b_k = self.chord_times[-1], self.chord_beats[-1]
            new = len(self.fast_times) - bisect.bisect_right(self.fast_times, t_k + CHORD_THRESHOLD)
            if new > 0:
                k = max(int(np.searchsorted(self.score_positions, b_k, side="right")) - 1, 0)
                target = float(self.score_positions[min(k + new, len(self.score_positions) - 1)])
                elapsed = min(max(self.current_perf_time - self.fast_times[-1], 0.0), self.extrapolation_cap)
                self.beat = target + self._speed() * elapsed
        self.latency_stats["total_latency"] += time.perf_counter() - started


def build_processor(mm):
    return DualProcessor(mm.sample_rate, mm.hop_length, mm.config["anchor_threshold"], mm.config["fast_threshold"])


def build_reference(mm):
    """The score's notes, as Matchmaker uses them for MIDI followers."""
    return mm.score_part.note_array()


def build_follower(mm):
    return DualFollower(mm.reference_features, mm.stream.queue, mm.config)


register_method(
    "bjork-gudmundsdottir-dual",
    input_type="audio",
    build_follower=build_follower,
    build_processor=build_processor,
    build_reference=build_reference,
    default_kwargs={
        # chosen on the MAESTRO validation split, never on the benchmark folds
        "anchor_threshold": float(SETTINGS["anchor"]["onset_threshold"]),
        "fast_threshold": float(SETTINGS["fast"]["onset_threshold"]),
        # moving on between chords, as in bjork-gudmundsdottir
        "tempo_window": 4.0,  # seconds of the follower's path the tempo is read from
        "extrapolation_cap": 1.5,  # seconds after a chord beyond which the position stops moving
        "max_speed": 8.0,  # beats per second
    },
)
