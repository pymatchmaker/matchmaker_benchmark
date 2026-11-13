"""Team Björk Guðmundsdóttir (Music Informatics WS26, JKU Linz): transcribe, then follow.

The audio is turned into notes while it arrives, and the notes are followed
in the score:

1. Processor: madmom's piano note CNN (Kelz, Böck and Widmer, ICASSP 2019),
   computed one spectrogram frame at a time (stream_cnn.py, weights in
   notes_cnn_2019.npz). Onsets are found with a causal rule: the onset
   activation of a key is at least 0.5, higher than one frame earlier and at
   least as high as one frame later, and the key has not started a note in
   the last 50 ms. Every onset keeps the time of the frame it belongs to.
2. Follower: onsets are grouped into chords the way Matchmaker's
   ChordOnsetProcessor does it (35 ms from the first note of a chord). Each
   chord goes to Matchmaker's event-level online time warping
   (OnlineTimeWarpingArztEvent, the follower behind Matchmaker's MIDI `arzt`).
3. Reported position: the position of the last chord, moved on at the
   followed tempo from that chord's own onset time. The transcriber hears a
   note about 170 ms after it is played, so without this the position would
   always be one note behind.

stream_cnn.py reimplements madmom's network in numpy so it can run one frame
at a time. notes_cnn_2019.npz holds madmom's weights (CC BY-NC-SA 4.0), see
README.md in this directory.
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
from matchmaker import register_method
from matchmaker.base import OnlineAlignment
from matchmaker.dp import OnlineTimeWarpingArztEvent
from matchmaker.features.midi import onset_pianoroll
from matchmaker.features.processor import Processor

HERE = Path(__file__).resolve().parent


def _load_helper(name):
    """Import a helper file from this directory (the benchmark does not put it on sys.path)."""
    spec = importlib.util.spec_from_file_location(f"{__name__}_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sc = _load_helper("stream_cnn")

ONSET_THRESHOLD = 0.5  # onset activation a note must reach
MIN_GAP_FRAMES = 2  # 50 ms at 50 frames per second before a key can start again
CHORD_THRESHOLD = 0.035  # seconds, Matchmaker's ChordOnsetProcessor
LOWEST_KEY = 21  # MIDI pitch of the first of the 88 keys


class TranscriptionProcessor(Processor):
    """Audio in, note onsets out.

    Returns, for every block of audio, an array whose first row is
    (horizon, -1) and whose other rows are (onset time, MIDI pitch) of the
    onsets found in this block. Onsets up to the horizon (seconds) are final.
    """

    def __init__(self, sample_rate, hop_length):
        super().__init__()
        if sample_rate != sc.SAMPLE_RATE:
            raise ValueError(f"the transcriber needs {sc.SAMPLE_RATE} Hz audio, got {sample_rate}")
        self.hop_length = hop_length
        # The stream hands over n_fft samples per block: the newest hop and the
        # one before it. Only the newest hop is new, so only that is used.
        self.n_fft = 2 * hop_length
        self.spectrogram, self.cnn = sc.load_npz(HERE / "notes_cnn_2019.npz")
        self.reset()

    def reset(self):
        self.cnn.reset()
        half = sc.FRAME // 2
        # Spectrogram frame k is centred on sample HOP * k, so frame 0 needs
        # half a frame of zeros before the recording starts, as in madmom.
        self.audio = np.zeros(half, np.float32)
        self.audio_start = -half  # sample index of self.audio[0]
        self.received = 0
        self.next_frame = 0
        self.act_index = -1  # index of the newest activation frame
        self.onset_before = None  # onset activations of frame t - 1
        self.onset_now = None  # onset activations of frame t
        self.last_onset = np.full(88, -(10**9))

    def _onsets(self, onset_next):
        """Onsets of frame t, decided now that frame t + 1 is known."""
        found = []
        if self.onset_before is not None:
            t = self.act_index - 1
            keys = np.flatnonzero(
                (self.onset_now >= ONSET_THRESHOLD)
                & (self.onset_now > self.onset_before)
                & (self.onset_now >= onset_next)
                & (t - self.last_onset >= MIN_GAP_FRAMES)
            )
            self.last_onset[keys] = t
            found = [(t / sc.FPS, float(key + LOWEST_KEY)) for key in keys]
        self.onset_before, self.onset_now = self.onset_now, onset_next
        return found

    def __call__(self, data):
        samples, perf_time = data
        new = np.asarray(samples[-self.hop_length :], dtype=np.float32)
        self.audio = np.concatenate((self.audio, new))
        self.received += len(new)

        onsets = []
        half = sc.FRAME // 2
        while self.next_frame * sc.HOP + half <= self.received:
            start = self.next_frame * sc.HOP - half - self.audio_start
            act = self.cnn.push(self.spectrogram.frame(self.audio[start : start + sc.FRAME]))
            self.next_frame += 1
            if act is not None:
                self.act_index += 1
                onsets.extend(self._onsets(act[:, 1]))

        keep_from = self.next_frame * sc.HOP - half  # first sample the next frame needs
        if keep_from > self.audio_start:
            self.audio = self.audio[keep_from - self.audio_start :]
            self.audio_start = keep_from

        horizon = (self.act_index - 1) / sc.FPS
        rows = [(horizon, -1.0)] + onsets
        return np.asarray(rows, dtype=np.float64), perf_time


class TranscriptionFollower(OnlineAlignment):
    def __init__(self, score_note_array, queue, config):
        features, positions = onset_pianoroll(score_note_array, onset_key="onset_beat", piano_range=True)
        super().__init__(features, positions, queue)
        # The same follower and settings as Matchmaker's MIDI `arzt`
        # (window of 30 events, 5 at the start, step size 5, cosine distance).
        self.inner = OnlineTimeWarpingArztEvent(reference_features=features, score_positions=positions)
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

    def step(self, features) -> None:
        started = time.perf_counter()
        horizon = float(features[0, 0])
        for onset, pitch in features[1:]:
            if self.chord_time is not None and onset - self.chord_time > CHORD_THRESHOLD:
                self._flush_chord()
            self.chord_notes.append(int(pitch))
            if self.chord_time is None:
                self.chord_time = float(onset)
        if self.chord_time is not None and horizon - self.chord_time > CHORD_THRESHOLD:
            self._flush_chord()

        if self.chord_times:
            t_k, b_k = self.chord_times[-1], self.chord_beats[-1]
            span = min(self.tempo_window, t_k - self.chord_times[0])
            speed = 0.0
            if span > 0:
                speed = float(np.clip((b_k - self._beat_at(t_k - span)) / span, 0.0, self.max_speed))
            elapsed = min(max(self.current_perf_time - t_k, 0.0), self.extrapolation_cap)
            self.beat = b_k + speed * elapsed
            self.current_index = max(int(np.searchsorted(self.score_positions, b_k, side="right")) - 1, 0)

        self.calls += 1
        self.latency_stats["total_latency"] += time.perf_counter() - started
        self.latency_stats["total_frames"] = self.calls

    def get_current_position(self) -> float:
        return self.beat


def build_processor(mm):
    return TranscriptionProcessor(mm.sample_rate, mm.hop_length)


def build_reference(mm):
    """The score's notes, as Matchmaker uses them for MIDI followers."""
    return mm.score_part.note_array()


def build_follower(mm):
    return TranscriptionFollower(mm.reference_features, mm.stream.queue, mm.config)


register_method(
    "bjork-gudmundsdottir",
    input_type="audio",
    build_follower=build_follower,
    build_processor=build_processor,
    build_reference=build_reference,
    default_kwargs={
        # moving on between chords
        "tempo_window": 4.0,  # seconds of the follower's path the tempo is read from
        "extrapolation_cap": 1.5,  # seconds after a chord beyond which the position stops moving
        "max_speed": 8.0,  # beats per second
    },
)
