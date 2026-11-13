"""Frame-by-frame version of madmom's 2019 piano note CNN (Kelz, Böck and Widmer).

madmom computes the note activations of a whole recording at once. A live
follower gets the audio a little at a time, so this module computes the same
activations one spectrogram frame at a time:

- ``StreamingSpectrogram``: the log-filtered spectrogram madmom uses, one
  frame per call (4096 samples, Hann window, 24 bands per octave from 30 Hz to
  10 kHz, log10(x + 1)). Frame k is centred on sample 882 * k, so it can be
  computed once the audio up to sample 882 * k + 2048 has arrived.
- ``StreamingNoteCNN``: the network, layer by layer. No layer pools over
  time, so every layer only needs to compute its newest time step. Each layer
  keeps the few most recent outputs of the layer before it (3 frames for the
  first convolution, 7 for the third, 3 for the output heads). That is 11
  frames of context in total, 5 of them in the future, so output frame k is
  ready once input frame k + 5 has arrived.

The weights, the window and the filterbank come either from madmom itself
(``from_madmom``) or from an npz file exported once with ``export_npz``, so a
submission needs nothing but numpy. On the 20 recordings of the validation
fold, the activations differ from madmom's offline output by at most 6.2e-6.

The weights of the network are madmom's (CC BY-NC-SA 4.0). This file only
reads them.
"""

import numpy as np

SAMPLE_RATE = 44100
FPS = 50
HOP = SAMPLE_RATE // FPS  # 882 samples
FRAME = 4096
CONTEXT = 5  # future frames the network needs for one output frame


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class StreamingSpectrogram:
    """madmom's spectrogram for the note CNN, one frame at a time."""

    def __init__(self, window, filterbank):
        self.window = np.asarray(window, dtype=np.float32)
        self.filterbank = np.asarray(filterbank, dtype=np.float32)
        self.num_bins = self.filterbank.shape[0]

    @classmethod
    def from_madmom(cls):
        from madmom.audio.signal import FramedSignalProcessor, Signal
        from madmom.audio.spectrogram import FilteredSpectrogramProcessor
        from madmom.audio.stft import ShortTimeFourierTransformProcessor

        # Build the madmom objects once on a short signal and keep their window
        # and filterbank, so the numbers are exactly madmom's.
        dummy = Signal(np.zeros(FRAME * 2, dtype=np.float32), sample_rate=SAMPLE_RATE, num_channels=1)
        frames = FramedSignalProcessor(frame_size=FRAME, fps=FPS)(dummy)
        stft = ShortTimeFourierTransformProcessor()(frames)
        filtered = FilteredSpectrogramProcessor(num_bands=24, fmin=30, fmax=10000)(stft)
        return cls(stft.window, filtered.filterbank)

    def frame(self, samples):
        """Spectrogram of one frame of FRAME samples (zero-padded where the recording has none)."""
        spectrum = np.fft.rfft(samples.astype(np.float32) * self.window)[: self.num_bins]
        return np.log10(np.abs(spectrum).astype(np.float32) @ self.filterbank + 1.0).astype(np.float32)

    def frames_of(self, audio):
        """All frames of a recording, the way a stream would produce them (for tests)."""
        audio = np.asarray(audio, dtype=np.float32)
        n_frames = int(np.ceil(len(audio) / HOP))
        padded = np.concatenate([np.zeros(FRAME // 2, np.float32), audio, np.zeros(FRAME, np.float32)])
        return np.stack([self.frame(padded[k * HOP : k * HOP + FRAME]) for k in range(n_frames)])


def _conv_step(window, weights, bias):
    """Newest time step of a 'valid' 2-D convolution, as madmom computes it.

    window:  (kt, F, C_in), the kt most recent input time steps
    weights: (C_in, C_out, kt, kf), madmom's layout. madmom convolves (the
             kernel is flipped), so the kernel is flipped here as well.
    returns: (F - kf + 1, C_out)
    """
    c_in, c_out, kt, kf = weights.shape
    patches = np.lib.stride_tricks.sliding_window_view(window, kf, axis=1)  # (kt, F', C_in, kf)
    patches = patches.transpose(1, 2, 0, 3).reshape(patches.shape[1], c_in * kt * kf)
    kernel = weights[:, :, ::-1, ::-1].transpose(0, 2, 3, 1).reshape(c_in * kt * kf, c_out)
    return patches @ kernel + bias


class StreamingNoteCNN:
    """madmom's NOTES_CNN, computed one frame at a time.

    Call ``push(spectrogram_frame)`` for every new frame. It returns the
    activations of the frame CONTEXT frames earlier, shape (88, 3) with the
    sounding-note, onset and offset activation of each key, or None while the
    network is still filling up. The first frames are padded by repeating
    frame 0, as madmom does. ``flush()`` pads the end by repeating the last
    frame and returns the outstanding outputs, which is what madmom does at the
    end of a file (a live follower never needs it).
    """

    def __init__(self, bn, convs, heads):
        self.bn = tuple(np.float32(v) for v in bn)  # mean, gamma * inv_std, beta
        self.convs = [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in convs]
        self.heads = [tuple(np.asarray(a, np.float32) for a in head) for head in heads]
        self.reset()

    @classmethod
    def from_madmom(cls):
        from madmom.ml.nn import NeuralNetworkEnsemble
        from madmom.models import NOTES_CNN

        if len(NOTES_CNN) != 1:
            raise RuntimeError("expected a single network in NOTES_CNN")
        ensemble = NeuralNetworkEnsemble.load(NOTES_CNN)
        layers = ensemble.processors[0].processors[0].processors
        bn, c1, c2, c3, heads = layers[0], layers[1], layers[2], layers[3], layers[4].processors
        if any(getattr(c, "activation_fn", None).__name__ != "elu" for c in (c1, c2, c3)):
            raise RuntimeError("unexpected activation")
        head_arrays = []
        for head in heads:
            conv, transpose, reshape, dense = head.processors
            if tuple(transpose.axes) != (0, 2, 1) or tuple(reshape.newshape) != (-1, 1060):
                raise RuntimeError("unexpected head layout")
            head_arrays.append((conv.weights, conv.bias, dense.weights, dense.bias))
        return cls(
            (bn.mean, bn.gamma * bn.inv_std, bn.beta),
            [(c.weights, c.bias) for c in (c1, c2, c3)],
            head_arrays,
        )

    def reset(self):
        self.inputs = []  # normalised spectrogram frames, last 3
        self.mid = []  # outputs of the second convolution, last 7
        self.top = []  # outputs of the third convolution, last 3
        self.started = False
        self.last = None

    def _advance(self, frame):
        mean, scale, beta = self.bn
        self.inputs.append(((frame - mean) * scale + beta)[:, None])  # (F, 1)
        self.inputs = self.inputs[-3:]
        if len(self.inputs) < 3:
            return None
        w, b = self.convs[0]
        x = _elu(_conv_step(np.stack(self.inputs), w, b))  # (142, 30)
        w, b = self.convs[1]
        x = _elu(_conv_step(x[None], w, b))  # (108, 30)
        self.mid = (self.mid + [x])[-7:]
        if len(self.mid) < 7:
            return None
        w, b = self.convs[2]
        x = _elu(_conv_step(np.stack(self.mid), w, b))  # (108, 30)
        self.top = (self.top + [x])[-3:]
        if len(self.top) < 3:
            return None
        window = np.stack(self.top)
        out = np.empty((88, len(self.heads)), np.float32)
        for i, (cw, cb, dw, db) in enumerate(self.heads):
            h = _elu(_conv_step(window, cw, cb))  # (106, 10)
            out[:, i] = _sigmoid(h.T.reshape(-1) @ dw + db)  # madmom: (time, freq, ch) -> (time, ch, freq) -> 1060
        return out

    def push(self, frame):
        frame = np.asarray(frame, np.float32)
        self.last = frame
        if not self.started:
            self.started = True
            for _ in range(CONTEXT):
                self._advance(frame)
        return self._advance(frame)

    def flush(self):
        return [self._advance(self.last) for _ in range(CONTEXT)]


def export_npz(path):
    """Write madmom's window, filterbank and network weights to one npz file."""
    spec = StreamingSpectrogram.from_madmom()
    cnn = StreamingNoteCNN.from_madmom()
    arrays = {"window": spec.window, "filterbank": spec.filterbank,
              "bn": np.asarray(cnn.bn, np.float32)}
    for i, (w, b) in enumerate(cnn.convs):
        arrays[f"conv{i}_w"], arrays[f"conv{i}_b"] = w, b
    for i, (cw, cb, dw, db) in enumerate(cnn.heads):
        arrays[f"head{i}_conv_w"], arrays[f"head{i}_conv_b"] = cw, cb
        arrays[f"head{i}_dense_w"], arrays[f"head{i}_dense_b"] = dw, db
    np.savez_compressed(path, **arrays)


def load_npz(path):
    """(StreamingSpectrogram, StreamingNoteCNN) from a file written by export_npz."""
    z = np.load(path)
    spec = StreamingSpectrogram(z["window"], z["filterbank"])
    convs = [(z[f"conv{i}_w"], z[f"conv{i}_b"]) for i in range(3)]
    heads = [(z[f"head{i}_conv_w"], z[f"head{i}_conv_b"], z[f"head{i}_dense_w"], z[f"head{i}_dense_b"])
             for i in range(3)]
    return spec, StreamingNoteCNN(tuple(z["bn"]), convs, heads)
