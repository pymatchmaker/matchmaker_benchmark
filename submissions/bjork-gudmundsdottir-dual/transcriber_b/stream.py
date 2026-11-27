"""numpy inference for our transcriber: offline and frame by frame.

Needs only numpy and the npz file written by ``export.py``. The interface
follows ``transcriber/stream_cnn.py``:

- ``Spectrogram`` (from features.py): ``frame(samples)`` gives one
  spectrogram frame from FRAME samples; frame k is centred on sample 882 k and
  can be computed once audio up to sample 882 k + 2048 has arrived.
- ``StreamingTranscriber.push(frame)`` takes spectrogram frame k and returns
  the (88, 3) activations (sounding, onset, offset) of frame k - LOOKAHEAD,
  or None while the network fills up. ``flush()`` returns the remaining
  frames at the end of a file (a live follower never needs it).
- ``OfflineTranscriber.activations(spec)`` computes a whole recording at once
  and gives the same numbers (tests/test_stream.py).

Edges: every convolution pads its input with zeros in time, in training and
here alike. The streaming version starts each layer with the same zero
columns and ``flush()`` feeds the same zero columns at the end, so streaming
and offline agree on every frame, the first and the last included.

Load both with ``load(path)``.
"""

import json

import numpy as np

from transcriber_b.features import Spectrogram


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # = 1 / (1 + exp(-x)), without overflow warnings


class Weights:
    """Arrays of an exported model, arranged for numpy."""

    def __init__(self, z):
        self.meta = json.loads(str(z["meta"]))
        cfg = self.cfg = self.meta["model"]
        self.in_scale, self.in_shift = z["input_scale"].astype(np.float32), z["input_shift"].astype(np.float32)
        self.convs = []
        for i in range(len(cfg["channels"])):
            w = z[f"conv{i}_w"].astype(np.float32)  # (Cout, Cin, 3, 3), batch norm folded in
            cout, cin = w.shape[:2]
            kernel = w.transpose(2, 3, 1, 0).reshape(9 * cin, cout)  # rows ordered (time, band, channel)
            self.convs.append({
                "kernel": np.ascontiguousarray(kernel), "bias": z[f"conv{i}_b"].astype(np.float32),
                "past": 2 - cfg["lookahead"][i], "future": cfg["lookahead"][i], "pool": i in cfg["pool_after"],
            })
        self.proj_w = np.ascontiguousarray(z["proj_w"].T.astype(np.float32))  # (flat, hidden)
        self.proj_b = z["proj_b"].astype(np.float32)
        self.rnn = cfg["rnn"]
        self.layers = []
        for i in range(cfg["rnn_layers"]):
            self.layers.append(tuple(z[f"rnn{i}_{k}"].astype(np.float32) for k in ("w_ih", "w_hh", "b_ih", "b_hh")))
        self.onset = (z["onset_w"].T.astype(np.float32), z["onset_b"].astype(np.float32))
        self.frame = (z["frame_w"].T.astype(np.float32), z["frame_b"].astype(np.float32))
        self.offset = (z["offset_w"].T.astype(np.float32), z["offset_b"].astype(np.float32))
        self.hidden = cfg["hidden"]
        self.lookahead = sum(cfg["lookahead"])


def _conv_patches(xpad, n_bands):
    """xpad: (t + 2, bands + 2, C) -> (t * bands, 9 C), rows ordered (time, band, channel)."""
    v = np.lib.stride_tricks.sliding_window_view(xpad, (3, 3), axis=(0, 1))  # (t, bands, C, 3, 3)
    t = v.shape[0]
    return v.transpose(0, 1, 3, 4, 2).reshape(t * n_bands, -1)


def _after_conv(y, conv):
    y = np.maximum(y, 0.0)
    if conv["pool"]:
        b = y.shape[-2] // 2 * 2
        y = y[..., :b, :].reshape(*y.shape[:-2], b // 2, 2, y.shape[-1]).max(axis=-2)
    return y


def _rnn_step_pre(kind, w_hh, b_hh, gi, state):
    """One recurrent step (PyTorch's GRU or LSTM equations) given gi = W_ih x + b_ih."""
    if kind == "gru":
        h = state
        gh = w_hh @ h + b_hh
        n3 = len(b_hh) // 3
        r = _sigmoid(gi[:n3] + gh[:n3])
        z = _sigmoid(gi[n3:2 * n3] + gh[n3:2 * n3])
        n = np.tanh(gi[2 * n3:] + r * gh[2 * n3:])
        h = (1.0 - z) * n + z * h
        return h, h
    h, c = state
    g = gi + w_hh @ h + b_hh
    n4 = len(b_hh) // 4
    i, f, gg, o = _sigmoid(g[:n4]), _sigmoid(g[n4:2 * n4]), np.tanh(g[2 * n4:3 * n4]), _sigmoid(g[3 * n4:])
    c = f * c + i * gg
    h = o * np.tanh(c)
    return h, (h, c)


def _rnn_step(kind, layer, x, state):
    w_ih, w_hh, b_ih, b_hh = layer
    return _rnn_step_pre(kind, w_hh, b_hh, w_ih @ x + b_ih, state)


def _heads(w, h):
    onset = _sigmoid(h @ w.onset[0] + w.onset[1])
    frame = _sigmoid(np.concatenate([h, onset], axis=-1) @ w.frame[0] + w.frame[1])
    offset = _sigmoid(h @ w.offset[0] + w.offset[1])
    return np.stack([frame, onset, offset], axis=-1).astype(np.float32)


def _initial_state(w):
    zero = np.zeros(w.hidden, np.float32)
    return [zero if w.rnn == "gru" else (zero, zero) for _ in w.layers]


class OfflineTranscriber:
    def __init__(self, weights):
        self.w = weights

    def conv_features(self, spec, chunk=1024):
        w = self.w
        x = (spec.astype(np.float32) * w.in_scale + w.in_shift)[:, :, None]  # (T, bands, 1)
        for conv in w.convs:
            t, bands, _ = x.shape
            xpad = np.pad(x, ((conv["past"], conv["future"]), (1, 1), (0, 0)))
            out = []
            for s in range(0, t, chunk):
                patches = _conv_patches(xpad[s:s + min(chunk, t - s) + 2], bands)
                y = (patches @ conv["kernel"] + conv["bias"]).reshape(-1, bands, conv["kernel"].shape[1])
                out.append(_after_conv(y, conv))
            x = np.concatenate(out)
        flat = x.transpose(0, 2, 1).reshape(x.shape[0], -1)  # channel-major, then band, as in PyTorch
        return np.maximum(flat @ w.proj_w + w.proj_b, 0.0)

    def activations(self, spec):
        """(T, bands) spectrogram -> (T, 88, 3) activations."""
        w = self.w
        x = self.conv_features(spec)
        for layer in w.layers:
            w_ih, w_hh, b_ih, b_hh = layer
            gi = x @ w_ih.T + b_ih  # input part for all frames at once
            state = _initial_state(w)[0]
            out = np.empty((len(x), w.hidden), np.float32)
            for t in range(len(x)):
                out[t], state = _rnn_step_pre(w.rnn, w_hh, b_hh, gi[t], state)
            x = out
        return _heads(w, x)

    def transcribe(self, audio, spectrogram=None):
        """44.1 kHz mono audio -> (frames, 88, 3) activations."""
        spectrogram = spectrogram or Spectrogram()
        return self.activations(spectrogram.frames_of(audio))


class _ConvStream:
    """One convolution layer, newest time step only."""

    def __init__(self, conv, bands_in, channels_in):
        self.conv = conv
        self.zero = np.zeros((bands_in, channels_in), np.float32)
        self.reset()

    def reset(self):
        self.buf = [self.zero] * self.conv["past"]

    def _compute(self, window):
        xpad = np.pad(np.stack(window), ((0, 0), (1, 1), (0, 0)))
        bands = xpad.shape[1] - 2
        y = (_conv_patches(xpad, bands) @ self.conv["kernel"] + self.conv["bias"])
        return _after_conv(y, self.conv)

    def push(self, x):
        self.buf.append(x)
        if len(self.buf) < 3:
            return None
        y = self._compute(self.buf[-3:])
        self.buf = self.buf[-2:]
        return y

    def finish(self):
        """Outputs that need the zero padding at the end of the file."""
        out = []
        for _ in range(self.conv["future"]):
            y = self.push(self.zero)
            if y is not None:
                out.append(y)
        return out


class StreamingTranscriber:
    def __init__(self, weights):
        self.w = w = weights
        self.lookahead = w.lookahead
        self.layers, bands, channels = [], w.cfg["n_bands"], 1
        for conv in w.convs:
            self.layers.append(_ConvStream(conv, bands, channels))
            channels = conv["kernel"].shape[1]
            bands = bands // 2 if conv["pool"] else bands
        self.reset()

    def reset(self):
        for layer in self.layers:
            layer.reset()
        self.state = _initial_state(self.w)

    def _tail(self, x):
        """Projection, recurrent step and heads for one frame of conv output (bands, C)."""
        w = self.w
        h = np.maximum(x.T.reshape(-1) @ w.proj_w + w.proj_b, 0.0)
        for i, layer in enumerate(w.layers):
            h, self.state[i] = _rnn_step(w.rnn, layer, h, self.state[i])
        return _heads(w, h)

    def push(self, frame):
        w = self.w
        x = (np.asarray(frame, np.float32) * w.in_scale + w.in_shift)[:, None]
        for layer in self.layers:
            x = layer.push(x)
            if x is None:
                return None
        return self._tail(x)

    def flush(self):
        pending = []
        for layer in self.layers:
            produced = []
            for x in pending:
                y = layer.push(x)
                if y is not None:
                    produced.append(y)
            produced.extend(layer.finish())
            pending = produced
        return [self._tail(x) for x in pending]


def load(path):
    """(Spectrogram, StreamingTranscriber, OfflineTranscriber) from an exported npz file."""
    with np.load(path) as z:
        weights = Weights(z)
    return Spectrogram(), StreamingTranscriber(weights), OfflineTranscriber(weights)
