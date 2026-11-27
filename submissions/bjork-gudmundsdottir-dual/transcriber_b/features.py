"""Log-filtered spectrogram at 50 frames per second, in numpy and in PyTorch.

The same features madmom computes for its 2019 note CNN, written here from
the definition so that nothing of madmom (code or data) is needed:

- frames of 4096 samples, Hann window, hop 882 samples (50 fps at 44.1 kHz),
  frame k centred on sample 882 k, zeros outside the recording;
- magnitude of the first 2048 FFT bins;
- triangular filters on a logarithmic frequency axis, 24 bands per octave from
  30 Hz to 10 kHz, centres on the grid 440 Hz * 2^(n / 24), each mapped to the
  nearest FFT bin, duplicates removed, every filter normalised to area 1
  (144 bands);
- log10(1 + x).

``tests/test_features.py`` checks the filterbank and window against the ones
stored in ``transcriber/notes_cnn_2019.npz`` when that file is present.
"""

import numpy as np

SAMPLE_RATE = 44100
FPS = 50
HOP = SAMPLE_RATE // FPS  # 882
FRAME = 4096
NUM_BINS = FRAME // 2
BANDS_PER_OCTAVE = 24
FMIN, FMAX, FREF = 30.0, 10000.0, 440.0


def log_frequencies(bands_per_octave=BANDS_PER_OCTAVE, fmin=FMIN, fmax=FMAX, fref=FREF):
    left = np.floor(np.log2(fmin / fref) * bands_per_octave)
    right = np.ceil(np.log2(fmax / fref) * bands_per_octave)
    freqs = fref * 2.0 ** (np.arange(left, right) / bands_per_octave)
    return freqs[(freqs >= fmin) & (freqs <= fmax)]


def frequencies_to_bins(freqs, bin_freqs):
    """Nearest FFT bin of every frequency, duplicates removed."""
    idx = np.clip(bin_freqs.searchsorted(freqs), 1, len(bin_freqs) - 1)
    left, right = bin_freqs[idx - 1], bin_freqs[idx]
    idx = idx - (freqs - left < right - freqs)
    return np.unique(idx)


def filterbank(frame=FRAME, sample_rate=SAMPLE_RATE):
    """(NUM_BINS, num_bands) float32 triangular filterbank."""
    num_bins = frame // 2
    bin_freqs = np.fft.fftfreq(frame, 1.0 / sample_rate)[:num_bins]
    bins = frequencies_to_bins(log_frequencies(), bin_freqs)
    filters = []
    for start, center, stop in zip(bins[:-2], bins[1:-1], bins[2:]):
        if stop - start < 2:  # too narrow for a triangle: a single bin
            center, stop = start, start + 1
        tri = np.zeros(stop - start)
        c = center - start
        tri[:c] = np.linspace(0, 1, c, endpoint=False)
        tri[c:] = np.linspace(1, 0, stop - start - c, endpoint=False)
        tri /= tri.sum()
        band = np.zeros(num_bins)
        band[start:stop] = tri
        filters.append(band)
    return np.stack(filters, axis=1).astype(np.float32)


def window(frame=FRAME):
    return np.hanning(frame).astype(np.float32)


def num_frames(num_samples):
    return int(np.ceil(num_samples / HOP))


class Spectrogram:
    """numpy features: one frame at a time (streaming) or a whole recording."""

    def __init__(self, window_=None, filterbank_=None):
        self.window = window() if window_ is None else np.asarray(window_, np.float32)
        self.filterbank = filterbank() if filterbank_ is None else np.asarray(filterbank_, np.float32)
        self.num_bins = self.filterbank.shape[0]
        self.num_bands = self.filterbank.shape[1]

    def frame(self, samples):
        """Features of one frame of FRAME samples."""
        spectrum = np.fft.rfft(np.asarray(samples, np.float32) * self.window)[: self.num_bins]
        return np.log10(np.abs(spectrum).astype(np.float32) @ self.filterbank + 1.0).astype(np.float32)

    def frames_of(self, audio, chunk=512):
        """(num_frames, num_bands) for a whole recording, same numbers as frame()."""
        audio = np.asarray(audio, np.float32)
        n = num_frames(len(audio))
        padded = np.concatenate([np.zeros(FRAME // 2, np.float32), audio, np.zeros(FRAME, np.float32)])
        frames = np.lib.stride_tricks.sliding_window_view(padded, FRAME)[::HOP][:n]
        out = np.empty((n, self.num_bands), np.float32)
        for s in range(0, n, chunk):
            spec = np.fft.rfft(frames[s:s + chunk] * self.window, axis=1)[:, : self.num_bins]
            out[s:s + chunk] = np.log10(np.abs(spec).astype(np.float32) @ self.filterbank + 1.0)
        return out


class TorchSpectrogram:
    """The same features in PyTorch, for batches on the GPU during training."""

    def __init__(self, device="cpu"):
        import torch

        self.torch = torch
        self.window = torch.from_numpy(window()).to(device)
        self.filterbank = torch.from_numpy(filterbank()).to(device)

    def to(self, device):
        self.window = self.window.to(device)
        self.filterbank = self.filterbank.to(device)
        return self

    def __call__(self, padded, n_frames):
        """Features of a batch whose audio is already padded.

        padded: (B, L) float tensor whose sample FRAME // 2 is the centre of
        frame 0; it must hold at least (n_frames - 1) * HOP + FRAME samples.
        Returns (B, n_frames, num_bands).
        """
        frames = padded.unfold(-1, FRAME, HOP)[:, :n_frames]  # (B, n, FRAME)
        spec = self.torch.fft.rfft(frames.float() * self.window, dim=-1)[..., :NUM_BINS].abs()
        return self.torch.log10(spec @ self.filterbank + 1.0)


def pad_recording(audio):
    """Recording padded the way frames_of pads it, for TorchSpectrogram."""
    audio = np.asarray(audio, np.float32)
    return np.concatenate([np.zeros(FRAME // 2, np.float32), audio, np.zeros(FRAME, np.float32)]), num_frames(len(audio))
