"""
Simple FFT peak detector using scipy find_peaks.

This detector uses scipy's find_peaks function for robust peak detection
with zero-padding for higher frequency resolution.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from scipy.signal import find_peaks

from accordion_tuner.constants import (
    A4_REFERENCE,
    SAMPLE_RATE,
)
from accordion_tuner.multi_pitch_detector import Maximum, MultiPitchResult


class SimpleFftPeakDetector:
    """
    Simple FFT-based pitch detector using scipy find_peaks.

    This detector uses zero-padding for higher frequency resolution and
    scipy's find_peaks for robust peak detection. Good for detecting
    multiple closely-spaced frequencies when they appear as clear peaks
    in the spectrum.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        reference: float = A4_REFERENCE,
        fft_size: int = 16384,
        hop_size: int = 1024,
        num_sources: int = 4,
    ):
        """
        Initialize detector.

        Args:
            sample_rate: Audio sample rate in Hz
            reference: Reference frequency for A4 in Hz
            fft_size: FFT window size (larger = more resolution)
            hop_size: Hop size between frames
            num_sources: Maximum number of peaks to detect
        """
        self.sample_rate = sample_rate
        self.reference = reference
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.num_sources = num_sources

        self._buffer = np.zeros(fft_size, dtype=np.float64)
        self._window = np.hamming(fft_size)

        self.fundamental_filter = False
        self.octave_filter = True
        self._min_magnitude = 0.0

    def set_octave_filter(self, enabled: bool):
        self.octave_filter = enabled

    def set_fundamental_filter(self, enabled: bool):
        self.fundamental_filter = enabled

    def set_min_magnitude(self, magnitude: float):
        self._min_magnitude = magnitude

    def set_reference(self, reference: float):
        self.reference = reference

    def set_peak_threshold(self, threshold: float):
        pass

    def set_downsample(self, downsample: bool):
        pass

    def set_temperament(self, temperament):
        pass

    def set_key(self, key: int):
        pass

    def reset(self):
        self._buffer = np.zeros(self.fft_size, dtype=np.float64)

    def process(self, samples: np.ndarray) -> MultiPitchResult:
        """
        Process audio samples and detect multiple pitches.

        Args:
            samples: Audio samples

        Returns:
            MultiPitchResult with detected peaks
        """
        if samples.dtype != np.float64:
            samples = samples.astype(np.float64)

        shift = min(len(samples), self.fft_size)
        self._buffer = np.roll(self._buffer, -shift)

        if len(samples) >= self.fft_size:
            self._buffer[:] = samples[-self.fft_size:]
        else:
            self._buffer[-len(samples):] = samples

        signal = self._buffer * self._window

        n_fft = self.fft_size * 4
        spectrum = np.fft.rfft(signal, n=n_fft)
        mags = np.abs(spectrum)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)

        valid_mask = (freqs >= 60) & (freqs <= 2000)
        freqs_valid = freqs[valid_mask]
        mags_valid = mags[valid_mask]

        if len(mags_valid) == 0:
            return MultiPitchResult()

        # Use 25% threshold like FFT to filter out small noise peaks
        min_magnitude = np.max(mags_valid) * 0.25
        distance = 1

        peaks, properties = find_peaks(
            mags_valid,
            height=min_magnitude,
            distance=distance,
            prominence=min_magnitude * 0.3,
        )

        if len(peaks) == 0:
            return MultiPitchResult()

        peaks = np.sort(peaks)

        maxima = []
        first_note = None
        first_freq = None

        for peak_idx in peaks:
            if len(maxima) >= self.num_sources:
                break

            freq = freqs_valid[peak_idx]
            mag = mags_valid[peak_idx]

            if peak_idx > 0 and peak_idx < len(mags_valid) - 1:
                y1, y2, y3 = mags_valid[peak_idx - 1], mags_valid[peak_idx], mags_valid[peak_idx + 1]
                denom = y1 - 2 * y2 + y3
                if abs(denom) > 1e-10:
                    delta = 0.5 * (y1 - y3) / denom
                    freq_step = freqs_valid[1] - freqs_valid[0]
                    freq = freqs_valid[peak_idx] + delta * freq_step

            note, cents = self._frequency_to_note(freq)

            if first_note is None:
                first_note = note
                first_freq = freq
            else:
                if (note % 12) != (first_note % 12):
                    continue
                if freq > first_freq * 2:
                    continue

            note_name, octave = self._note_number_to_name(note)

            note_ref_freq = self.reference * (2 ** ((note - 69) / 12))

            maxima.append(
                Maximum(
                    frequency=freq,
                    ref_frequency=note_ref_freq,
                    note=note,
                    cents=cents,
                    note_name=note_name,
                    octave=octave,
                    magnitude=mag,
                )
            )

        primary = maxima[0] if maxima else Maximum()
        return MultiPitchResult(
            maxima=maxima,
            primary_frequency=primary.frequency,
            primary_note=primary.note,
            primary_cents=primary.cents,
            valid=len(maxima) > 0,
        )

    def _frequency_to_note(self, frequency: float) -> tuple[int, float]:
        """Convert frequency to note number and cents deviation."""
        if frequency <= 0:
            return 0, 0.0
        note = int(round(12.0 * np.log2(frequency / self.reference) + 69))
        cents = 1200.0 * np.log2(frequency / self.reference) - (note - 69) * 100.0
        return note, cents

    def _note_number_to_name(self, note: int) -> tuple[str, int]:
        """Convert note number to note name and octave."""
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = note // 12 - 1
        note_name = note_names[note % 12]
        return note_name, octave
