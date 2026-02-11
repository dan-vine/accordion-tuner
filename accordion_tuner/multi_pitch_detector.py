"""
Multi-pitch detection using FFT + phase vocoder (matching C++ implementation)

This module provides multi-note detection by analyzing the FFT spectrum
and finding multiple peaks, similar to the C++ implementation.
"""

from dataclasses import dataclass, field

import numpy as np

from .constants import (
    A4_REFERENCE,
    A_OFFSET,
    C5_OFFSET,
    NOTE_NAMES,
    OCTAVE,
    SAMPLE_RATE,
)
from .temperaments import TEMPERAMENT_RATIOS, Temperament

# Constants matching C++ implementation
K_SAMPLES = 16384  # FFT size
K_LOG2_SAMPLES = 14
K_SAMPLES2 = K_SAMPLES // 2
K_RANGE = K_SAMPLES * 7 // 16  # Frequency range to analyze
K_STEP = 1024  # Hop size (BUFFER_SIZE)
K_OVERSAMPLE = K_SAMPLES // K_STEP
K_MAXIMA = 8  # Maximum number of simultaneous notes
K_MIN = 0.5  # Minimum magnitude threshold
K_SCALE = 2048.0


@dataclass
class Maximum:
    """Detected peak in the spectrum"""
    frequency: float = 0.0
    ref_frequency: float = 0.0
    note: int = 0
    cents: float = 0.0
    note_name: str = ""
    octave: int = 0
    magnitude: float = 0.0


@dataclass
class MultiPitchResult:
    """Result of multi-pitch detection"""
    maxima: list[Maximum] = field(default_factory=list)
    primary_frequency: float = 0.0
    primary_note: int = 0
    primary_cents: float = 0.0
    valid: bool = False

    @property
    def note_count(self) -> int:
        return len(self.maxima)

    @property
    def frequencies(self) -> list[float]:
        return [m.frequency for m in self.maxima]

    @property
    def note_names(self) -> list[str]:
        return [f"{m.note_name}{m.octave}" for m in self.maxima]


class MultiPitchDetector:
    """
    Multi-pitch detector using FFT + phase vocoder.

    This implementation matches the C++ algorithm for detecting
    multiple simultaneous notes in the spectrum.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        fft_size: int = K_SAMPLES,
        hop_size: int = K_STEP,
        reference: float = A4_REFERENCE,
    ):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.reference = reference
        self.temperament = Temperament.EQUAL
        self.key = 0

        # Calculate derived constants
        self.range = fft_size * 7 // 16
        self.oversample = fft_size // hop_size
        self.fps = sample_rate / fft_size  # Frequency per bin
        self.expect = 2.0 * np.pi * hop_size / fft_size

        # State for phase vocoder
        self._buffer = np.zeros(fft_size, dtype=np.float64)
        self._prev_phase = np.zeros(self.range, dtype=np.float64)
        self._window = np.hamming(fft_size)
        self._dmax = 0.125  # Dynamic range normalization

        # Filters
        self.fundamental_filter = False  # Only detect harmonics of fundamental
        self.downsample = False  # Downsample spectrum for low notes
        self.octave_filter = True  # Limit search to one octave above fundamental (C++ behavior)

        # Detection threshold (can be lowered for quiet inputs)
        self.min_magnitude = K_MIN

    def process(self, samples: np.ndarray) -> MultiPitchResult:
        """
        Process audio samples and detect multiple pitches.

        Args:
            samples: Audio samples (will be added to internal buffer)

        Returns:
            MultiPitchResult with detected notes
        """
        # Ensure correct dtype
        if samples.dtype != np.float64:
            samples = samples.astype(np.float64)

        # Shift buffer and add new samples
        shift = min(len(samples), self.fft_size)
        self._buffer = np.roll(self._buffer, -shift)

        if len(samples) >= self.fft_size:
            self._buffer[:] = samples[-self.fft_size:]
        else:
            self._buffer[-len(samples):] = samples

        # Normalize
        dmax = np.max(np.abs(self._buffer))
        if dmax < 0.125:
            dmax = 0.125

        norm = self._dmax
        self._dmax = dmax

        input_signal = self._buffer / norm

        # Apply window
        windowed = input_signal * self._window

        # FFT
        spectrum = np.fft.rfft(windowed)

        # Zero DC
        spectrum[0] = 0

        # Scale
        spectrum = spectrum / K_SCALE

        # Magnitude and phase
        xa = np.abs(spectrum[:self.range])
        xq = np.angle(spectrum[:self.range])

        # Phase difference
        dxp = xq - self._prev_phase

        # Calculate frequencies using phase vocoder
        xf = np.zeros(self.range, dtype=np.float64)
        dxa = np.zeros(self.range, dtype=np.float64)

        for i in range(1, self.range):
            dp = dxp[i]
            dp -= i * self.expect

            # Unwrap phase
            qpd = int(dp / np.pi)
            if qpd >= 0:
                qpd += qpd & 1
            else:
                qpd -= qpd & 1
            dp -= np.pi * qpd

            # Frequency correction
            df = self.oversample * dp / (2.0 * np.pi)
            xf[i] = i * self.fps + df * self.fps

            # Difference for peak detection
            dxa[i] = xa[i] - xa[i - 1]

        # Save phase for next iteration
        self._prev_phase = xq.copy()

        # Find maximum
        max_val = np.max(xa)
        if max_val < self.min_magnitude:
            return MultiPitchResult()

        # Find maxima (peaks in spectrum)
        maxima = []
        limit = self.range - 1

        # Note: Use range with full extent and check limit dynamically,
        # since limit can decrease during iteration (unlike C where loop
        # condition is checked each iteration)
        for i in range(1, self.range - 1):
            if i >= limit:
                break
            if len(maxima) >= K_MAXIMA:
                break

            # Skip if below threshold
            if xa[i] <= self.min_magnitude or xa[i] <= max_val / 4:
                continue

            # Peak detection: rising then falling
            if i >= len(dxa) - 1:
                continue
            if dxa[i] <= 0.0 or dxa[i + 1] >= 0.0:
                continue

            # Calculate note from frequency
            if xf[i] <= 0:
                continue

            cf = -12.0 * np.log2(self.reference / xf[i])
            if np.isnan(cf):
                continue

            note = int(round(cf)) + C5_OFFSET
            if note < 0:
                continue

            # Fundamental filter: only allow harmonics of first detected note
            if self.fundamental_filter and len(maxima) > 0:
                if (note % OCTAVE) != (maxima[0].note % OCTAVE):
                    continue

            # Calculate reference frequency with temperament
            ref_freq = self._get_reference_frequency(note)

            # Calculate cents
            cents = 1200 * np.log2(xf[i] / ref_freq) if ref_freq > 0 else 0.0

            maximum = Maximum(
                frequency=xf[i],
                ref_frequency=ref_freq,
                note=note,
                cents=cents,
                note_name=NOTE_NAMES[note % OCTAVE],
                octave=note // OCTAVE,
                magnitude=xa[i],
            )
            maxima.append(maximum)

            # Limit search to avoid harmonics (one octave above fundamental)
            # When octave_filter is True (default), matches C++ behavior
            # When False, allows detecting notes across multiple octaves (e.g., octave pairs)
            if self.octave_filter and not self.downsample and limit > i * 2:
                limit = i * 2 - 1

        if not maxima:
            return MultiPitchResult()

        # Primary note is the first (lowest frequency) maximum
        primary = maxima[0]

        return MultiPitchResult(
            maxima=maxima,
            primary_frequency=primary.frequency,
            primary_note=primary.note,
            primary_cents=primary.cents,
            valid=True,
        )

    def _get_reference_frequency(self, note: int) -> float:
        """Calculate temperament-adjusted reference frequency for a note."""
        note_in_octave = note % OCTAVE

        # Get temperament ratios
        ratios = TEMPERAMENT_RATIOS[self.temperament]
        equal_ratios = TEMPERAMENT_RATIOS[Temperament.EQUAL]

        # Adjust for key
        n = (note_in_octave - self.key + OCTAVE) % OCTAVE
        a = (A_OFFSET - self.key + OCTAVE) % OCTAVE

        # Calculate temperament adjustment
        temper_ratio = ratios[n] / ratios[a]
        equal_ratio = equal_ratios[n] / equal_ratios[a]
        temper_adjust = temper_ratio / equal_ratio

        # Calculate equal temperament frequency
        semitones_from_a4 = note - C5_OFFSET
        equal_freq = self.reference * (2 ** (semitones_from_a4 / 12))

        return equal_freq * temper_adjust

    def set_reference(self, freq: float):
        """Set reference frequency for A4."""
        self.reference = freq

    def set_temperament(self, temperament: Temperament):
        """Set musical temperament."""
        self.temperament = temperament

    def set_key(self, key: int):
        """Set key for temperament (0=C, 1=C#, ..., 11=B)."""
        self.key = key % OCTAVE

    def set_fundamental_filter(self, enabled: bool):
        """Enable/disable fundamental filter (only detect harmonics)."""
        self.fundamental_filter = enabled

    def set_downsample(self, enabled: bool):
        """Enable/disable downsampling for low frequency detection."""
        self.downsample = enabled

    def set_octave_filter(self, enabled: bool):
        """Enable/disable octave filter.

        When enabled (default, C++ behavior): limits search to one octave above
        the fundamental frequency, preventing detection of octave pairs and harmonics.

        When disabled: allows detecting notes across multiple octaves, useful for
        detecting octave pairs like A3+A4.
        """
        self.octave_filter = enabled

    def set_min_magnitude(self, threshold: float):
        """Set minimum magnitude threshold for peak detection.

        Lower values increase sensitivity but may detect more noise.
        Default is 0.5 (matching C++ implementation).
        For quiet microphone input, try 0.1 or lower.
        """
        self.min_magnitude = max(0.01, threshold)

    def reset(self):
        """Reset internal state."""
        self._buffer.fill(0)
        self._prev_phase.fill(0)
        self._dmax = 0.125
