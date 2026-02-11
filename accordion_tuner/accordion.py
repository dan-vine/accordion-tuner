"""
Accordion reed tuning detection.

This module provides detection of multiple detuned reeds playing the same note,
suitable for accordion tuning where 2-4 reeds may play simultaneously with
intentional detuning (tremolo/musette effects).
"""

from dataclasses import dataclass, field

import numpy as np

from .constants import (
    A4_REFERENCE,
    SAMPLE_RATE,
)
from .multi_pitch_detector import Maximum, MultiPitchDetector
from .temperaments import Temperament


@dataclass
class ReedInfo:
    """Information about a single detected reed."""
    frequency: float = 0.0      # Detected frequency in Hz
    cents: float = 0.0          # Deviation from reference in cents
    magnitude: float = 0.0      # Signal strength (for confidence)


@dataclass
class AccordionResult:
    """Result of accordion reed detection."""
    valid: bool = False
    note_name: str = ""         # e.g., "C"
    octave: int = 0             # e.g., 4
    ref_frequency: float = 0.0  # Reference frequency for this note
    reeds: list[ReedInfo] = field(default_factory=list)
    beat_frequencies: list[float] = field(default_factory=list)  # |f1-f2|, |f2-f3|, etc.
    spectrum_data: tuple[np.ndarray, np.ndarray] | None = None  # (frequencies, magnitudes)

    @property
    def reed_count(self) -> int:
        """Number of detected reeds."""
        return len(self.reeds)

    @property
    def average_cents(self) -> float:
        """Average cents deviation across all reeds."""
        if not self.reeds:
            return 0.0
        return sum(r.cents for r in self.reeds) / len(self.reeds)


class AccordionDetector:
    """
    Detector for accordion reed tuning.

    This detector finds multiple frequency peaks that correspond to the same
    musical note, typically 2-4 reeds tuned slightly apart for tremolo effects.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        reference: float = A4_REFERENCE,
        max_reeds: int = 4,
        reed_spread_cents: float = 50.0,
    ):
        """
        Initialize accordion detector.

        Args:
            sample_rate: Audio sample rate in Hz
            reference: Reference frequency for A4 in Hz
            max_reeds: Maximum number of reeds to detect (2-4)
            reed_spread_cents: Maximum cents spread to consider as same note
        """
        self.sample_rate = sample_rate
        self.reference = reference
        self.max_reeds = min(max(1, max_reeds), 4)
        self.reed_spread_cents = reed_spread_cents

        # Internal multi-pitch detector with accordion-specific settings
        self._detector = MultiPitchDetector(
            sample_rate=sample_rate,
            reference=reference,
        )
        # Disable octave filter to detect closely-spaced frequencies
        self._detector.set_octave_filter(False)
        # Lower threshold for typical microphone input levels
        self._detector.set_min_magnitude(0.1)

        # FFT state for spectrum display
        self._fft_freqs: np.ndarray | None = None
        self._fft_mags: np.ndarray | None = None

    def process(self, samples: np.ndarray) -> AccordionResult:
        """
        Process audio samples and detect accordion reeds.

        Args:
            samples: Audio samples as numpy array

        Returns:
            AccordionResult with detected reed information
        """
        # Ensure correct dtype
        if samples.dtype != np.float64:
            samples = samples.astype(np.float64)

        # Get multi-pitch detection result
        multi_result = self._detector.process(samples)

        # Compute spectrum for display
        self._compute_spectrum(samples)

        if not multi_result.valid or not multi_result.maxima:
            return AccordionResult(spectrum_data=self._get_spectrum_tuple())

        # Group peaks that correspond to the same note (within reed_spread_cents)
        reeds = self._group_reeds(multi_result.maxima)

        if not reeds:
            return AccordionResult(spectrum_data=self._get_spectrum_tuple())

        # Calculate beat frequencies between adjacent reeds
        beat_freqs = []
        for i in range(len(reeds) - 1):
            beat = abs(reeds[i].frequency - reeds[i + 1].frequency)
            beat_freqs.append(beat)

        # Primary note from the first (strongest) detection
        primary = multi_result.maxima[0]

        return AccordionResult(
            valid=True,
            note_name=primary.note_name,
            octave=primary.octave,
            ref_frequency=primary.ref_frequency,
            reeds=reeds,
            beat_frequencies=beat_freqs,
            spectrum_data=self._get_spectrum_tuple(),
        )

    def _group_reeds(self, maxima: list[Maximum]) -> list[ReedInfo]:
        """
        Group detected peaks into reeds for the same note.

        Args:
            maxima: List of detected frequency maxima

        Returns:
            List of ReedInfo for reeds detected as playing the same note
        """
        if not maxima:
            return []

        # Use the strongest peak as reference
        primary = maxima[0]
        primary_note = primary.note

        reeds = []
        for m in maxima:
            if len(reeds) >= self.max_reeds:
                break

            # Check if this peak is within the reed spread of the primary note
            # Allow same note or adjacent semitone if within cents spread
            note_diff = abs(m.note - primary_note)
            if note_diff > 1:
                continue

            # Calculate cents from primary reference frequency
            if primary.ref_frequency > 0:
                cents_from_ref = 1200 * np.log2(m.frequency / primary.ref_frequency)
            else:
                cents_from_ref = m.cents

            # Check if within spread tolerance
            if abs(cents_from_ref) > self.reed_spread_cents:
                continue

            reeds.append(ReedInfo(
                frequency=m.frequency,
                cents=cents_from_ref,
                magnitude=m.magnitude,
            ))

        # Sort by frequency
        reeds.sort(key=lambda r: r.frequency)

        return reeds

    def _compute_spectrum(self, samples: np.ndarray):
        """Compute FFT spectrum for display with zero-padding for finer resolution."""
        # Use the detector's accumulated buffer for full frequency resolution
        # The detector maintains a 16384-sample buffer that accumulates across calls
        fft_size = self._detector.fft_size
        buffer = self._detector._buffer

        # Apply window
        window = np.hamming(fft_size)
        windowed = buffer * window

        # Zero-pad to 16x for finer frequency interpolation in display
        # This gives ~0.17 Hz per bin instead of ~2.7 Hz per bin
        display_fft_size = fft_size * 16

        # FFT with zero-padding
        spectrum = np.fft.rfft(windowed, n=display_fft_size)
        magnitudes = np.abs(spectrum)

        # Frequency bins for the zero-padded FFT
        freqs = np.fft.rfftfreq(display_fft_size, 1.0 / self.sample_rate)

        # Limit to musical range (20 Hz to 2000 Hz for accordion)
        mask = (freqs >= 20) & (freqs <= 2000)
        self._fft_freqs = freqs[mask]
        self._fft_mags = magnitudes[mask]

        # Normalize magnitudes
        max_mag = np.max(self._fft_mags)
        if max_mag > 0:
            self._fft_mags = self._fft_mags / max_mag

    def _get_spectrum_tuple(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get spectrum data as tuple if both arrays are available."""
        if self._fft_freqs is not None and self._fft_mags is not None:
            return (self._fft_freqs, self._fft_mags)
        return None

    def set_reference(self, freq: float):
        """Set reference frequency for A4."""
        self.reference = freq
        self._detector.set_reference(freq)

    def set_max_reeds(self, count: int):
        """Set maximum number of reeds to detect."""
        self.max_reeds = min(max(1, count), 4)

    def set_reed_spread(self, cents: float):
        """Set maximum cents spread to consider as same note."""
        self.reed_spread_cents = max(10.0, min(100.0, cents))

    def set_temperament(self, temperament: Temperament):
        """Set musical temperament for reference frequency calculation."""
        self._detector.set_temperament(temperament)

    def set_key(self, key: int):
        """Set key for temperament (0=C, 1=C#, ..., 11=B)."""
        self._detector.set_key(key)

    def set_octave_filter(self, enabled: bool):
        """Enable/disable octave filter.

        When enabled: limits search to one octave above fundamental.
        When disabled (default for accordion): allows detecting closely-spaced
        frequencies for multiple reeds playing the same note.
        """
        self._detector.set_octave_filter(enabled)

    def set_sensitivity(self, threshold: float):
        """Set detection sensitivity (min_magnitude threshold).

        Lower values increase sensitivity but may detect more noise.
        Range: 0.05 - 0.5, default 0.1
        """
        self._detector.set_min_magnitude(max(0.05, min(0.5, threshold)))

    def set_fundamental_filter(self, enabled: bool):
        """Enable/disable fundamental filter (only detect harmonics)."""
        self._detector.set_fundamental_filter(enabled)

    def set_downsample(self, enabled: bool):
        """Enable/disable downsampling for better low frequency detection."""
        self._detector.set_downsample(enabled)

    def reset(self):
        """Reset internal state."""
        self._detector.reset()
        self._fft_freqs = None
        self._fft_mags = None
