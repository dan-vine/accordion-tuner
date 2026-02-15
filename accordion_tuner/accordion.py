"""
Accordion reed tuning detection.

This module provides detection of multiple detuned reeds playing the same note,
suitable for accordion tuning where 2-4 reeds may play simultaneously with
intentional detuning (tremolo/musette effects).
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from .constants import (
    A4_REFERENCE,
    SAMPLE_RATE,
)
from .esprit_detector import EspritPitchDetector
from .measurement_smoother import MeasurementSmoother
from .multi_pitch_detector import Maximum, MultiPitchDetector
from .simple_fft_detector import SimpleFftPeakDetector
from .temperaments import Temperament

if TYPE_CHECKING:
    from .tremolo_profile import TremoloProfile


class DetectorType(Enum):
    """Type of pitch detection algorithm."""

    FFT = "fft"  # FFT + Phase Vocoder (default)
    ESPRIT = "esprit"  # FFT-ESPRIT (best for close frequencies)
    SIMPLE_FFT = "simple_fft"  # Simple FFT with scipy find_peaks


@dataclass
class ReedInfo:
    """Information about a single detected reed."""

    frequency: float = 0.0  # Detected frequency in Hz
    cents: float = 0.0  # Deviation from reference in cents
    magnitude: float = 0.0  # Signal strength (for confidence)
    target_cents: float | None = None  # Deviation from target (when profile active)
    stability: float = 0.0  # Measurement stability (0.0-1.0, higher = more stable)
    sample_count: int = 0  # Number of samples in the smoothed average
    # Precision mode fields (from long-window FFT)
    precision_frequency: float | None = (
        None  # High-resolution frequency (when precision mode enabled)
    )
    precision_cents: float | None = None  # High-resolution cents deviation


@dataclass
class PrecisionInfo:
    """Information about precision detection state."""

    enabled: bool = False
    fill_level: float = 0.0  # Buffer fill level (0.0 to 1.0)
    resolution: float = 0.0  # Current frequency resolution in Hz
    duration: float = 0.0  # Duration of accumulated audio in seconds
    is_stable: bool = False  # True if buffer is full enough for reliable measurement


@dataclass
class AccordionResult:
    """Result of accordion reed detection."""

    valid: bool = False
    note_name: str = ""  # e.g., "C"
    octave: int = 0  # e.g., 4
    ref_frequency: float = 0.0  # Reference frequency for this note
    reeds: list[ReedInfo] = field(default_factory=list)
    beat_frequencies: list[float] = field(default_factory=list)  # |f1-f2|, |f2-f3|, etc.
    spectrum_data: tuple[np.ndarray, np.ndarray] | None = None  # (frequencies, magnitudes)
    precision_info: PrecisionInfo | None = None  # Precision mode state

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
        detector_type: DetectorType = DetectorType.FFT,
    ):
        """
        Initialize accordion detector.

        Args:
            sample_rate: Audio sample rate in Hz
            reference: Reference frequency for A4 in Hz
            max_reeds: Maximum number of reeds to detect (2-4)
            reed_spread_cents: Maximum cents spread to consider as same note
            detector_type: Type of pitch detection algorithm to use
        """
        self.sample_rate = sample_rate
        self.reference = reference
        self.max_reeds = min(max(1, max_reeds), 4)
        self.reed_spread_cents = reed_spread_cents
        self._detector_type = detector_type

        # Store current temperament and key for detector configuration/switching
        # Must be initialized before _create_detector/_configure_detector
        self._temperament = Temperament.EQUAL
        self._key = 0

        # Internal multi-pitch detector with accordion-specific settings
        self._detector = self._create_detector(detector_type)
        self._configure_detector()

        # FFT state for spectrum display
        self._fft_freqs: np.ndarray | None = None
        self._fft_mags: np.ndarray | None = None

        # Tremolo profile for target deviation calculation
        self._tremolo_profile: TremoloProfile | None = None

        # Temporal smoothing for stable measurements
        self._smoother = MeasurementSmoother(
            max_samples=20,  # ~2 seconds at 10 Hz update rate
            max_reeds=self.max_reeds,
        )
        self._smoothing_enabled = True

        # Precision mode: accumulate audio for high-resolution FFT
        self._precision_enabled = False
        self._precision_window = 3.0  # seconds
        self._precision_buffer_size = int(self.sample_rate * self._precision_window)
        self._precision_buffer: deque[float] = deque(maxlen=self._precision_buffer_size)
        self._precision_stable_threshold = 0.75  # 75% fill for stable measurement
        self._precision_current_note: str | None = None
        self._precision_current_octave: int | None = None

    def _create_detector(
        self, detector_type: DetectorType
    ) -> MultiPitchDetector | EspritPitchDetector | SimpleFftPeakDetector:
        """Create a pitch detector of the specified type."""
        if detector_type == DetectorType.ESPRIT:
            return EspritPitchDetector(
                sample_rate=self.sample_rate,
                reference=self.reference,
                num_sources=self.max_reeds,
            )
        elif detector_type == DetectorType.SIMPLE_FFT:
            return SimpleFftPeakDetector(
                sample_rate=self.sample_rate,
                reference=self.reference,
                num_sources=self.max_reeds,
            )
        else:
            return MultiPitchDetector(
                sample_rate=self.sample_rate,
                reference=self.reference,
            )

    def _configure_detector(self):
        """Configure the current detector with accordion-specific settings."""
        if self._detector_type == DetectorType.SIMPLE_FFT:
            # Enable filters for SimpleFFT to focus on fundamentals
            self._detector.set_octave_filter(True)
            self._detector.set_fundamental_filter(True)
        else:
            # Disable octave filter to detect closely-spaced frequencies
            self._detector.set_octave_filter(False)
        # Lower threshold for typical microphone input levels
        self._detector.set_min_magnitude(0.03)

        # Apply stored temperament and key
        self._detector.set_temperament(self._temperament)
        self._detector.set_key(self._key)

    def set_detector_type(self, detector_type: DetectorType):
        """
        Switch the pitch detection algorithm.

        Args:
            detector_type: Type of detector to use (FFT or ESPRIT)
        """
        if detector_type == self._detector_type:
            return

        self._detector_type = detector_type
        self._detector = self._create_detector(detector_type)
        self._configure_detector()

    def get_detector_type(self) -> DetectorType:
        """Get the current detector type."""
        return self._detector_type

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
        self._compute_spectrum()

        if not multi_result.valid or not multi_result.maxima:
            self._smoother.set_inactive()
            precision_info = self._get_precision_info() if self._precision_enabled else None
            return AccordionResult(
                spectrum_data=self._get_spectrum_tuple(),
                precision_info=precision_info,
            )

        # Primary note from the first (strongest) detection
        primary = multi_result.maxima[0]

        # Group peaks that correspond to the same note (within reed_spread_cents)
        reeds = self._group_reeds(
            multi_result.maxima,
            primary.note_name,
            primary.octave,
            primary.ref_frequency,
        )

        if not reeds:
            self._smoother.set_inactive()
            precision_info = self._get_precision_info() if self._precision_enabled else None
            return AccordionResult(
                spectrum_data=self._get_spectrum_tuple(),
                precision_info=precision_info,
            )

        # Apply temporal smoothing if enabled
        if self._smoothing_enabled:
            reeds = self._apply_smoothing(
                primary.note_name,
                primary.octave,
                primary.ref_frequency,
                reeds,
            )

        # Apply precision mode if enabled
        precision_info = None
        if self._precision_enabled:
            reeds, precision_info = self._apply_precision(
                samples,
                primary.note_name,
                primary.octave,
                primary.ref_frequency,
                reeds,
            )

        # Calculate beat frequencies between adjacent reeds
        # Use precision frequencies if available, otherwise regular frequencies
        beat_freqs = []
        for i in range(len(reeds) - 1):
            f1 = reeds[i].precision_frequency or reeds[i].frequency
            f2 = reeds[i + 1].precision_frequency or reeds[i + 1].frequency
            beat = abs(f1 - f2)
            beat_freqs.append(beat)

        return AccordionResult(
            valid=True,
            note_name=primary.note_name,
            octave=primary.octave,
            ref_frequency=primary.ref_frequency,
            reeds=reeds,
            beat_frequencies=beat_freqs,
            spectrum_data=self._get_spectrum_tuple(),
            precision_info=precision_info,
        )

    def _apply_smoothing(
        self,
        note_name: str,
        octave: int,
        ref_frequency: float,
        reeds: list[ReedInfo],
    ) -> list[ReedInfo]:
        """
        Apply temporal smoothing to reed measurements.

        Args:
            note_name: Detected note name
            octave: Detected octave
            ref_frequency: Reference frequency for the note
            reeds: Raw reed measurements

        Returns:
            Smoothed reed measurements
        """
        # Prepare measurements for smoother
        measurements = [(r.frequency, r.cents, r.magnitude) for r in reeds]

        # Update smoother and get smoothed results
        smoothed_results = self._smoother.update(note_name, octave, measurements)

        # Build smoothed ReedInfo list
        smoothed_reeds = []
        for i, reed in enumerate(reeds):
            smoothed = smoothed_results[i] if i < len(smoothed_results) else None

            if smoothed is not None:
                # Use smoothed values
                smoothed_reed = ReedInfo(
                    frequency=smoothed.frequency,
                    cents=smoothed.cents,
                    magnitude=smoothed.magnitude,
                    target_cents=reed.target_cents,  # Keep original target_cents for now
                    stability=smoothed.stability,
                    sample_count=smoothed.sample_count,
                )
                smoothed_reeds.append(smoothed_reed)
            else:
                # No smoothed data yet, use raw
                smoothed_reeds.append(reed)

        # Recompute target_cents for smoothed frequencies if profile active
        if self._tremolo_profile is not None and smoothed_reeds:
            beat_freq = self._tremolo_profile.get_beat_frequency(note_name, octave)
            if beat_freq is not None:
                self._compute_target_cents(smoothed_reeds, ref_frequency, beat_freq)

        return smoothed_reeds

    def _group_reeds(
        self,
        maxima: list[Maximum],
        note_name: str,
        octave: int,
        ref_frequency: float,
    ) -> list[ReedInfo]:
        """
        Group detected peaks into reeds for the same note.

        Args:
            maxima: List of detected frequency maxima
            note_name: Note name for tremolo profile lookup
            octave: Octave number for tremolo profile lookup
            ref_frequency: Reference frequency for the note

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

            reeds.append(
                ReedInfo(
                    frequency=m.frequency,
                    cents=cents_from_ref,
                    magnitude=m.magnitude,
                    target_cents=None,  # Will be computed after sorting
                )
            )

        # Sort by frequency
        reeds.sort(key=lambda r: r.frequency)

        # Compute target_cents if tremolo profile is active
        if self._tremolo_profile is not None and reeds:
            beat_freq = self._tremolo_profile.get_beat_frequency(note_name, octave)
            if beat_freq is not None:
                self._compute_target_cents(reeds, ref_frequency, beat_freq)

        return reeds

    def _compute_target_cents(
        self,
        reeds: list[ReedInfo],
        ref_frequency: float,
        beat_frequency: float,
    ) -> None:
        """
        Compute target_cents for each reed based on tremolo profile.

        Display logic by reed mode:
        - 1 reed: deviation from target (ref ± beat)
        - 2 reed: Reed 1 = reference (0¢), Reed 2 = deviation from (ref + beat)
        - 3 reed: Reed 1 = deviation from (ref - beat), Reed 2 = reference (0¢),
                  Reed 3 = deviation from (ref + beat)
        - 4 reed: Reed 1 = deviation from (ref - beat), Reed 2 = reference (0¢),
                  Reed 3 = reference (0¢), Reed 4 = deviation from (ref + beat)

        Args:
            reeds: List of ReedInfo (already sorted by frequency)
            ref_frequency: Reference frequency for the note
            beat_frequency: Target beat frequency from profile
        """
        num_reeds = len(reeds)

        if num_reeds == 1:
            # Single reed - show deviation from nearest target (ref + beat or ref - beat)
            # Choose whichever is closer
            target_high = ref_frequency + beat_frequency
            target_low = ref_frequency - beat_frequency
            freq = reeds[0].frequency
            if abs(freq - target_high) < abs(freq - target_low):
                target = target_high
            else:
                target = target_low
            reeds[0].target_cents = 1200 * np.log2(freq / target)

        elif num_reeds == 2:
            # 2 reeds: Reed 1 = reference, Reed 2 = ref + beat
            reeds[0].target_cents = reeds[0].cents  # deviation from reference
            target = ref_frequency + beat_frequency
            reeds[1].target_cents = 1200 * np.log2(reeds[1].frequency / target)

        elif num_reeds == 3:
            # 3 reeds: Reed 1 = ref - beat, Reed 2 = reference, Reed 3 = ref + beat
            target_low = ref_frequency - beat_frequency
            reeds[0].target_cents = 1200 * np.log2(reeds[0].frequency / target_low)
            reeds[1].target_cents = reeds[1].cents  # deviation from reference
            target_high = ref_frequency + beat_frequency
            reeds[2].target_cents = 1200 * np.log2(reeds[2].frequency / target_high)

        elif num_reeds >= 4:
            # 4 reeds: Reed 1 = ref - beat, Reed 2 = reference, Reed 3 = reference,
            #          Reed 4 = ref + beat
            target_low = ref_frequency - beat_frequency
            reeds[0].target_cents = 1200 * np.log2(reeds[0].frequency / target_low)
            reeds[1].target_cents = reeds[1].cents  # deviation from reference
            reeds[2].target_cents = reeds[2].cents  # deviation from reference
            target_high = ref_frequency + beat_frequency
            reeds[3].target_cents = 1200 * np.log2(reeds[3].frequency / target_high)

    def _apply_precision(
        self,
        samples: np.ndarray,
        note_name: str,
        octave: int,
        ref_frequency: float,
        reeds: list[ReedInfo],
    ) -> tuple[list[ReedInfo], PrecisionInfo]:
        """
        Apply precision mode analysis using accumulated audio.

        Args:
            samples: Current audio samples
            note_name: Detected note name
            octave: Detected octave
            ref_frequency: Reference frequency for the note
            reeds: Current reed measurements

        Returns:
            Tuple of (updated reeds with precision data, precision info)
        """
        # Check for note change - reset buffer if note changed
        if self._precision_current_note != note_name or self._precision_current_octave != octave:
            self._precision_buffer.clear()
            self._precision_current_note = note_name
            self._precision_current_octave = octave

        # Add new samples to precision buffer
        for sample in samples:
            self._precision_buffer.append(sample)

        # Get precision info
        precision_info = self._get_precision_info()

        # Only run precision FFT if we have enough samples
        if len(self._precision_buffer) < self.sample_rate // 2:  # At least 0.5 seconds
            return reeds, precision_info

        # Run precision FFT analysis
        precision_freqs = self._precision_fft_analysis(reeds)

        # Update reeds with precision frequencies
        for i, reed in enumerate(reeds):
            if i < len(precision_freqs) and precision_freqs[i] is not None:
                reed.precision_frequency = precision_freqs[i]
                # Calculate precision cents
                if ref_frequency > 0:
                    reed.precision_cents = 1200 * np.log2(precision_freqs[i] / ref_frequency)

        return reeds, precision_info

    def _precision_fft_analysis(self, reeds: list[ReedInfo]) -> list[float | None]:
        """
        Run precision FFT on accumulated buffer to refine frequency measurements.

        Args:
            reeds: Current reed measurements (used to know where to look)

        Returns:
            List of precision frequencies for each reed, None if not found
        """
        buffer_len = len(self._precision_buffer)
        if buffer_len < 100:
            return [None] * len(reeds)

        # Convert buffer to numpy array
        signal = np.array(self._precision_buffer, dtype=np.float64)

        # Normalize
        signal_max = np.max(np.abs(signal))
        if signal_max < 1e-6:
            return [None] * len(reeds)
        signal = signal / signal_max

        # Apply window
        window = np.hamming(buffer_len)
        windowed = signal * window

        # Zero-pad to next power of 2 for efficient FFT
        n_fft = 1 << (buffer_len - 1).bit_length()
        n_fft = max(n_fft, buffer_len * 2)  # At least 2x for interpolation

        # FFT
        spectrum = np.fft.rfft(windowed, n=n_fft)
        magnitudes = np.abs(spectrum)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)

        # First, find ALL peaks in the frequency range of interest
        # (around all the reed frequencies)
        if not reeds:
            return []

        min_freq = min(r.frequency for r in reeds) - 10.0
        max_freq = max(r.frequency for r in reeds) + 10.0
        min_freq = max(20.0, min_freq)
        max_freq = min(2000.0, max_freq)

        low_idx = np.searchsorted(freqs, min_freq)
        high_idx = np.searchsorted(freqs, max_freq)

        if low_idx >= high_idx or high_idx >= len(magnitudes):
            return [None] * len(reeds)

        search_mags = magnitudes[low_idx:high_idx]
        search_freqs = freqs[low_idx:high_idx]

        if len(search_mags) < 3:
            return [None] * len(reeds)

        # Find all local maxima (peaks)
        max_mag = np.max(search_mags)
        threshold = max_mag * 0.1  # At least 10% of max

        all_peaks = []  # List of (frequency, magnitude)
        for i in range(1, len(search_mags) - 1):
            if search_mags[i] < threshold:
                continue
            if search_mags[i] > search_mags[i - 1] and search_mags[i] > search_mags[i + 1]:
                # Parabolic interpolation for sub-bin accuracy
                y1, y2, y3 = search_mags[i - 1], search_mags[i], search_mags[i + 1]
                denom = y1 - 2 * y2 + y3
                if abs(denom) > 1e-10:
                    delta = 0.5 * (y1 - y3) / denom
                    freq_step = search_freqs[1] - search_freqs[0]
                    peak_freq = search_freqs[i] + delta * freq_step
                    peak_mag = y2 - 0.25 * (y1 - y3) * delta
                else:
                    peak_freq = search_freqs[i]
                    peak_mag = y2
                all_peaks.append((peak_freq, peak_mag))

        if not all_peaks:
            return [None] * len(reeds)

        # Sort peaks by magnitude (strongest first)
        all_peaks.sort(key=lambda x: x[1], reverse=True)

        # Assign peaks to reeds - each peak can only be used once
        # Match each reed to the closest available peak
        precision_freqs: list[float | None] = [None] * len(reeds)
        used_peaks = set()

        # Sort reeds by their detected frequency
        reed_indices = sorted(range(len(reeds)), key=lambda i: reeds[i].frequency)

        for reed_idx in reed_indices:
            reed = reeds[reed_idx]
            target_freq = reed.frequency

            # Find closest unused peak within ±5 Hz
            best_peak = None
            best_dist = float("inf")

            for i, (peak_freq, _peak_mag) in enumerate(all_peaks):
                if i in used_peaks:
                    continue
                dist = abs(peak_freq - target_freq)
                if dist < 5.0 and dist < best_dist:
                    best_dist = dist
                    best_peak = i

            if best_peak is not None:
                precision_freqs[reed_idx] = all_peaks[best_peak][0]
                used_peaks.add(best_peak)

        return precision_freqs

    def _get_precision_info(self) -> PrecisionInfo:
        """Get current precision mode state."""
        buffer_len = len(self._precision_buffer)
        fill_level = buffer_len / self._precision_buffer_size
        duration = buffer_len / self.sample_rate
        resolution = self.sample_rate / buffer_len if buffer_len > 0 else float("inf")
        is_stable = fill_level >= self._precision_stable_threshold

        return PrecisionInfo(
            enabled=self._precision_enabled,
            fill_level=fill_level,
            resolution=resolution,
            duration=duration,
            is_stable=is_stable,
        )

    def _compute_spectrum(self):
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
        """Set maximum number of reeds to detect (minimum 2, maximum 4)."""
        self.max_reeds = min(max(2, count), 4)
        if isinstance(self._detector, EspritPitchDetector):
            self._detector.set_num_sources(self.max_reeds)
        elif isinstance(self._detector, SimpleFftPeakDetector):
            self._detector.set_num_sources(self.max_reeds)

    def set_reed_spread(self, cents: float):
        """Set maximum cents spread to consider as same note."""
        self.reed_spread_cents = max(10.0, min(100.0, cents))

    def set_temperament(self, temperament: Temperament):
        """Set musical temperament for reference frequency calculation."""
        self._temperament = temperament
        self._detector.set_temperament(temperament)

    def set_key(self, key: int):
        """Set key for temperament (0=C, 1=C#, ..., 11=B)."""
        self._key = key
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
        Range: 0.01 - 0.5, default 0.1
        """
        self._detector.set_min_magnitude(max(0.01, min(0.5, threshold)))

    def set_peak_threshold(self, threshold: float):
        """Set relative peak threshold for detection.

        Peaks must be at least this fraction of the maximum peak to be detected.
        Lower values detect weaker peaks (useful for quiet reeds) but may pick
        up noise or harmonics.

        Range: 0.05 - 0.50, default 0.25 (FFT) or 0.10 (ESPRIT)
        """
        self._detector.set_peak_threshold(max(0.05, min(0.50, threshold)))

    def set_fundamental_filter(self, enabled: bool):
        """Enable/disable fundamental filter (only detect harmonics)."""
        self._detector.set_fundamental_filter(enabled)

    def set_tremolo_profile(self, profile: "TremoloProfile | None") -> None:
        """
        Set the tremolo tuning profile.

        When a profile is active, target_cents will be computed for each reed
        showing deviation from the target frequency based on the profile's
        beat frequency for that note.

        Args:
            profile: TremoloProfile to use, or None to disable
        """
        self._tremolo_profile = profile

    # ESPRIT-specific settings (only effective when using ESPRIT detector)
    def set_esprit_width_threshold(self, threshold: float):
        """
        Set ESPRIT width threshold for detecting merged peaks.

        Only effective when using ESPRIT detector.
        A single Hamming-windowed sinusoid has ~0.08 ratio at ±2 bins.
        Lower thresholds = more sensitive (better close-freq detection but may hallucinate).

        Args:
            threshold: Ratio threshold (0.1 to 0.5, default 0.18)
        """
        if isinstance(self._detector, EspritPitchDetector):
            self._detector.set_width_threshold(threshold)

    def get_esprit_width_threshold(self) -> float:
        """Get ESPRIT width threshold (returns 0.25 if not using ESPRIT)."""
        if isinstance(self._detector, EspritPitchDetector):
            return self._detector.get_width_threshold()
        return 0.25

    def set_esprit_candidate_offsets(self, offsets: list[float]):
        """
        Set ESPRIT frequency offsets for candidate generation.

        Only effective when using ESPRIT detector.
        When a merged peak is detected, candidates are added at these offsets
        to help ESPRIT resolve the close frequencies.

        Args:
            offsets: List of Hz offsets (e.g., [-0.6, -0.3, 0.3, 0.6])
        """
        if isinstance(self._detector, EspritPitchDetector):
            self._detector.set_candidate_offsets(offsets)

    def get_esprit_candidate_offsets(self) -> list[float]:
        """Get ESPRIT candidate offsets (returns default if not using ESPRIT)."""
        if isinstance(self._detector, EspritPitchDetector):
            return self._detector.get_candidate_offsets()
        return [-0.6, -0.3, 0.3, 0.6]

    def set_esprit_min_separation(self, separation: float):
        """
        Set ESPRIT minimum separation between detected frequencies.

        Only effective when using ESPRIT detector.

        Args:
            separation: Minimum Hz between peaks (0.3 to 1.0, default 0.5)
        """
        if isinstance(self._detector, EspritPitchDetector):
            self._detector.set_min_separation(separation)

    def get_esprit_min_separation(self) -> float:
        """Get ESPRIT minimum separation (returns 0.5 if not using ESPRIT)."""
        if isinstance(self._detector, EspritPitchDetector):
            return self._detector.get_min_separation()
        return 0.5

    # SimpleFFT-specific settings (only effective when using SimpleFFT detector)
    def set_simple_fft_second_reed_search(self, hz: float):
        """
        Set SimpleFFT search range for second reed detection.

        Only effective when using SimpleFFT detector.
        After finding the first (fundamental) reed, this defines the maximum
        Hz range above the fundamental to search for a second reed.

        Args:
            hz: Search range in Hz (1.0 to 8.0, default 3.0)
        """
        if isinstance(self._detector, SimpleFftPeakDetector):
            self._detector.set_second_reed_search(hz)

    def get_simple_fft_second_reed_search(self) -> float:
        """Get SimpleFFT second reed search range (returns 3.0 if not using SimpleFFT)."""
        if isinstance(self._detector, SimpleFftPeakDetector):
            return self._detector.second_reed_search_hz
        return 3.0

    def set_simple_fft_second_reed_threshold(self, threshold: float):
        """
        Set SimpleFFT threshold for second reed detection.

        Only effective when using SimpleFFT detector.
        Lower threshold = more sensitive to second reed.
        Higher threshold = fewer false positives.

        Args:
            threshold: Magnitude threshold ratio (0.05 to 0.25, default 0.10)
        """
        if isinstance(self._detector, SimpleFftPeakDetector):
            search_hz = self._detector.second_reed_search_hz
            self._detector.set_second_reed_search(search_hz, threshold)

    def get_simple_fft_second_reed_threshold(self) -> float:
        """Get SimpleFFT second reed threshold (returns 0.10 if not using SimpleFFT)."""
        if isinstance(self._detector, SimpleFftPeakDetector):
            return self._detector.second_reed_threshold
        return 0.10

    def reset(self):
        """Reset internal state."""
        self._detector.reset()
        self._fft_freqs = None
        self._fft_mags = None
        self._smoother.reset()
        self._precision_buffer.clear()
        self._precision_current_note = None
        self._precision_current_octave = None

    # Smoothing settings
    def set_smoothing_enabled(self, enabled: bool):
        """
        Enable or disable temporal smoothing.

        When enabled, measurements are averaged over time for stability.
        When disabled, raw instantaneous measurements are returned.

        Args:
            enabled: True to enable smoothing
        """
        self._smoothing_enabled = enabled
        if not enabled:
            self._smoother.reset()

    def is_smoothing_enabled(self) -> bool:
        """Check if smoothing is enabled."""
        return self._smoothing_enabled

    def set_smoothing_window(self, samples: int):
        """
        Set the smoothing window size.

        Larger windows provide more stable readings but respond slower to changes.
        At typical 10 Hz update rate:
        - 5 samples = 0.5 seconds
        - 10 samples = 1 second
        - 20 samples = 2 seconds (default)
        - 30 samples = 3 seconds

        Args:
            samples: Number of measurements to average (5-50)
        """
        samples = max(5, min(50, samples))
        self._smoother.set_max_samples(samples)

    def get_smoothing_window(self) -> int:
        """Get current smoothing window size."""
        return self._smoother.max_samples

    def reset_smoothing(self):
        """Reset the smoother, clearing accumulated measurements."""
        self._smoother.reset()

    # Precision mode settings
    def set_precision_enabled(self, enabled: bool):
        """
        Enable or disable precision mode.

        When enabled, accumulates audio over a longer window (2-4 seconds)
        for high-resolution frequency detection. This provides ~0.5 Hz or
        better resolution, improving measurement stability over time.

        Args:
            enabled: True to enable precision mode
        """
        self._precision_enabled = enabled
        if not enabled:
            self._precision_buffer.clear()

    def is_precision_enabled(self) -> bool:
        """Check if precision mode is enabled."""
        return self._precision_enabled

    def set_precision_window(self, duration: float):
        """
        Set the precision mode accumulation window.

        Longer windows provide finer frequency resolution:
        - 2 seconds: ~0.5 Hz resolution
        - 3 seconds: ~0.37 Hz resolution
        - 4 seconds: ~0.28 Hz resolution

        Args:
            duration: Window duration in seconds (1.0 to 5.0)
        """
        duration = max(1.0, min(5.0, duration))
        self._precision_window = duration
        self._precision_buffer_size = int(self.sample_rate * duration)
        self._precision_buffer = deque(maxlen=self._precision_buffer_size)

    def get_precision_window(self) -> float:
        """Get current precision window duration in seconds."""
        return self._precision_window

    def get_precision_fill_level(self) -> float:
        """Get precision buffer fill level (0.0 to 1.0)."""
        return len(self._precision_buffer) / self._precision_buffer_size

    def get_precision_resolution(self) -> float:
        """Get current frequency resolution in Hz based on buffer fill."""
        buffer_len = len(self._precision_buffer)
        if buffer_len == 0:
            return float("inf")
        return self.sample_rate / buffer_len

    def reset_precision(self):
        """Reset precision buffer, clearing accumulated audio."""
        self._precision_buffer.clear()
        self._precision_current_note = None
        self._precision_current_octave = None
