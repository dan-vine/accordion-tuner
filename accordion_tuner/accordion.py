"""
Accordion reed tuning detection.

This module provides detection of multiple detuned reeds playing the same note,
suitable for accordion tuning where 2-4 reeds may play simultaneously with
intentional detuning (tremolo/musette effects).
"""

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
from .temperaments import Temperament

if TYPE_CHECKING:
    from .tremolo_profile import TremoloProfile


class DetectorType(Enum):
    """Type of pitch detection algorithm."""
    FFT = "fft"        # FFT + Phase Vocoder (default)
    ESPRIT = "esprit"  # FFT-ESPRIT (best for close frequencies)


@dataclass
class ReedInfo:
    """Information about a single detected reed."""
    frequency: float = 0.0      # Detected frequency in Hz
    cents: float = 0.0          # Deviation from reference in cents
    magnitude: float = 0.0      # Signal strength (for confidence)
    target_cents: float | None = None  # Deviation from target (when profile active)
    stability: float = 0.0      # Measurement stability (0.0-1.0, higher = more stable)
    sample_count: int = 0       # Number of samples in the smoothed average


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

    def _create_detector(
        self, detector_type: DetectorType
    ) -> MultiPitchDetector | EspritPitchDetector:
        """Create a pitch detector of the specified type."""
        if detector_type == DetectorType.ESPRIT:
            return EspritPitchDetector(
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
        # Disable octave filter to detect closely-spaced frequencies
        self._detector.set_octave_filter(False)
        # Lower threshold for typical microphone input levels
        self._detector.set_min_magnitude(0.1)

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
        self._compute_spectrum(samples)

        if not multi_result.valid or not multi_result.maxima:
            self._smoother.set_inactive()
            return AccordionResult(spectrum_data=self._get_spectrum_tuple())

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
            return AccordionResult(spectrum_data=self._get_spectrum_tuple())

        # Apply temporal smoothing if enabled
        if self._smoothing_enabled:
            reeds = self._apply_smoothing(
                primary.note_name,
                primary.octave,
                primary.ref_frequency,
                reeds,
            )

        # Calculate beat frequencies between adjacent reeds
        beat_freqs = []
        for i in range(len(reeds) - 1):
            beat = abs(reeds[i].frequency - reeds[i + 1].frequency)
            beat_freqs.append(beat)

        return AccordionResult(
            valid=True,
            note_name=primary.note_name,
            octave=primary.octave,
            ref_frequency=primary.ref_frequency,
            reeds=reeds,
            beat_frequencies=beat_freqs,
            spectrum_data=self._get_spectrum_tuple(),
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
        measurements = [
            (r.frequency, r.cents, r.magnitude) for r in reeds
        ]

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

            reeds.append(ReedInfo(
                frequency=m.frequency,
                cents=cents_from_ref,
                magnitude=m.magnitude,
                target_cents=None,  # Will be computed after sorting
            ))

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
        """Set maximum number of reeds to detect (minimum 2, maximum 4)."""
        self.max_reeds = min(max(2, count), 4)
        # Update ESPRIT detector's num_sources if applicable
        if isinstance(self._detector, EspritPitchDetector):
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
        Range: 0.05 - 0.5, default 0.1
        """
        self._detector.set_min_magnitude(max(0.05, min(0.5, threshold)))

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

    def set_downsample(self, enabled: bool):
        """Enable/disable downsampling for better low frequency detection."""
        self._detector.set_downsample(enabled)

    # ESPRIT-specific settings (only effective when using ESPRIT detector)
    def set_esprit_width_threshold(self, threshold: float):
        """
        Set ESPRIT width threshold for detecting merged peaks.

        Only effective when using ESPRIT detector.
        A single Hamming-windowed sinusoid has ~0.08 ratio at ±2 bins.
        Higher thresholds = less sensitive to merged peaks (fewer false positives).
        Lower thresholds = more sensitive (better close-freq detection but may hallucinate).

        Args:
            threshold: Ratio threshold (0.1 to 0.5, default 0.25)
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
            offsets: List of Hz offsets (e.g., [-0.8, -0.4, 0.4, 0.8])
        """
        if isinstance(self._detector, EspritPitchDetector):
            self._detector.set_candidate_offsets(offsets)

    def get_esprit_candidate_offsets(self) -> list[float]:
        """Get ESPRIT candidate offsets (returns default if not using ESPRIT)."""
        if isinstance(self._detector, EspritPitchDetector):
            return self._detector.get_candidate_offsets()
        return [-0.8, -0.4, 0.4, 0.8]

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

    def reset(self):
        """Reset internal state."""
        self._detector.reset()
        self._fft_freqs = None
        self._fft_mags = None
        self._smoother.reset()

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
