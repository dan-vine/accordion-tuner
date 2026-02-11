"""
Temporal averaging/smoothing for pitch measurements.

This module provides smoothing of frequency measurements over time to produce
more stable readings, especially useful for accordion reed tuning where
instantaneous measurements can jitter.
"""

from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class ReedMeasurement(NamedTuple):
    """Single measurement for a reed."""
    frequency: float
    cents: float
    magnitude: float


@dataclass
class SmoothedReed:
    """Smoothed measurement result for a single reed."""
    frequency: float
    cents: float
    magnitude: float
    sample_count: int  # Number of samples in the average
    stability: float   # 0.0 to 1.0, higher = more stable


class ReedSmoother:
    """
    Smoother for a single reed's measurements.

    Uses weighted median filtering where measurements are weighted by their
    magnitude (signal strength). This provides:
    - Rejection of outliers (median)
    - Preference for high-confidence measurements (magnitude weighting)
    """

    def __init__(self, max_samples: int = 20):
        """
        Initialize reed smoother.

        Args:
            max_samples: Maximum number of measurements to keep in history
        """
        self.max_samples = max_samples
        self._history: deque[ReedMeasurement] = deque(maxlen=max_samples)

    def add(self, frequency: float, cents: float, magnitude: float):
        """Add a new measurement to the history."""
        self._history.append(ReedMeasurement(frequency, cents, magnitude))

    def get_smoothed(self) -> SmoothedReed | None:
        """
        Get the smoothed measurement.

        Returns:
            SmoothedReed with averaged values, or None if no measurements
        """
        if not self._history:
            return None

        # Extract arrays
        freqs = np.array([m.frequency for m in self._history])
        cents_arr = np.array([m.cents for m in self._history])
        mags = np.array([m.magnitude for m in self._history])

        # Use magnitude as weight (normalized)
        weights = mags / (np.sum(mags) + 1e-10)

        # Weighted median for frequency and cents
        smoothed_freq = self._weighted_median(freqs, weights)
        smoothed_cents = self._weighted_median(cents_arr, weights)

        # Average magnitude
        avg_mag = np.mean(mags)

        # Calculate stability as inverse of coefficient of variation
        # Lower variance relative to mean = higher stability
        if len(freqs) >= 3:
            freq_std = np.std(freqs)
            freq_mean = np.mean(freqs)
            # Convert to stability score (0-1)
            # A spread of 0.1 Hz is considered very stable, 2 Hz is unstable
            cv = freq_std / (freq_mean + 1e-10) * freq_mean  # Absolute std in Hz
            stability = max(0.0, min(1.0, 1.0 - cv / 2.0))
        else:
            stability = 0.0  # Not enough samples to judge

        return SmoothedReed(
            frequency=smoothed_freq,
            cents=smoothed_cents,
            magnitude=avg_mag,
            sample_count=len(self._history),
            stability=stability,
        )

    def _weighted_median(self, values: np.ndarray, weights: np.ndarray) -> float:
        """Compute weighted median."""
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cumsum = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumsum, 0.5)
        return float(sorted_values[min(median_idx, len(sorted_values) - 1)])

    def clear(self):
        """Clear all history."""
        self._history.clear()

    @property
    def sample_count(self) -> int:
        """Number of samples currently in history."""
        return len(self._history)


class MeasurementSmoother:
    """
    Multi-reed measurement smoother.

    Manages smoothing for multiple reeds simultaneously, handling:
    - Note changes (resets smoothers when note changes)
    - Reed count changes
    - Association of measurements to reed positions
    """

    def __init__(
        self,
        max_samples: int = 20,
        max_reeds: int = 4,
        note_change_threshold_cents: float = 100.0,
    ):
        """
        Initialize measurement smoother.

        Args:
            max_samples: Maximum samples per reed (at ~10 Hz = ~2 seconds)
            max_reeds: Maximum number of reeds to track
            note_change_threshold_cents: Cents deviation to consider a note change
        """
        self.max_samples = max_samples
        self.max_reeds = max_reeds
        self.note_change_threshold = note_change_threshold_cents

        # One smoother per reed position
        self._reed_smoothers: list[ReedSmoother] = [
            ReedSmoother(max_samples) for _ in range(max_reeds)
        ]

        # Track current note for change detection
        self._current_note: str | None = None
        self._current_octave: int | None = None

        # Track if we're in a valid measurement period
        self._is_active = False

    def update(
        self,
        note_name: str,
        octave: int,
        reed_measurements: list[tuple[float, float, float]],  # (freq, cents, mag)
    ) -> list[SmoothedReed | None]:
        """
        Update with new measurements and return smoothed results.

        Args:
            note_name: Detected note name (e.g., "C", "F#")
            octave: Detected octave
            reed_measurements: List of (frequency, cents, magnitude) per reed

        Returns:
            List of SmoothedReed for each reed position, None if no data
        """
        # Check for note change
        if self._current_note != note_name or self._current_octave != octave:
            # Note changed - reset all smoothers
            self.reset()
            self._current_note = note_name
            self._current_octave = octave

        self._is_active = True

        # Update each reed's smoother
        for i, smoother in enumerate(self._reed_smoothers):
            if i < len(reed_measurements):
                freq, cents, mag = reed_measurements[i]
                smoother.add(freq, cents, mag)

        # Return smoothed results
        return [s.get_smoothed() for s in self._reed_smoothers]

    def get_smoothed(self) -> list[SmoothedReed | None]:
        """Get current smoothed values without adding new measurements."""
        return [s.get_smoothed() for s in self._reed_smoothers]

    def reset(self):
        """Reset all smoothers (e.g., on note change or silence)."""
        for smoother in self._reed_smoothers:
            smoother.clear()
        self._is_active = False
        self._current_note = None
        self._current_octave = None

    def set_inactive(self):
        """Mark as inactive (no valid signal)."""
        # Don't clear immediately - keep last values for hold mode
        self._is_active = False

    def set_max_samples(self, max_samples: int):
        """Set maximum samples for averaging window."""
        self.max_samples = max_samples
        # Recreate smoothers with new size
        self._reed_smoothers = [
            ReedSmoother(max_samples) for _ in range(self.max_reeds)
        ]

    @property
    def is_active(self) -> bool:
        """Whether smoother is currently tracking a note."""
        return self._is_active

    @property
    def current_note(self) -> str | None:
        """Currently tracked note name."""
        return self._current_note

    @property
    def current_octave(self) -> int | None:
        """Currently tracked octave."""
        return self._current_octave
