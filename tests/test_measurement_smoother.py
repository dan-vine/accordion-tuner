"""Tests for the measurement smoother module."""

import pytest
import numpy as np

from accordion_tuner.measurement_smoother import (
    MeasurementSmoother,
    ReedSmoother,
    SmoothedReed,
)


class TestReedSmoother:
    """Tests for single reed smoother."""

    def test_empty_returns_none(self):
        """Empty smoother returns None."""
        smoother = ReedSmoother(max_samples=10)
        assert smoother.get_smoothed() is None

    def test_single_measurement(self):
        """Single measurement returns that measurement."""
        smoother = ReedSmoother(max_samples=10)
        smoother.add(440.0, 0.0, 1.0)

        result = smoother.get_smoothed()
        assert result is not None
        assert result.frequency == pytest.approx(440.0)
        assert result.cents == pytest.approx(0.0)
        assert result.sample_count == 1

    def test_averaging_multiple_measurements(self):
        """Multiple measurements are averaged."""
        smoother = ReedSmoother(max_samples=10)

        # Add 5 measurements around 440 Hz
        smoother.add(439.5, -2.0, 1.0)
        smoother.add(440.0, 0.0, 1.0)
        smoother.add(440.5, 2.0, 1.0)
        smoother.add(440.0, 0.0, 1.0)
        smoother.add(440.0, 0.0, 1.0)

        result = smoother.get_smoothed()
        assert result is not None
        # Median of [439.5, 440.0, 440.0, 440.0, 440.5] should be 440.0
        assert result.frequency == pytest.approx(440.0, abs=0.1)
        assert result.sample_count == 5

    def test_outlier_rejection(self):
        """Outliers are rejected by median filtering."""
        smoother = ReedSmoother(max_samples=10)

        # Add mostly consistent measurements with one outlier
        smoother.add(440.0, 0.0, 1.0)
        smoother.add(440.0, 0.0, 1.0)
        smoother.add(500.0, 100.0, 0.1)  # Outlier with low magnitude
        smoother.add(440.0, 0.0, 1.0)
        smoother.add(440.0, 0.0, 1.0)

        result = smoother.get_smoothed()
        assert result is not None
        # Outlier should be rejected
        assert result.frequency == pytest.approx(440.0, abs=1.0)

    def test_max_samples_limit(self):
        """Smoother respects max_samples limit."""
        smoother = ReedSmoother(max_samples=3)

        smoother.add(435.0, -20.0, 1.0)  # Will be pushed out
        smoother.add(440.0, 0.0, 1.0)
        smoother.add(440.0, 0.0, 1.0)
        smoother.add(440.0, 0.0, 1.0)

        result = smoother.get_smoothed()
        assert result is not None
        assert result.sample_count == 3
        # Old measurement should be gone
        assert result.frequency == pytest.approx(440.0, abs=0.1)

    def test_clear(self):
        """Clear removes all history."""
        smoother = ReedSmoother(max_samples=10)
        smoother.add(440.0, 0.0, 1.0)
        smoother.add(440.0, 0.0, 1.0)

        smoother.clear()
        assert smoother.get_smoothed() is None
        assert smoother.sample_count == 0

    def test_stability_calculation(self):
        """Stability is calculated based on variance."""
        smoother = ReedSmoother(max_samples=10)

        # Add very stable measurements
        for _ in range(10):
            smoother.add(440.0, 0.0, 1.0)

        result = smoother.get_smoothed()
        assert result is not None
        assert result.stability > 0.9  # Should be very stable

        # Now test with unstable measurements
        smoother2 = ReedSmoother(max_samples=10)
        for i in range(10):
            smoother2.add(435.0 + i, float(i * 5), 1.0)

        result2 = smoother2.get_smoothed()
        assert result2 is not None
        assert result2.stability < result.stability  # Should be less stable


class TestMeasurementSmoother:
    """Tests for multi-reed measurement smoother."""

    def test_empty_returns_list_of_none(self):
        """Empty smoother returns list of None."""
        smoother = MeasurementSmoother(max_samples=10, max_reeds=4)
        results = smoother.get_smoothed()
        assert len(results) == 4
        assert all(r is None for r in results)

    def test_update_single_reed(self):
        """Update with single reed measurement."""
        smoother = MeasurementSmoother(max_samples=10, max_reeds=4)

        measurements = [(440.0, 0.0, 1.0)]
        results = smoother.update("A", 4, measurements)

        assert results[0] is not None
        assert results[0].frequency == pytest.approx(440.0)
        assert results[1] is None  # Other reeds should be None

    def test_update_multiple_reeds(self):
        """Update with multiple reed measurements."""
        smoother = MeasurementSmoother(max_samples=10, max_reeds=4)

        measurements = [
            (439.0, -4.0, 1.0),  # Reed 1 (dry)
            (441.0, 4.0, 1.0),   # Reed 2 (wet)
        ]
        results = smoother.update("A", 4, measurements)

        assert results[0] is not None
        assert results[1] is not None
        assert results[0].frequency == pytest.approx(439.0)
        assert results[1].frequency == pytest.approx(441.0)

    def test_note_change_resets_smoother(self):
        """Changing notes resets all smoothers."""
        smoother = MeasurementSmoother(max_samples=10, max_reeds=4)

        # First note
        smoother.update("A", 4, [(440.0, 0.0, 1.0)])
        smoother.update("A", 4, [(440.0, 0.0, 1.0)])

        # Verify accumulated
        results = smoother.get_smoothed()
        assert results[0].sample_count == 2

        # Change note
        smoother.update("C", 4, [(261.63, 0.0, 1.0)])

        # Should be reset
        results = smoother.get_smoothed()
        assert results[0].sample_count == 1  # Only the new note

    def test_octave_change_resets_smoother(self):
        """Changing octaves resets all smoothers."""
        smoother = MeasurementSmoother(max_samples=10, max_reeds=4)

        smoother.update("A", 4, [(440.0, 0.0, 1.0)])
        smoother.update("A", 4, [(440.0, 0.0, 1.0)])

        # Change octave
        smoother.update("A", 5, [(880.0, 0.0, 1.0)])

        results = smoother.get_smoothed()
        assert results[0].sample_count == 1

    def test_set_inactive(self):
        """Set inactive marks smoother as inactive."""
        smoother = MeasurementSmoother(max_samples=10, max_reeds=4)

        smoother.update("A", 4, [(440.0, 0.0, 1.0)])
        assert smoother.is_active

        smoother.set_inactive()
        assert not smoother.is_active

    def test_reset(self):
        """Reset clears all state."""
        smoother = MeasurementSmoother(max_samples=10, max_reeds=4)

        smoother.update("A", 4, [(440.0, 0.0, 1.0)])
        smoother.reset()

        assert not smoother.is_active
        assert smoother.current_note is None
        results = smoother.get_smoothed()
        assert all(r is None for r in results)

    def test_set_max_samples(self):
        """set_max_samples recreates smoothers."""
        smoother = MeasurementSmoother(max_samples=20, max_reeds=4)

        # Add some data
        smoother.update("A", 4, [(440.0, 0.0, 1.0)])
        smoother.update("A", 4, [(440.0, 0.0, 1.0)])

        # Change max samples
        smoother.set_max_samples(5)

        # Data should be cleared (new smoothers created)
        results = smoother.get_smoothed()
        assert all(r is None for r in results)


class TestSmootherIntegration:
    """Integration tests with simulated audio scenarios."""

    def test_tremolo_reed_pair_stabilizes(self):
        """Two close frequencies stabilize over time."""
        smoother = MeasurementSmoother(max_samples=20, max_reeds=4)

        # Simulate tremolo pair: 440 Hz and 442 Hz with some jitter
        np.random.seed(42)
        for i in range(20):
            f1 = 440.0 + np.random.normal(0, 0.1)
            f2 = 442.0 + np.random.normal(0, 0.1)
            c1 = (f1 - 440.0) / 440.0 * 1200  # Approximate cents
            c2 = (f2 - 440.0) / 440.0 * 1200

            measurements = [(f1, c1, 1.0), (f2, c2, 1.0)]
            results = smoother.update("A", 4, measurements)

        # Final results should be stable
        assert results[0] is not None
        assert results[1] is not None
        assert results[0].frequency == pytest.approx(440.0, abs=0.2)
        assert results[1].frequency == pytest.approx(442.0, abs=0.2)
        assert results[0].stability > 0.5
        assert results[1].stability > 0.5

    def test_stability_increases_with_samples(self):
        """Stability should increase as more samples accumulate."""
        smoother = MeasurementSmoother(max_samples=20, max_reeds=4)

        stabilities = []
        for i in range(15):
            measurements = [(440.0, 0.0, 1.0)]
            results = smoother.update("A", 4, measurements)
            if results[0]:
                stabilities.append(results[0].stability)

        # Stability should generally increase
        assert len(stabilities) > 5
        # Later samples should have higher stability than earlier ones
        assert stabilities[-1] >= stabilities[2]
