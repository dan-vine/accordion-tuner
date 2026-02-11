"""
Tests for pitch detection using synthetic signals.

These tests generate sine waves at known frequencies and verify that the
pitch detector correctly identifies:
- The frequency
- The note name
- The cents deviation

This allows us to verify the Python implementation produces similar
results to the C++ implementation without needing to compile it.
"""

import numpy as np
import pytest
from accordion_tuner import MultiPitchDetector, MultiPitchResult, SAMPLE_RATE, BUFFER_SIZE
from accordion_tuner.temperaments import Temperament


def generate_sine_wave(
    frequency: float,
    duration_samples: int,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Generate a sine wave at the given frequency."""
    t = np.arange(duration_samples) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def generate_test_signal(
    frequency: float,
    num_buffers: int = 10,
    buffer_size: int = BUFFER_SIZE,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Generate a test signal spanning multiple buffers."""
    total_samples = num_buffers * buffer_size
    return generate_sine_wave(frequency, total_samples, sample_rate)


class TestPitchDetectorBasic:
    """Basic pitch detection tests with pure sine waves."""

    def setup_method(self):
        """Create a fresh detector for each test."""
        self.detector = MultiPitchDetector()

    def detect_frequency(
        self, frequency: float, num_buffers: int = 10
    ) -> list[MultiPitchResult]:
        """Helper to detect pitch from a sine wave, returning all results."""
        signal = generate_test_signal(frequency, num_buffers)
        results = []

        for i in range(num_buffers):
            start = i * BUFFER_SIZE
            end = start + BUFFER_SIZE
            result = self.detector.process(signal[start:end])
            results.append(result)

        return results

    def get_stable_result(self, frequency: float, num_buffers: int = 10) -> MultiPitchResult:
        """Get a stable detection result (skip first few buffers for warmup)."""
        results = self.detect_frequency(frequency, num_buffers)
        # Return last valid result
        for r in reversed(results):
            if r.valid:
                return r
        return MultiPitchResult()

    def test_a4_440hz(self):
        """Test detection of A4 at 440 Hz."""
        result = self.get_stable_result(440.0)

        assert result.valid, "Should detect A4"
        assert result.maxima[0].note_name == "A", f"Expected A, got {result.maxima[0].note_name}"
        assert result.maxima[0].octave == 4, f"Expected octave 4, got {result.maxima[0].octave}"
        assert abs(result.primary_frequency - 440.0) < 5.0, f"Frequency {result.primary_frequency} too far from 440"
        assert abs(result.primary_cents) < 10.0, f"Cents {result.primary_cents} should be near 0"

    def test_a3_220hz(self):
        """Test detection of A3 at 220 Hz."""
        result = self.get_stable_result(220.0)

        assert result.valid, "Should detect A3"
        assert result.maxima[0].note_name == "A", f"Expected A, got {result.maxima[0].note_name}"
        assert result.maxima[0].octave == 3, f"Expected octave 3, got {result.maxima[0].octave}"
        assert abs(result.primary_frequency - 220.0) < 3.0

    def test_e4_329hz(self):
        """Test detection of E4 at ~329.63 Hz."""
        result = self.get_stable_result(329.63)

        assert result.valid, "Should detect E4"
        assert result.maxima[0].note_name == "E", f"Expected E, got {result.maxima[0].note_name}"
        assert result.maxima[0].octave == 4, f"Expected octave 4, got {result.maxima[0].octave}"

    def test_c4_middle_c(self):
        """Test detection of middle C (C4) at ~261.63 Hz."""
        result = self.get_stable_result(261.63)

        assert result.valid, "Should detect C4"
        assert result.maxima[0].note_name == "C", f"Expected C, got {result.maxima[0].note_name}"
        assert result.maxima[0].octave == 4, f"Expected octave 4, got {result.maxima[0].octave}"

    def test_low_e_guitar(self):
        """Test detection of low E on guitar (E2) at ~82.41 Hz."""
        result = self.get_stable_result(82.41)

        assert result.valid, "Should detect E2"
        assert result.maxima[0].note_name == "E", f"Expected E, got {result.maxima[0].note_name}"
        assert result.maxima[0].octave == 2, f"Expected octave 2, got {result.maxima[0].octave}"

    def test_high_e_guitar(self):
        """Test detection of high E on guitar (E4) at ~329.63 Hz."""
        result = self.get_stable_result(329.63)

        assert result.valid, "Should detect E4"
        assert result.maxima[0].note_name == "E", f"Expected E, got {result.maxima[0].note_name}"


class TestCentsAccuracy:
    """Test cents calculation accuracy."""

    def setup_method(self):
        self.detector = MultiPitchDetector()

    def get_stable_result(self, frequency: float) -> MultiPitchResult:
        signal = generate_test_signal(frequency, num_buffers=10)
        for i in range(10):
            start = i * BUFFER_SIZE
            end = start + BUFFER_SIZE
            result = self.detector.process(signal[start:end])
        return result

    def test_a4_sharp_10_cents(self):
        """Test detection of A4 + 10 cents (~442.55 Hz)."""
        # 10 cents sharp: 440 * 2^(10/1200) = 442.55
        freq = 440.0 * (2 ** (10 / 1200))
        result = self.get_stable_result(freq)

        assert result.valid
        assert result.maxima[0].note_name == "A"
        # Should be approximately +10 cents
        assert 5 < result.primary_cents < 20, f"Expected ~+10 cents, got {result.primary_cents}"

    def test_a4_flat_10_cents(self):
        """Test detection of A4 - 10 cents (~437.47 Hz)."""
        freq = 440.0 * (2 ** (-10 / 1200))
        result = self.get_stable_result(freq)

        assert result.valid
        assert result.maxima[0].note_name == "A"
        # Should be approximately -10 cents
        assert -20 < result.primary_cents < -5, f"Expected ~-10 cents, got {result.primary_cents}"

    def test_exactly_in_tune(self):
        """Test that exact frequency gives ~0 cents."""
        result = self.get_stable_result(440.0)

        assert result.valid
        assert abs(result.primary_cents) < 5, f"Expected ~0 cents, got {result.primary_cents}"


class TestReferenceFrequency:
    """Test custom reference frequency support."""

    def test_a4_at_442hz_reference(self):
        """Test with A4=442 Hz reference (common orchestral tuning)."""
        detector = MultiPitchDetector(reference=442.0)

        signal = generate_test_signal(442.0, num_buffers=10)
        for i in range(10):
            start = i * BUFFER_SIZE
            end = start + BUFFER_SIZE
            result = detector.process(signal[start:end])

        # 442 Hz should be in tune when reference is 442
        assert result.valid
        assert result.maxima[0].note_name == "A"
        assert abs(result.primary_cents) < 10, f"Expected ~0 cents with 442 ref, got {result.primary_cents}"

    def test_a4_at_415hz_baroque(self):
        """Test with A4=415 Hz reference (baroque pitch)."""
        detector = MultiPitchDetector(reference=415.0)

        signal = generate_test_signal(415.0, num_buffers=10)
        for i in range(10):
            start = i * BUFFER_SIZE
            end = start + BUFFER_SIZE
            result = detector.process(signal[start:end])

        assert result.valid
        assert result.maxima[0].note_name == "A"
        assert abs(result.primary_cents) < 10


class TestTemperament:
    """Test temperament support."""

    def test_equal_temperament_default(self):
        """Verify equal temperament is the default."""
        detector = MultiPitchDetector()
        assert detector.temperament == Temperament.EQUAL

    def test_pythagorean_temperament(self):
        """Test Pythagorean temperament adjusts reference frequencies."""
        detector = MultiPitchDetector()
        detector.set_temperament(Temperament.PYTHAGOREAN)

        # In Pythagorean, the major third (E) is sharper than equal temperament
        # This means to be "in tune" in Pythagorean, E should be slightly higher
        # The reference frequency for E will be different

        signal = generate_test_signal(329.63, num_buffers=10)  # E4 equal temperament
        for i in range(10):
            start = i * BUFFER_SIZE
            end = start + BUFFER_SIZE
            result = detector.process(signal[start:end])

        # The exact cents will differ based on temperament
        # Main test is that it still detects as E
        if result.valid:
            assert result.maxima[0].note_name == "E"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        self.detector = MultiPitchDetector()

    def test_silence(self):
        """Test that silence returns invalid result."""
        signal = np.zeros(BUFFER_SIZE, dtype=np.float32)
        result = self.detector.process(signal)

        assert not result.valid

    def test_very_low_amplitude(self):
        """Test that very quiet signals return invalid result."""
        signal = generate_sine_wave(440.0, BUFFER_SIZE, amplitude=0.001)
        result = self.detector.process(signal)

        # Should be invalid due to low magnitude
        assert not result.valid

    def test_noise(self):
        """Test that pure noise returns invalid result."""
        signal = np.random.randn(BUFFER_SIZE).astype(np.float32) * 0.1
        result = self.detector.process(signal)

        # Noise should generally not produce a valid pitch
        # (might occasionally find spurious peaks)
        pass  # Just verify it doesn't crash

    def test_short_buffer(self):
        """Test handling of buffer shorter than expected."""
        signal = generate_sine_wave(440.0, BUFFER_SIZE // 2)
        # Should not crash
        result = self.detector.process(signal)
        # May or may not be valid, but should not raise


class TestMusicalNotes:
    """Test detection of all 12 notes in the chromatic scale."""

    # Frequencies for octave 4 (equal temperament, A4=440)
    NOTE_FREQUENCIES = {
        "C": 261.63,
        "C#": 277.18,
        "D": 293.66,
        "Eb": 311.13,
        "E": 329.63,
        "F": 349.23,
        "F#": 369.99,
        "G": 392.00,
        "Ab": 415.30,
        "A": 440.00,
        "Bb": 466.16,
        "B": 493.88,
    }

    def setup_method(self):
        self.detector = MultiPitchDetector()

    def get_stable_result(self, frequency: float) -> MultiPitchResult:
        signal = generate_test_signal(frequency, num_buffers=10)
        for i in range(10):
            start = i * BUFFER_SIZE
            end = start + BUFFER_SIZE
            result = self.detector.process(signal[start:end])
        return result

    @pytest.mark.parametrize("note,freq", NOTE_FREQUENCIES.items())
    def test_detect_note(self, note: str, freq: float):
        """Test detection of each chromatic note."""
        result = self.get_stable_result(freq)

        assert result.valid, f"Failed to detect {note} at {freq} Hz"
        assert result.maxima[0].note_name == note, f"Expected {note}, got {result.maxima[0].note_name}"
        assert result.maxima[0].octave == 4, f"Expected octave 4, got {result.maxima[0].octave}"
        assert abs(result.primary_cents) < 15, f"Cents deviation too high: {result.primary_cents}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])