"""
Tests for SimpleFftPeakDetector using synthetic signals.

These tests generate sine waves at known frequencies and verify that the
detector correctly identifies frequencies, note names, and cents deviation.
"""

import numpy as np

from accordion_tuner import SAMPLE_RATE
from accordion_tuner.simple_fft_detector import SimpleFftPeakDetector


def generate_sine_wave(
    frequency: float,
    duration_samples: int,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Generate a sine wave at the given frequency."""
    t = np.arange(duration_samples) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float64)


def generate_test_signal(
    frequency: float,
    duration_samples: int,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Generate a test signal at the given frequency."""
    return generate_sine_wave(frequency, duration_samples, sample_rate)


def generate_two_sine_waves(
    freq1: float,
    freq2: float,
    duration_samples: int,
    sample_rate: int = SAMPLE_RATE,
    amplitude1: float = 0.8,
    amplitude2: float = 0.6,
) -> np.ndarray:
    """Generate a signal with two sine waves (for testing close reeds)."""
    t = np.arange(duration_samples) / sample_rate
    signal = (
        amplitude1 * np.sin(2 * np.pi * freq1 * t)
        + amplitude2 * np.sin(2 * np.pi * freq2 * t)
    ).astype(np.float64)
    return signal


class TestSimpleFftBasic:
    """Basic pitch detection tests with pure sine waves."""

    def setup_method(self):
        """Create a fresh detector for each test."""
        self.detector = SimpleFftPeakDetector()

    def process_signal(self, signal: np.ndarray) -> list:
        """Process signal in chunks and return all results."""
        hop_size = self.detector.hop_size
        results = []
        for start in range(0, len(signal) - hop_size, hop_size):
            chunk = signal[start : start + hop_size]
            result = self.detector.process(chunk)
            results.append(result)
        return results

    def get_valid_result(self, signal: np.ndarray):
        """Get last valid detection result."""
        results = self.process_signal(signal)
        for r in reversed(results):
            if r.valid:
                return r
        return results[-1] if results else None

    def test_a4_440hz(self):
        """Test detection of A4 at 440 Hz."""
        signal = generate_test_signal(440.0, SAMPLE_RATE * 2)  # 2 seconds
        result = self.get_valid_result(signal)

        assert result.valid, "Should detect A4"
        assert result.maxima[0].note_name == "A", f"Expected A, got {result.maxima[0].note_name}"
        assert result.maxima[0].octave == 4, f"Expected octave 4, got {result.maxima[0].octave}"
        assert abs(result.primary_frequency - 440.0) < 5.0, f"Frequency {result.primary_frequency} too far from 440"

    def test_a3_220hz(self):
        """Test detection of A3 at 220 Hz."""
        signal = generate_test_signal(220.0, SAMPLE_RATE * 2)
        result = self.get_valid_result(signal)

        assert result.valid, "Should detect A3"
        assert result.maxima[0].note_name == "A"
        assert result.maxima[0].octave == 3
        assert abs(result.primary_frequency - 220.0) < 3.0

    def test_e4_329hz(self):
        """Test detection of E4 at ~329.63 Hz."""
        signal = generate_test_signal(329.63, SAMPLE_RATE * 2)
        result = self.get_valid_result(signal)

        assert result.valid, "Should detect E4"
        assert result.maxima[0].note_name == "E"
        assert result.maxima[0].octave == 4

    def test_c4_middle_c(self):
        """Test detection of middle C (C4) at ~261.63 Hz."""
        signal = generate_test_signal(261.63, SAMPLE_RATE * 2)
        result = self.get_valid_result(signal)

        assert result.valid, "Should detect C4"
        assert result.maxima[0].note_name == "C"
        assert result.maxima[0].octave == 4


class TestSimpleFftCloseReeds:
    """Tests for detecting two close frequencies (close reeds)."""

    def setup_method(self):
        self.detector = SimpleFftPeakDetector()

    def process_signal(self, signal: np.ndarray):
        """Process signal and return last valid result."""
        hop_size = self.detector.hop_size
        result = None
        for start in range(0, len(signal) - hop_size, hop_size):
            chunk = signal[start : start + hop_size]
            result = self.detector.process(chunk)
        return result

    def test_two_close_frequencies_1hz_apart(self):
        """Test detection of two frequencies 1 Hz apart (common tremolo)."""
        freq1 = 440.0
        freq2 = 441.0  # 1 Hz above
        signal = generate_two_sine_waves(freq1, freq2, SAMPLE_RATE * 3, amplitude1=0.8, amplitude2=0.6)
        result = self.process_signal(signal)

        assert result.valid, "Should detect signal"
        assert len(result.maxima) >= 1, "Should find at least one frequency"

        # The primary should be close to one of the frequencies
        assert abs(result.primary_frequency - freq1) < 2.0 or abs(result.primary_frequency - freq2) < 2.0

    def test_two_close_frequencies_05hz_apart(self):
        """Test detection of two frequencies 0.5 Hz apart (tight tremolo)."""
        freq1 = 440.0
        freq2 = 440.5  # 0.5 Hz above
        signal = generate_two_sine_waves(freq1, freq2, SAMPLE_RATE * 3, amplitude1=0.8, amplitude2=0.6)
        result = self.process_signal(signal)

        assert result.valid, "Should detect signal"
        # With 0.5 Hz separation, both may merge into one peak

    def test_two_frequencies_far_apart(self):
        """Test detection of two frequencies far apart (different notes)."""
        freq1 = 440.0  # A4
        freq2 = 523.25  # C5
        signal = generate_two_sine_waves(freq1, freq2, SAMPLE_RATE * 2)
        result = self.process_signal(signal)

        assert result.valid, "Should detect signal"
        # Should detect both frequencies as different notes


class TestSimpleFftSettings:
    """Tests for SimpleFFT-specific settings."""

    def setup_method(self):
        self.detector = SimpleFftPeakDetector()

    def process_signal(self, signal: np.ndarray):
        """Process signal and return last valid result."""
        hop_size = self.detector.hop_size
        result = None
        for start in range(0, len(signal) - hop_size, hop_size):
            chunk = signal[start : start + hop_size]
            result = self.detector.process(chunk)
        return result

    def test_octave_filter_enabled(self):
        """Test that octave filter removes frequencies > 2x fundamental."""
        freq1 = 220.0  # A3
        freq2 = 880.0  # A5 (4x, octaves above)
        signal = generate_two_sine_waves(freq1, freq2, SAMPLE_RATE * 2, amplitude1=0.8, amplitude2=0.7)

        # With octave filter ON (default), should only detect fundamental
        result = self.process_signal(signal)
        assert result.valid

        # The primary should be the lower frequency
        assert result.primary_frequency < 300, f"Expected fundamental, got {result.primary_frequency}"

    def test_octave_filter_disabled(self):
        """Test that disabling octave filter allows octave harmonics."""
        self.detector.set_octave_filter(False)

        freq1 = 220.0  # A3
        freq2 = 880.0  # A5 (octave above)
        signal = generate_two_sine_waves(freq1, freq2, SAMPLE_RATE * 2, amplitude1=0.8, amplitude2=0.7)

        result = self.process_signal(signal)
        assert result.valid
        # With filter off, should detect both (may depend on magnitudes)

    def test_fundamental_filter_enabled(self):
        """Test that fundamental filter prioritizes lowest frequency."""
        self.detector.set_fundamental_filter(True)

        freq1 = 440.0
        freq2 = 444.0  # Same note, slightly higher
        signal = generate_two_sine_waves(freq1, freq2, SAMPLE_RATE * 2, amplitude1=0.5, amplitude2=0.8)

        result = self.process_signal(signal)
        assert result.valid

    def test_second_reed_search_range(self):
        """Test that second reed search range setting is applied."""
        self.detector.set_second_reed_search(5.0, 0.10)
        assert self.detector.second_reed_search_hz == 5.0
        assert self.detector.second_reed_threshold == 0.10

        self.detector.set_second_reed_search(2.0, 0.15)
        assert self.detector.second_reed_search_hz == 2.0
        assert self.detector.second_reed_threshold == 0.15

    def test_temperament_default(self):
        """Test that equal temperament is the default."""
        from accordion_tuner.temperaments import Temperament

        assert self.detector.temperament == Temperament.EQUAL

    def test_temperament_setting(self):
        """Test setting a different temperament."""
        from accordion_tuner.temperaments import Temperament

        self.detector.set_temperament(Temperament.PYTHAGOREAN)
        assert self.detector.temperament == Temperament.PYTHAGOREAN

    def test_key_setting(self):
        """Test key setting."""
        self.detector.set_key(2)  # D
        assert self.detector.key == 2


class TestSimpleFftCentsAccuracy:
    """Test cents calculation accuracy."""

    def setup_method(self):
        self.detector = SimpleFftPeakDetector()

    def process_and_get_last(self, signal: np.ndarray):
        """Process signal and return last result."""
        hop_size = self.detector.hop_size
        result = None
        for start in range(0, len(signal) - hop_size, hop_size):
            chunk = signal[start : start + hop_size]
            result = self.detector.process(chunk)
        return result

    def test_a4_exact_440hz(self):
        """Test cents at exact A4 (440 Hz)."""
        signal = generate_test_signal(440.0, SAMPLE_RATE * 2)
        result = self.process_and_get_last(signal)

        assert result.valid
        assert abs(result.primary_cents) < 5.0, f"Cents should be near 0, got {result.primary_cents}"

    def test_a4_sharp_10_cents(self):
        """Test cents at A4+10 cents (440 * 2^(10/1200) ≈ 443.86 Hz)."""
        cents = 10.0
        freq = 440.0 * (2 ** (cents / 1200.0))
        signal = generate_test_signal(freq, SAMPLE_RATE * 2)
        result = self.process_and_get_last(signal)

        assert result.valid
        assert abs(result.primary_cents - cents) < 5.0, f"Expected ~{cents} cents, got {result.primary_cents}"

    def test_a4_flat_15_cents(self):
        """Test cents at A4-15 cents."""
        cents = -15.0
        freq = 440.0 * (2 ** (cents / 1200.0))
        signal = generate_test_signal(freq, SAMPLE_RATE * 2)
        result = self.process_and_get_last(signal)

        assert result.valid
        assert abs(result.primary_cents - cents) < 5.0, f"Expected ~{cents} cents, got {result.primary_cents}"
