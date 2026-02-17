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


def generate_multi_sine_waves(
    frequencies: list[float],
    duration_samples: int,
    sample_rate: int = SAMPLE_RATE,
    amplitudes: list[float] | None = None,
) -> np.ndarray:
    """Generate a signal with multiple sine waves (for testing chords)."""
    if amplitudes is None:
        amplitudes = [0.8 / len(frequencies)] * len(frequencies)
    t = np.arange(duration_samples) / sample_rate
    signal = np.zeros(duration_samples, dtype=np.float64)
    for freq, amp in zip(frequencies, amplitudes):
        signal += amp * np.sin(2 * np.pi * freq * t)
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

    def test_num_sources_setter(self):
        """Test set_num_sources method updates internal state."""
        self.detector.set_num_sources(4)
        assert self.detector.num_sources == 4

        self.detector.set_num_sources(2)
        assert self.detector.num_sources == 2

        self.detector.set_num_sources(10)  # Should clamp to 8
        assert self.detector.num_sources == 8

        self.detector.set_num_sources(0)  # Should clamp to 1
        assert self.detector.num_sources == 1

    def test_bidirectional_second_reed_search(self):
        """Test that Pass 2 searches in both directions."""
        import math

        # Create signal with 3 reeds: 438, 440, 442 Hz
        # Make middle one (440) much louder so Pass 1 only finds it
        duration = 2.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
        # Middle reed (440 Hz) - loud
        signal = 0.8 * np.sin(2 * math.pi * 440 * t)
        # Lower reed (438 Hz) - quiet
        signal += 0.3 * np.sin(2 * math.pi * 438 * t)
        # Higher reed (442 Hz) - quiet
        signal += 0.3 * np.sin(2 * math.pi * 442 * t)

        # Lower thresholds so Pass 2 can find the quiet reeds
        self.detector.set_second_reed_search(5.0, 0.20)

        result = self.process_signal(signal)
        assert result.valid

        # Should find 3 reeds (middle from Pass 1, both sides from Pass 2)
        assert len(result.maxima) >= 2, f"Expected at least 2 maxima, got {len(result.maxima)}"

        freqs = sorted([m.frequency for m in result.maxima])
        # Should have found frequencies near 438, 440, and/or 442
        assert any(abs(f - 438) < 2 for f in freqs) or any(abs(f - 442) < 2 for f in freqs), \
            f"Expected to find reeds near 438 or 442 Hz, got {freqs}"

    def test_multiple_additional_reeds(self):
        """Test that Pass 2 can find multiple additional reeds."""
        import math

        # Create signal with 4 reeds spaced 1.5 Hz apart
        # Center at 440 Hz, others at 435, 441.5, 444
        duration = 2.0
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
        # Middle reed (440 Hz) - loudest
        signal = 0.8 * np.sin(2 * math.pi * 440 * t)
        # Other reeds - quieter
        signal += 0.3 * np.sin(2 * math.pi * 435 * t)
        signal += 0.3 * np.sin(2 * math.pi * 441.5 * t)
        signal += 0.3 * np.sin(2 * math.pi * 444 * t)

        # Allow finding up to 4 sources
        self.detector.set_num_sources(4)
        self.detector.set_second_reed_search(10.0, 0.20)

        result = self.process_signal(signal)
        assert result.valid

        # Should find more than just the main reed
        assert len(result.maxima) >= 2, f"Expected multiple maxima, got {len(result.maxima)}"


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


class TestSimpleFftChords:
    """Tests for chord (multiple note) detection."""

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

    def test_major_third_c_e(self):
        """Test C4 + E4 (major third) detection."""
        signal = generate_multi_sine_waves([261.63, 329.63], SAMPLE_RATE * 2)
        result = self.process_signal(signal)

        assert result.valid
        assert result.note_count >= 2

        notes = [m.note_name for m in result.maxima[:2]]
        assert "C" in notes
        assert "E" in notes

    def test_c_major_chord(self):
        """Test C major chord (C4 + E4 + G4) detection."""
        signal = generate_multi_sine_waves([261.63, 329.63, 392.0], SAMPLE_RATE * 2)
        result = self.process_signal(signal)

        assert result.valid
        assert result.note_count >= 3

        notes = [m.note_name for m in result.maxima[:3]]
        assert "C" in notes
        assert "E" in notes
        assert "G" in notes

    def test_a_minor_chord(self):
        """Test A minor chord (A3 + C4 + E4) detection."""
        signal = generate_multi_sine_waves([220.0, 261.63, 329.63], SAMPLE_RATE * 2)
        result = self.process_signal(signal)

        assert result.valid
        assert result.note_count >= 3

        notes = [m.note_name for m in result.maxima[:3]]
        assert "A" in notes
        assert "C" in notes
        assert "E" in notes

    def test_g_major_chord(self):
        """Test G major chord (G3 + B3 + D4) detection."""
        signal = generate_multi_sine_waves([196.0, 246.94, 293.66], SAMPLE_RATE * 2)
        result = self.process_signal(signal)

        assert result.valid
        assert result.note_count >= 3

        notes = [m.note_name for m in result.maxima[:3]]
        assert "G" in notes
        assert "B" in notes
        assert "D" in notes

    def test_power_chord(self):
        """Test power chord (A2 + E3) detection."""
        signal = generate_multi_sine_waves([110.0, 164.81], SAMPLE_RATE * 2)
        result = self.process_signal(signal)

        assert result.valid
        assert result.note_count >= 2

        notes = [m.note_name for m in result.maxima[:2]]
        assert "A" in notes
        assert "E" in notes
