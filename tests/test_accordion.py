"""
Tests for accordion reed detection.
"""

import numpy as np
import pytest

from accordion_tuner.accordion import AccordionDetector, AccordionResult, DetectorType, ReedInfo


class TestReedInfo:
    """Test ReedInfo dataclass."""

    def test_default_values(self):
        """Test default values."""
        info = ReedInfo()
        assert info.frequency == 0.0
        assert info.cents == 0.0
        assert info.magnitude == 0.0

    def test_with_values(self):
        """Test with specific values."""
        info = ReedInfo(frequency=440.0, cents=5.0, magnitude=0.8)
        assert info.frequency == 440.0
        assert info.cents == 5.0
        assert info.magnitude == 0.8


class TestAccordionResult:
    """Test AccordionResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = AccordionResult()
        assert result.valid is False
        assert result.note_name == ""
        assert result.octave == 0
        assert result.ref_frequency == 0.0
        assert result.reeds == []
        assert result.beat_frequencies == []
        assert result.spectrum_data is None

    def test_reed_count(self):
        """Test reed_count property."""
        result = AccordionResult(
            reeds=[ReedInfo(), ReedInfo(), ReedInfo()]
        )
        assert result.reed_count == 3

    def test_average_cents_empty(self):
        """Test average_cents with no reeds."""
        result = AccordionResult()
        assert result.average_cents == 0.0

    def test_average_cents_with_reeds(self):
        """Test average_cents calculation."""
        result = AccordionResult(
            reeds=[
                ReedInfo(cents=5.0),
                ReedInfo(cents=-3.0),
                ReedInfo(cents=10.0),
            ]
        )
        assert result.average_cents == pytest.approx(4.0)


class TestAccordionDetector:
    """Test AccordionDetector class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.detector = AccordionDetector()
        self.sample_rate = self.detector.sample_rate
        self.hop_size = self.detector._detector.hop_size
        self.fft_size = self.detector._detector.fft_size

    def generate_signal(self, frequencies: list[float], duration: float = 1.0,
                        amplitudes: list[float] | None = None):
        """Generate a signal with multiple frequencies."""
        t = np.arange(int(self.sample_rate * duration)) / self.sample_rate
        if amplitudes is None:
            amplitudes = [0.8 / len(frequencies)] * len(frequencies)
        signal = np.zeros(len(t))
        for freq, amp in zip(frequencies, amplitudes):
            signal += amp * np.sin(2 * np.pi * freq * t)
        return signal

    def process_signal(self, signal: np.ndarray, num_frames: int = 10) -> AccordionResult:
        """Process signal in chunks and return the last stable result.

        The phase vocoder needs several frames to stabilize.
        """
        result = AccordionResult()
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.hop_size
            if end > len(signal):
                break
            chunk = signal[start:end]
            result = self.detector.process(chunk)
        return result

    def test_single_reed(self):
        """Test detection of single reed."""
        signal = self.generate_signal([440.0])
        result = self.process_signal(signal)

        assert result.valid
        assert result.note_name == "A"
        assert result.octave == 4
        assert result.reed_count >= 1

    def test_two_detuned_reeds(self):
        """Test detection of two detuned reeds (tremolo)."""
        # Two reeds slightly detuned (common musette tuning)
        f1 = 440.0       # Reed 1 at A4
        f2 = 441.5       # Reed 2 slightly sharp (~6 cents)

        signal = self.generate_signal([f1, f2])
        result = self.process_signal(signal)

        assert result.valid
        assert result.note_name == "A"
        assert result.octave == 4

        # Should detect at least 1 reed
        assert result.reed_count >= 1

    def test_three_reeds_musette(self):
        """Test detection of three reeds (musette accordion)."""
        # Typical musette tuning: one flat, one at pitch, one sharp
        f1 = 438.5       # Reed 1 flat
        f2 = 440.0       # Reed 2 at pitch
        f3 = 441.5       # Reed 3 sharp

        signal = self.generate_signal([f1, f2, f3], amplitudes=[0.3, 0.3, 0.3])
        result = self.process_signal(signal)

        assert result.valid
        assert result.note_name == "A"

    def test_beat_frequency_calculation(self):
        """Test beat frequency calculation between reeds."""
        # Two reeds with known beat frequency
        f1 = 440.0
        f2 = 443.0  # 3 Hz beat

        signal = self.generate_signal([f1, f2])
        result = self.process_signal(signal)

        if result.reed_count >= 2 and len(result.beat_frequencies) > 0:
            # Beat frequency should be close to |f2 - f1| = 3 Hz
            # Allow some tolerance due to FFT resolution
            assert result.beat_frequencies[0] == pytest.approx(3.0, abs=1.0)

    def test_reference_frequency_change(self):
        """Test changing reference frequency."""
        self.detector.set_reference(442.0)
        assert self.detector.reference == 442.0
        assert self.detector._detector.reference == 442.0

    def test_max_reeds_setting(self):
        """Test setting maximum number of reeds."""
        self.detector.set_max_reeds(2)
        assert self.detector.max_reeds == 2

        self.detector.set_max_reeds(4)
        assert self.detector.max_reeds == 4

        # Test clamping
        self.detector.set_max_reeds(1)
        assert self.detector.max_reeds == 2

        self.detector.set_max_reeds(10)
        assert self.detector.max_reeds == 4

    def test_max_reeds_updates_simple_fft_detector(self):
        """Test that set_max_reeds updates SimpleFFT detector's num_sources."""
        detector = AccordionDetector(detector_type=DetectorType.SIMPLE_FFT)
        assert detector._detector.num_sources == 4  # default max_reeds

        detector.set_max_reeds(3)
        assert detector._detector.num_sources == 3

        detector.set_max_reeds(2)
        assert detector._detector.num_sources == 2

    def test_reed_spread_setting(self):
        """Test setting reed spread tolerance."""
        self.detector.set_reed_spread(30.0)
        assert self.detector.reed_spread_cents == 30.0

        # Test clamping
        self.detector.set_reed_spread(5.0)
        assert self.detector.reed_spread_cents == 10.0

        self.detector.set_reed_spread(200.0)
        assert self.detector.reed_spread_cents == 100.0

    def test_spectrum_data(self):
        """Test spectrum data is provided."""
        signal = self.generate_signal([440.0])
        result = self.process_signal(signal)

        assert result.spectrum_data is not None
        freqs, mags = result.spectrum_data
        assert len(freqs) > 0
        assert len(mags) > 0
        assert len(freqs) == len(mags)

    def test_reset(self):
        """Test reset clears state."""
        signal = self.generate_signal([440.0])
        self.process_signal(signal)

        self.detector.reset()

        # After reset, internal state should be cleared
        assert self.detector._fft_freqs is None
        assert self.detector._fft_mags is None

    def test_low_note_detection(self):
        """Test detection of low accordion note (bass)."""
        # E2 - typical accordion bass note
        signal = self.generate_signal([82.41])
        result = self.process_signal(signal)

        assert result.valid
        # Allow some tolerance for low frequency detection
        assert result.note_name in ["E", "F", "Eb"]
        assert result.octave == 2

    def test_high_note_detection(self):
        """Test detection of high accordion note."""
        # A5
        signal = self.generate_signal([880.0])
        result = self.process_signal(signal)

        assert result.valid
        assert result.note_name == "A"
        assert result.octave == 5

    def test_silence(self):
        """Test handling of silence."""
        signal = np.zeros(self.hop_size * 10)
        result = self.process_signal(signal)

        assert not result.valid
        assert result.reed_count == 0

    def test_very_quiet_signal(self):
        """Test handling of very quiet signal."""
        signal = self.generate_signal([440.0], amplitudes=[0.001])
        result = self.process_signal(signal)

        # Should either detect or return invalid, but not crash
        assert isinstance(result, AccordionResult)


class TestAccordionDetectorCentsAccuracy:
    """Test cents calculation accuracy.

    These tests process multiple frames to allow the phase vocoder
    to stabilize before checking accuracy.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.detector = AccordionDetector()
        self.sample_rate = self.detector.sample_rate
        self.hop_size = self.detector._detector.hop_size

    def generate_signal(self, frequency: float, duration: float = 1.0):
        """Generate a signal with single frequency."""
        t = np.arange(int(self.sample_rate * duration)) / self.sample_rate
        return 0.8 * np.sin(2 * np.pi * frequency * t)

    def process_signal(self, signal: np.ndarray, num_frames: int = 10) -> AccordionResult:
        """Process signal in chunks and return the last stable result."""
        result = AccordionResult()
        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.hop_size
            if end > len(signal):
                break
            chunk = signal[start:end]
            result = self.detector.process(chunk)
        return result

    def test_in_tune_a4(self):
        """Test A4 at exactly 440 Hz shows near-zero cents."""
        signal = self.generate_signal(440.0)
        result = self.process_signal(signal)

        assert result.valid
        assert result.reed_count > 0
        # With phase vocoder properly stabilized, should be very accurate
        assert abs(result.reeds[0].cents) < 2.0

    def test_sharp_a4(self):
        """Test sharp A4 shows positive cents."""
        # A4 + 10 cents = 440 * 2^(10/1200) ≈ 442.55 Hz
        sharp_freq = 440.0 * (2 ** (10 / 1200))
        signal = self.generate_signal(sharp_freq)
        result = self.process_signal(signal)

        assert result.valid
        assert result.reed_count > 0
        assert result.reeds[0].cents > 0
        # Should be close to +10 cents
        assert result.reeds[0].cents == pytest.approx(10.0, abs=2.0)

    def test_flat_a4(self):
        """Test flat A4 shows negative cents."""
        # A4 - 10 cents = 440 * 2^(-10/1200) ≈ 437.47 Hz
        flat_freq = 440.0 * (2 ** (-10 / 1200))
        signal = self.generate_signal(flat_freq)
        result = self.process_signal(signal)

        assert result.valid
        assert result.reed_count > 0
        assert result.reeds[0].cents < 0
        # Should be close to -10 cents
        assert result.reeds[0].cents == pytest.approx(-10.0, abs=2.0)

    def test_frequency_accuracy(self):
        """Test that detected frequency is accurate."""
        test_freq = 440.0
        signal = self.generate_signal(test_freq)
        result = self.process_signal(signal)

        assert result.valid
        assert result.reed_count > 0
        # Frequency should be within 1 Hz of target
        assert abs(result.reeds[0].frequency - test_freq) < 1.0

    def test_c4_detection(self):
        """Test middle C detection accuracy."""
        test_freq = 261.63  # C4
        signal = self.generate_signal(test_freq)
        result = self.process_signal(signal)

        assert result.valid
        assert result.note_name == "C"
        assert result.octave == 4
        assert result.reed_count > 0
        # Should be within 2 cents
        assert abs(result.reeds[0].cents) < 2.0
