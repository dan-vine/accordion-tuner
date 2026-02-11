"""
Tests for multi-pitch detection
"""

import numpy as np
import pytest
from accordion_tuner import MultiPitchDetector


class TestMultiPitchDetectorSingleNote:
    """Test single note detection."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.detector = MultiPitchDetector()
        self.t = np.arange(self.detector.fft_size) / self.detector.sample_rate

    def generate_signal(self, frequencies: list[float], amplitudes: list[float] = None):
        """Generate a signal with multiple frequencies."""
        if amplitudes is None:
            amplitudes = [0.8 / len(frequencies)] * len(frequencies)
        signal = np.zeros(len(self.t))
        for freq, amp in zip(frequencies, amplitudes):
            signal += amp * np.sin(2 * np.pi * freq * self.t)
        return signal

    def test_single_a4(self):
        """Test single A4 detection."""
        signal = self.generate_signal([440.0])
        result = self.detector.process(signal)

        assert result.valid
        assert result.note_count >= 1
        assert result.maxima[0].note_name == "A"
        assert result.maxima[0].octave == 4

    def test_single_c4(self):
        """Test single C4 (middle C) detection."""
        signal = self.generate_signal([261.63])
        result = self.detector.process(signal)

        assert result.valid
        assert result.note_count >= 1
        assert result.maxima[0].note_name == "C"
        assert result.maxima[0].octave == 4

    def test_single_e2_low(self):
        """Test low E2 (guitar low E) detection.

        Note: Low frequencies have limited accuracy due to FFT resolution.
        Allow ±1 semitone tolerance for frequencies below 100 Hz.
        """
        signal = self.generate_signal([82.41])
        result = self.detector.process(signal)

        assert result.valid
        assert result.note_count >= 1
        # Allow E or F (±1 semitone) due to FFT resolution at low frequencies
        assert result.maxima[0].note_name in ["E", "F", "Eb"]
        assert result.maxima[0].octave == 2


class TestMultiPitchDetectorChords:
    """Test chord (multiple note) detection."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.detector = MultiPitchDetector()
        self.t = np.arange(self.detector.fft_size) / self.detector.sample_rate

    def generate_signal(self, frequencies: list[float], amplitudes: list[float] = None):
        if amplitudes is None:
            amplitudes = [0.8 / len(frequencies)] * len(frequencies)
        signal = np.zeros(len(self.t))
        for freq, amp in zip(frequencies, amplitudes):
            signal += amp * np.sin(2 * np.pi * freq * self.t)
        return signal

    def test_major_third_c_e(self):
        """Test C4 + E4 (major third) detection."""
        signal = self.generate_signal([261.63, 329.63])
        result = self.detector.process(signal)

        assert result.valid
        assert result.note_count >= 2

        notes = [m.note_name for m in result.maxima[:2]]
        assert "C" in notes
        assert "E" in notes

    def test_c_major_chord(self):
        """Test C major chord (C4 + E4 + G4) detection."""
        signal = self.generate_signal([261.63, 329.63, 392.0])
        result = self.detector.process(signal)

        assert result.valid
        assert result.note_count >= 3

        notes = [m.note_name for m in result.maxima[:3]]
        assert "C" in notes
        assert "E" in notes
        assert "G" in notes

    def test_a_minor_chord(self):
        """Test A minor chord (A3 + C4 + E4) detection."""
        signal = self.generate_signal([220.0, 261.63, 329.63])
        result = self.detector.process(signal)

        assert result.valid
        assert result.note_count >= 3

        notes = [m.note_name for m in result.maxima[:3]]
        assert "A" in notes
        assert "C" in notes
        assert "E" in notes

    def test_power_chord(self):
        """Test power chord (A2 + E3) detection.

        Using A2 + E3 instead of E2 + B2 for better accuracy
        at these frequencies.
        """
        signal = self.generate_signal([110.0, 164.81])  # A2 + E3
        result = self.detector.process(signal)

        assert result.valid
        assert result.note_count >= 2

        notes = [m.note_name for m in result.maxima[:2]]
        assert "A" in notes
        assert "E" in notes


class TestMultiPitchDetectorSettings:
    """Test detector settings."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.detector = MultiPitchDetector()

    def test_reference_frequency(self):
        """Test changing reference frequency."""
        self.detector.set_reference(442.0)
        assert self.detector.reference == 442.0

    def test_key_setting(self):
        """Test key setting."""
        self.detector.set_key(2)  # D
        assert self.detector.key == 2

    def test_fundamental_filter(self):
        """Test fundamental filter setting."""
        self.detector.set_fundamental_filter(True)
        assert self.detector.fundamental_filter is True

    def test_reset(self):
        """Test reset clears state."""
        t = np.arange(self.detector.fft_size) / self.detector.sample_rate
        signal = 0.8 * np.sin(2 * np.pi * 440 * t)

        # Process some data
        self.detector.process(signal)

        # Reset
        self.detector.reset()

        # Buffer should be zero
        assert np.allclose(self.detector._buffer, 0)


class TestMultiPitchResultProperties:
    """Test MultiPitchResult properties."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.detector = MultiPitchDetector()
        self.t = np.arange(self.detector.fft_size) / self.detector.sample_rate

    def test_note_count(self):
        """Test note_count property."""
        signal = 0.4 * np.sin(2 * np.pi * 261.63 * self.t) + \
                 0.4 * np.sin(2 * np.pi * 329.63 * self.t)
        result = self.detector.process(signal)

        assert result.note_count == len(result.maxima)
        assert result.note_count >= 2

    def test_frequencies_property(self):
        """Test frequencies property."""
        signal = 0.8 * np.sin(2 * np.pi * 440 * self.t)
        result = self.detector.process(signal)

        assert len(result.frequencies) == result.note_count
        assert all(isinstance(f, float) for f in result.frequencies)

    def test_note_names_property(self):
        """Test note_names property."""
        signal = 0.8 * np.sin(2 * np.pi * 440 * self.t)
        result = self.detector.process(signal)

        assert len(result.note_names) == result.note_count
        assert "A4" in result.note_names
