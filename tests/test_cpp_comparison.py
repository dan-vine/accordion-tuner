"""
Comparison tests between Python and C++ implementations.

This module provides:
1. Test signal generation that can be saved to files for C++ testing
2. Loading and comparison of results from both implementations
3. Tolerance-based comparison for pitch detection results

To use:
1. Run `python -m pytest tests/test_cpp_comparison.py --generate` to create test WAV files
2. Process the WAV files with the C++ implementation and save results as JSON
3. Run the comparison tests to verify both implementations agree
"""

import json
import wave
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import pytest

from accordion_tuner import MultiPitchDetector, SAMPLE_RATE, BUFFER_SIZE


# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"


@dataclass
class TestCase:
    """A test case with input signal and expected output."""
    name: str
    frequency: float
    expected_note: str
    expected_octave: int
    description: str = ""


# Standard test cases for comparison
COMPARISON_TEST_CASES = [
    TestCase("a4_440", 440.0, "A", 4, "Concert A"),
    TestCase("a3_220", 220.0, "A", 3, "A below middle C"),
    TestCase("a5_880", 880.0, "A", 5, "A above concert A"),
    TestCase("c4_middle", 261.63, "C", 4, "Middle C"),
    TestCase("e2_low_guitar", 82.41, "E", 2, "Low E string on guitar"),
    TestCase("e4_high_guitar", 329.63, "E", 4, "High E string on guitar"),
    TestCase("g3_196", 196.0, "G", 3, "G3"),
    TestCase("b4_494", 493.88, "B", 4, "B4"),
    # Slightly detuned signals
    TestCase("a4_sharp_5c", 440.0 * (2 ** (5/1200)), "A", 4, "A4 +5 cents"),
    TestCase("a4_flat_5c", 440.0 * (2 ** (-5/1200)), "A", 4, "A4 -5 cents"),
    TestCase("a4_sharp_20c", 440.0 * (2 ** (20/1200)), "A", 4, "A4 +20 cents"),
    TestCase("a4_flat_20c", 440.0 * (2 ** (-20/1200)), "A", 4, "A4 -20 cents"),
]


def generate_sine_wave(
    frequency: float,
    duration_seconds: float,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Generate a sine wave."""
    num_samples = int(duration_seconds * sample_rate)
    t = np.arange(num_samples) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def save_wav(filename: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Save samples to a WAV file (16-bit mono)."""
    # Convert to 16-bit integers
    samples_int16 = (samples * 32767).astype(np.int16)

    with wave.open(str(filename), 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(samples_int16.tobytes())


def load_wav(filename: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file and return samples as float32."""
    with wave.open(str(filename), 'r') as wav:
        sample_rate = wav.getframerate()
        num_frames = wav.getnframes()
        raw_data = wav.readframes(num_frames)

        # Convert to numpy array
        samples_int16 = np.frombuffer(raw_data, dtype=np.int16)
        samples = samples_int16.astype(np.float32) / 32767.0

    return samples, sample_rate


def generate_test_files():
    """Generate WAV files for all test cases."""
    TEST_DATA_DIR.mkdir(exist_ok=True)

    duration = 1.0  # 1 second of audio

    for tc in COMPARISON_TEST_CASES:
        signal = generate_sine_wave(tc.frequency, duration)
        filename = TEST_DATA_DIR / f"{tc.name}.wav"
        save_wav(filename, signal)
        print(f"Generated: {filename}")

    # Also save test case metadata
    metadata = [asdict(tc) for tc in COMPARISON_TEST_CASES]
    with open(TEST_DATA_DIR / "test_cases.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Generated: {TEST_DATA_DIR / 'test_cases.json'}")


class TestPythonResults:
    """Run Python pitch detection on test files and save results."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure test data exists."""
        if not TEST_DATA_DIR.exists():
            generate_test_files()
        self.detector = MultiPitchDetector()

    def process_file(self, filename: Path) -> dict:
        """Process a WAV file and return aggregated results."""
        samples, _ = load_wav(filename)

        results = []
        num_buffers = len(samples) // BUFFER_SIZE

        for i in range(num_buffers):
            start = i * BUFFER_SIZE
            end = start + BUFFER_SIZE
            result = self.detector.process(samples[start:end])
            if result.valid and result.maxima:
                primary = result.maxima[0]
                results.append({
                    "frequency": primary.frequency,
                    "note_name": primary.note_name,
                    "octave": primary.octave,
                    "cents": primary.cents,
                })

        if not results:
            return {"valid": False}

        # Aggregate results (use median for stability)
        frequencies = [r["frequency"] for r in results]
        cents_values = [r["cents"] for r in results]

        return {
            "valid": True,
            "frequency_median": float(np.median(frequencies)),
            "frequency_std": float(np.std(frequencies)),
            "cents_median": float(np.median(cents_values)),
            "cents_std": float(np.std(cents_values)),
            "note_name": results[-1]["note_name"],  # Should be consistent
            "octave": results[-1]["octave"],
            "num_valid_frames": len(results),
        }

    @pytest.mark.parametrize("tc", COMPARISON_TEST_CASES, ids=lambda tc: tc.name)
    def test_python_detection(self, tc: TestCase):
        """Test Python detection on generated signals."""
        filename = TEST_DATA_DIR / f"{tc.name}.wav"

        if not filename.exists():
            generate_test_files()

        result = self.process_file(filename)

        assert result["valid"], f"Failed to detect pitch for {tc.name}"
        assert result["note_name"] == tc.expected_note, \
            f"Expected {tc.expected_note}, got {result['note_name']}"
        assert result["octave"] == tc.expected_octave, \
            f"Expected octave {tc.expected_octave}, got {result['octave']}"

    def test_save_python_results(self):
        """Process all test files and save results for C++ comparison."""
        results = {}

        for tc in COMPARISON_TEST_CASES:
            filename = TEST_DATA_DIR / f"{tc.name}.wav"
            if filename.exists():
                results[tc.name] = self.process_file(filename)

        output_file = TEST_DATA_DIR / "python_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved Python results to: {output_file}")


class TestCppComparison:
    """Compare Python results with C++ results."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load results if available."""
        self.python_results = None
        self.cpp_results = None

        python_file = TEST_DATA_DIR / "python_results.json"
        cpp_file = TEST_DATA_DIR / "cpp_results.json"

        if python_file.exists():
            with open(python_file) as f:
                self.python_results = json.load(f)

        if cpp_file.exists():
            with open(cpp_file) as f:
                self.cpp_results = json.load(f)

    def test_results_exist(self):
        """Check that we have results to compare."""
        if self.cpp_results is None:
            pytest.skip("C++ results not available (run C++ tests first)")

        assert self.python_results is not None, "Python results not generated"

    @pytest.mark.parametrize("tc", COMPARISON_TEST_CASES, ids=lambda tc: tc.name)
    def test_compare_detection(self, tc: TestCase):
        """Compare Python and C++ detection results."""
        if self.cpp_results is None:
            pytest.skip("C++ results not available")

        py = self.python_results.get(tc.name)
        cpp = self.cpp_results.get(tc.name)

        if py is None or cpp is None:
            pytest.skip(f"Results missing for {tc.name}")

        if not py.get("valid") or not cpp.get("valid"):
            pytest.skip(f"Detection failed for {tc.name}")

        # Compare note detection (should match exactly)
        assert py["note_name"] == cpp["note_name"], \
            f"Note mismatch: Python={py['note_name']}, C++={cpp['note_name']}"

        assert py["octave"] == cpp["octave"], \
            f"Octave mismatch: Python={py['octave']}, C++={cpp['octave']}"

        # Compare frequency (within 2 Hz tolerance)
        freq_diff = abs(py["frequency_median"] - cpp["frequency_median"])
        assert freq_diff < 2.0, \
            f"Frequency difference too large: {freq_diff:.2f} Hz"

        # Compare cents with frequency-dependent tolerance
        # Low frequencies have less resolution, so allow more tolerance
        frequency = tc.frequency
        if frequency < 100:
            cents_tolerance = 5.0  # Bass notes - FFT resolution limited
        elif frequency < 200:
            cents_tolerance = 3.0  # Low notes
        else:
            cents_tolerance = 2.0  # Higher notes - tighter tolerance

        cents_diff = abs(py["cents_median"] - cpp["cents_median"])
        assert cents_diff < cents_tolerance, \
            f"Cents difference too large: {cents_diff:.2f} (tolerance: {cents_tolerance} for {frequency:.0f} Hz)"


# Command-line entry point for generating test data
if __name__ == "__main__":
    import sys
    if "--generate" in sys.argv:
        generate_test_files()
    else:
        pytest.main([__file__, "-v"])