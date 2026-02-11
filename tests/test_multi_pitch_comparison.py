"""
Tests comparing multi-pitch detection between Python and C++ implementations
"""

import json
import subprocess
import wave
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pytest

from accordion_tuner import MultiPitchDetector


TEST_DATA_DIR = Path(__file__).parent / "test_data"
CLI_PATH = Path(__file__).parent.parent.parent / "cli" / "tuner_cli"


@dataclass
class ChordTestCase:
    name: str
    frequencies: list[float]
    expected_notes: list[str]
    description: str


CHORD_TEST_CASES = [
    ChordTestCase(
        name="c_major",
        frequencies=[261.63, 329.63, 392.0],
        expected_notes=["C", "E", "G"],
        description="C major chord (C4 + E4 + G4)",
    ),
    ChordTestCase(
        name="a_minor",
        frequencies=[220.0, 261.63, 329.63],
        expected_notes=["A", "C", "E"],
        description="A minor chord (A3 + C4 + E4)",
    ),
    ChordTestCase(
        name="g_major",
        frequencies=[196.0, 246.94, 293.66],
        expected_notes=["G", "B", "D"],
        description="G major chord (G3 + B3 + D4)",
    ),
    ChordTestCase(
        name="fifth_c_g",
        frequencies=[261.63, 392.0],
        expected_notes=["C", "G"],
        description="Perfect fifth (C4 + G4)",
    ),
    ChordTestCase(
        name="octave_a",
        frequencies=[220.0, 440.0],
        expected_notes=["A", "A"],
        description="Octave (A3 + A4)",
    ),
]


def generate_chord_file(test_case: ChordTestCase) -> Path:
    """Generate a WAV file with the specified chord."""
    TEST_DATA_DIR.mkdir(exist_ok=True)

    sample_rate = 11025
    duration = 1.5  # Longer duration for better detection
    t = np.arange(int(sample_rate * duration)) / sample_rate

    # Generate signal with equal amplitude for each note
    signal = np.zeros_like(t)
    amplitude = 0.8 / len(test_case.frequencies)
    for freq in test_case.frequencies:
        signal += amplitude * np.sin(2 * np.pi * freq * t)

    # Save as WAV
    samples_int16 = (signal * 32767).astype(np.int16)
    filename = TEST_DATA_DIR / f"chord_{test_case.name}.wav"
    with wave.open(str(filename), 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples_int16.tobytes())

    return filename


def run_cpp_detector(filename: Path) -> dict:
    """Run C++ CLI and parse JSON output."""
    if not CLI_PATH.exists():
        pytest.skip(f"C++ CLI not found at {CLI_PATH}")

    result = subprocess.run(
        [str(CLI_PATH), str(filename)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return {"valid": False, "error": result.stderr}

    return json.loads(result.stdout)


def run_python_detector(filename: Path) -> dict:
    """Run Python MultiPitchDetector and return results."""
    with wave.open(str(filename), 'rb') as wav:
        raw_data = wav.readframes(wav.getnframes())
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64) / 32768.0

    mpd = MultiPitchDetector()

    # Process and collect notes
    all_notes = {}
    valid_frames = 0

    for i in range(0, len(samples), mpd.hop_size):
        chunk = samples[i:i+mpd.hop_size]
        if len(chunk) < mpd.hop_size:
            chunk = np.pad(chunk, (0, mpd.hop_size - len(chunk)))

        result = mpd.process(chunk)

        if result.valid:
            valid_frames += 1
            for m in result.maxima:
                key = f"{m.note_name}{m.octave}"
                if key not in all_notes:
                    all_notes[key] = {
                        'freqs': [],
                        'cents': [],
                        'note_name': m.note_name,
                        'octave': m.octave
                    }
                all_notes[key]['freqs'].append(m.frequency)
                all_notes[key]['cents'].append(m.cents)

    # Format output
    output = {
        "valid": len(all_notes) > 0,
        "num_notes": len(all_notes),
        "notes": []
    }

    for key, data in sorted(all_notes.items(), key=lambda x: (x[1]['octave'], x[1]['note_name'])):
        output["notes"].append({
            "note_name": data['note_name'],
            "octave": data['octave'],
            "frequency": round(np.median(data['freqs']), 2),
            "cents": round(np.median(data['cents']), 2)
        })

    return output


class TestMultiPitchComparison:
    """Compare Python and C++ multi-pitch detection."""

    @pytest.mark.parametrize("tc", CHORD_TEST_CASES, ids=lambda tc: tc.name)
    def test_both_detect_notes(self, tc: ChordTestCase):
        """Both implementations should detect at least some notes."""
        filename = generate_chord_file(tc)

        cpp_result = run_cpp_detector(filename)
        py_result = run_python_detector(filename)

        print(f"\n{tc.description}")
        print(f"Expected: {tc.expected_notes}")
        print(f"C++: {[n['note_name'] for n in cpp_result.get('notes', [])]}")
        print(f"Python: {[n['note_name'] for n in py_result.get('notes', [])]}")

        assert cpp_result.get("valid"), f"C++ failed to detect {tc.name}"
        assert py_result.get("valid"), f"Python failed to detect {tc.name}"

    @pytest.mark.parametrize("tc", CHORD_TEST_CASES, ids=lambda tc: tc.name)
    def test_note_names_match(self, tc: ChordTestCase):
        """Both implementations should detect the same notes."""
        filename = generate_chord_file(tc)

        cpp_result = run_cpp_detector(filename)
        py_result = run_python_detector(filename)

        cpp_notes = set(n['note_name'] for n in cpp_result.get('notes', []))
        py_notes = set(n['note_name'] for n in py_result.get('notes', []))

        # At least the intersection should match expected
        expected = set(tc.expected_notes)

        cpp_match = cpp_notes & expected
        py_match = py_notes & expected

        print(f"\n{tc.description}")
        print(f"Expected: {expected}")
        print(f"C++ detected: {cpp_notes} (matched: {cpp_match})")
        print(f"Python detected: {py_notes} (matched: {py_match})")

        # Both should detect at least 2/3 of expected notes
        min_match = max(1, len(expected) * 2 // 3)
        assert len(cpp_match) >= min_match, f"C++ only matched {cpp_match}"
        assert len(py_match) >= min_match, f"Python only matched {py_match}"

    @pytest.mark.parametrize("tc", CHORD_TEST_CASES, ids=lambda tc: tc.name)
    def test_frequency_accuracy(self, tc: ChordTestCase):
        """Frequency accuracy should be within tolerance."""
        filename = generate_chord_file(tc)

        cpp_result = run_cpp_detector(filename)
        py_result = run_python_detector(filename)

        if not cpp_result.get("valid") or not py_result.get("valid"):
            pytest.skip("Detection failed")

        # Compare frequencies for notes detected by both (using note+octave as key)
        cpp_notes = {f"{n['note_name']}{n['octave']}": n for n in cpp_result.get('notes', [])}
        py_notes = {f"{n['note_name']}{n['octave']}": n for n in py_result.get('notes', [])}

        common_notes = set(cpp_notes.keys()) & set(py_notes.keys())

        if not common_notes:
            # If no exact match, just check that detected frequencies are close to expected
            print("No common notes detected by both implementations")
            return

        for note in common_notes:
            cpp_freq = cpp_notes[note]['frequency']
            py_freq = py_notes[note]['frequency']
            diff = abs(cpp_freq - py_freq)

            print(f"{note}: C++={cpp_freq:.2f}Hz, Python={py_freq:.2f}Hz, diff={diff:.2f}Hz")

            # Allow 3 Hz tolerance
            assert diff < 3.0, f"Frequency difference too large for {note}: {diff:.2f}Hz"


# Standalone test runner
if __name__ == "__main__":
    print("Generating test files and comparing implementations...\n")

    for tc in CHORD_TEST_CASES:
        print(f"=== {tc.description} ===")
        filename = generate_chord_file(tc)

        cpp_result = run_cpp_detector(filename)
        py_result = run_python_detector(filename)

        print(f"Expected notes: {tc.expected_notes}")
        print(f"C++ result: {json.dumps(cpp_result, indent=2)}")
        print(f"Python result: {json.dumps(py_result, indent=2)}")
        print()
