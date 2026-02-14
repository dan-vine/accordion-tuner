"""
Record close reeds for detection testing and analysis.

This script records:
1. Reed 1 alone
2. Reed 2 alone  
3. Both reeds together

Then analyzes the recordings to:
- Extract ground truth frequencies from individual reeds
- Test detection algorithms on the combined recording
- Generate a report with detection accuracy metrics

Usage:
    python record_close_reeds.py

Output files (in current directory):
    reed1_<note>_<timestamp>.npy      - Reed 1 recording
    reed2_<note>_<timestamp>.npy      - Reed 2 recording
    combined_<note>_<timestamp>.npy   - Combined recording
    report_<note>_<timestamp>.json    - Analysis report
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from accordion_tuner.accordion import AccordionDetector, DetectorType
from accordion_tuner.constants import SAMPLE_RATE


def wait_for_enter(prompt: str) -> None:
    """Wait for user to press Enter."""
    input(f"\n{prompt}")


def countdown(seconds: int = 3) -> None:
    """Countdown before recording."""
    for i in range(seconds, 0, -1):
        print(f"{i}...", end=" ", flush=True)
        time.sleep(1)
    print("GO!")


def record_audio(duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record audio for specified duration.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Audio sample rate
        
    Returns:
        Recorded audio as numpy array
    """
    import sounddevice as sd

    num_samples = int(duration * sample_rate)
    print(f"Recording {duration}s at {sample_rate}Hz...")

    audio = sd.rec(num_samples, samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()

    return audio.flatten()


def save_recording(audio: np.ndarray, filename: str) -> Path:
    """Save audio recording to file.
    
    Args:
        audio: Audio data
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    filepath = Path(filename)
    np.save(filepath, audio)
    print(f"Saved: {filepath} ({len(audio)} samples, {len(audio)/SAMPLE_RATE:.1f}s)")
    return filepath


def extract_ground_truth_frequency(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Extract precise frequency from single reed recording using high-resolution FFT.
    
    Uses a long window with zero-padding for sub-Hz resolution.
    
    Args:
        audio: Audio recording
        sample_rate: Sample rate
        
    Returns:
        Detected frequency in Hz
    """
    # Use full recording with windowing
    signal = audio.astype(np.float64)

    # Normalize
    signal_max = np.max(np.abs(signal))
    if signal_max < 1e-6:
        return 0.0
    signal = signal / signal_max

    # Apply window
    window = np.hamming(len(signal))
    windowed = signal * window

    # Zero-pad to 4x for better frequency resolution
    n_fft = len(signal) * 4
    spectrum = np.fft.rfft(windowed, n=n_fft)
    magnitudes = np.abs(spectrum)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

    # Find peak in accordion range (60-2000 Hz)
    valid_mask = (freqs >= 60) & (freqs <= 2000)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return 0.0

    # Find maximum
    peak_idx = valid_indices[np.argmax(magnitudes[valid_indices])]

    # Parabolic interpolation for sub-bin accuracy
    if peak_idx > 0 and peak_idx < len(magnitudes) - 1:
        y1, y2, y3 = magnitudes[peak_idx - 1], magnitudes[peak_idx], magnitudes[peak_idx + 1]
        denom = y1 - 2 * y2 + y3
        if abs(denom) > 1e-10:
            delta = 0.5 * (y1 - y3) / denom
            freq_step = freqs[1] - freqs[0]
            return float(freqs[peak_idx] + delta * freq_step)

    return float(freqs[peak_idx])


def detect_note_from_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> tuple[str, int]:
    """Auto-detect note from audio using ESPRIT detector.
    
    Args:
        audio: Audio recording
        sample_rate: Sample rate
        
    Returns:
        Tuple of (note_name, octave)
    """
    detector = AccordionDetector(detector_type=DetectorType.ESPRIT)

    # Process in chunks like the real app
    hop_size = 1024
    results = []

    for i in range(0, len(audio) - hop_size, hop_size):
        chunk = audio[i:i + hop_size]
        result = detector.process(chunk)
        if result.valid:
            results.append((result.note_name, result.octave))

    if not results:
        return ("Unknown", 0)

    # Return most common detection
    from collections import Counter
    most_common = Counter(results).most_common(1)[0][0]
    return most_common


def analyze_combined_recording(
    audio: np.ndarray,
    ground_truth_reed1: float,
    ground_truth_reed2: float,
    sample_rate: int = SAMPLE_RATE
) -> dict:
    """Analyze combined recording with detection algorithms.
    
    Args:
        audio: Combined recording
        ground_truth_reed1: Known frequency of reed 1
        ground_truth_reed2: Known frequency of reed 2
        sample_rate: Sample rate
        
    Returns:
        Dictionary with detection results
    """
    results = {
        "ground_truth": {
            "reed1_hz": ground_truth_reed1,
            "reed2_hz": ground_truth_reed2,
            "separation_hz": abs(ground_truth_reed2 - ground_truth_reed1),
            "separation_cents": 1200 * np.log2(ground_truth_reed2 / ground_truth_reed1) if ground_truth_reed1 > 0 else 0
        },
        "fft_detection": {},
        "esprit_detection": {}
    }

    hop_size = 1024

    # Test FFT detector
    fft_detector = AccordionDetector(detector_type=DetectorType.FFT)
    fft_readings = []

    for i in range(0, len(audio) - hop_size, hop_size):
        chunk = audio[i:i + hop_size]
        result = fft_detector.process(chunk)
        if result.valid and result.reed_count >= 1:
            fft_readings.append([r.frequency for r in result.reeds])

    if fft_readings:
        # Take median of readings for stability
        all_freqs = [f for reading in fft_readings for f in reading]
        if all_freqs:
            results["fft_detection"]["detected_frequencies"] = sorted(all_freqs[:2]) if len(all_freqs) >= 2 else all_freqs
            results["fft_detection"]["reed_count_detected"] = len(results["fft_detection"]["detected_frequencies"])

    # Test ESPRIT detector with default settings
    esprit_detector = AccordionDetector(detector_type=DetectorType.ESPRIT)
    esprit_detector.set_esprit_min_separation(0.1)
    esprit_readings = []

    for i in range(0, len(audio) - hop_size, hop_size):
        chunk = audio[i:i + hop_size]
        result = esprit_detector.process(chunk)
        if result.valid and result.reed_count >= 1:
            esprit_readings.append([r.frequency for r in result.reeds])

    if esprit_readings:
        all_freqs = [f for reading in esprit_readings for f in reading]
        if all_freqs:
            results["esprit_detection"]["detected_frequencies"] = sorted(all_freqs[:2]) if len(all_freqs) >= 2 else all_freqs
            results["esprit_detection"]["reed_count_detected"] = len(results["esprit_detection"]["detected_frequencies"])

    return results


def generate_report(
    note_name: str,
    octave: int,
    reed1_file: str,
    reed2_file: str,
    combined_file: str,
    analysis_results: dict
) -> dict:
    """Generate final report dictionary.
    
    Args:
        note_name: Detected note name
        octave: Detected octave
        reed1_file: Reed 1 filename
        reed2_file: Reed 2 filename
        combined_file: Combined filename
        analysis_results: Analysis results from analyze_combined_recording
        
    Returns:
        Complete report dictionary
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "note": f"{note_name}{octave}",
        "note_name": note_name,
        "octave": octave,
        "sample_rate": SAMPLE_RATE,
        "files": {
            "reed1": reed1_file,
            "reed2": reed2_file,
            "combined": combined_file
        },
        "ground_truth": analysis_results["ground_truth"],
        "fft_detection": analysis_results["fft_detection"],
        "esprit_detection": analysis_results["esprit_detection"]
    }

    return report


def print_report_summary(report: dict) -> None:
    """Print a human-readable summary of the report."""
    print("\n" + "=" * 60)
    print("CLOSE REED ANALYSIS REPORT")
    print("=" * 60)

    print(f"\nNote: {report['note']}")
    print(f"Timestamp: {report['timestamp']}")

    gt = report["ground_truth"]
    print("\nGround Truth (from individual recordings):")
    print(f"  Reed 1: {gt['reed1_hz']:.3f} Hz")
    print(f"  Reed 2: {gt['reed2_hz']:.3f} Hz")
    print(f"  Separation: {gt['separation_hz']:.3f} Hz ({gt['separation_cents']:.2f} cents)")

    print("\nFFT Detection:")
    fft = report["fft_detection"]
    if "detected_frequencies" in fft:
        print(f"  Detected frequencies: {[f'{f:.3f}' for f in fft['detected_frequencies']]} Hz")
        print(f"  Reeds detected: {fft['reed_count_detected']}")
    else:
        print("  No frequencies detected")

    print("\nESPRIT Detection:")
    esprit = report["esprit_detection"]
    if "detected_frequencies" in esprit:
        print(f"  Detected frequencies: {[f'{f:.3f}' for f in esprit['detected_frequencies']]} Hz")
        print(f"  Reeds detected: {esprit['reed_count_detected']}")
    else:
        print("  No frequencies detected")

    # Calculate accuracy
    if "detected_frequencies" in esprit and len(esprit["detected_frequencies"]) >= 2:
        detected = sorted(esprit["detected_frequencies"])
        true_freqs = sorted([gt["reed1_hz"], gt["reed2_hz"]])

        # Calculate errors
        errors = []
        for i, true_freq in enumerate(true_freqs):
            if i < len(detected):
                error_hz = abs(detected[i] - true_freq)
                error_cents = 1200 * np.log2(detected[i] / true_freq) if true_freq > 0 else 0
                errors.append((error_hz, error_cents))

        print("\nDetection Accuracy (ESPRIT):")
        for i, (err_hz, err_cents) in enumerate(errors):
            print(f"  Reed {i+1}: {err_hz:.3f} Hz error ({err_cents:+.2f} cents)")

    print("\n" + "=" * 60)


def main():
    """Main recording and analysis workflow."""
    print("=" * 60)
    print("CLOSE REED RECORDING SCRIPT")
    print("=" * 60)
    print("\nThis script will record two close reeds (0.5-1 Hz apart)")
    print("and analyze detection accuracy.\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Record Reed 1
    print("\n" + "-" * 60)
    print("STEP 1: Record Reed 1 Alone")
    print("-" * 60)
    wait_for_enter("Press Enter when ready to record REED 1 ONLY...")
    countdown()
    reed1_audio = record_audio(5.0)
    reed1_file = f"reed1_temp_{timestamp}.npy"
    save_recording(reed1_audio, reed1_file)

    # Step 2: Record Reed 2
    print("\n" + "-" * 60)
    print("STEP 2: Record Reed 2 Alone")
    print("-" * 60)
    wait_for_enter("Press Enter when ready to record REED 2 ONLY...")
    countdown()
    reed2_audio = record_audio(5.0)
    reed2_file = f"reed2_temp_{timestamp}.npy"
    save_recording(reed2_audio, reed2_file)

    # Step 3: Record Both
    print("\n" + "-" * 60)
    print("STEP 3: Record Both Reeds Together")
    print("-" * 60)
    wait_for_enter("Press Enter when ready to record BOTH REEDS...")
    countdown()
    combined_audio = record_audio(5.0)
    combined_file = f"combined_temp_{timestamp}.npy"
    save_recording(combined_audio, combined_file)

    # Step 4: Analysis
    print("\n" + "-" * 60)
    print("STEP 4: Analyzing Recordings...")
    print("-" * 60)

    # Auto-detect note from combined recording
    print("\nAuto-detecting note...")
    note_name, octave = detect_note_from_audio(combined_audio)
    print(f"Detected: {note_name}{octave}")

    # Extract ground truth frequencies
    print("\nExtracting ground truth frequencies...")
    reed1_freq = extract_ground_truth_frequency(reed1_audio)
    reed2_freq = extract_ground_truth_frequency(reed2_audio)
    print(f"  Reed 1: {reed1_freq:.3f} Hz")
    print(f"  Reed 2: {reed2_freq:.3f} Hz")
    print(f"  Separation: {abs(reed2_freq - reed1_freq):.3f} Hz")

    # Analyze combined recording
    print("\nTesting detection algorithms on combined recording...")
    analysis_results = analyze_combined_recording(combined_audio, reed1_freq, reed2_freq)

    # Rename files with note
    note_str = f"{note_name}{octave}"
    reed1_final = f"reed1_{note_str}_{timestamp}.npy"
    reed2_final = f"reed2_{note_str}_{timestamp}.npy"
    combined_final = f"combined_{note_str}_{timestamp}.npy"
    report_file = f"report_{note_str}_{timestamp}.json"

    Path(reed1_file).rename(reed1_final)
    Path(reed2_file).rename(reed2_final)
    Path(combined_file).rename(combined_final)

    # Generate report
    report = generate_report(
        note_name, octave,
        reed1_final, reed2_final, combined_final,
        analysis_results
    )

    # Save report
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_file}")

    # Print summary
    print_report_summary(report)

    print("\nDone!")


if __name__ == "__main__":
    main()
