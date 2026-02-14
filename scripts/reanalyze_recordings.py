"""
Re-analyze close reed recordings with proper temporal analysis.

This script re-analyzes existing recordings frame-by-frame to properly
track reed detection across time and calculate accurate statistics.

Usage:
    python reanalyze_recordings.py

Analyzes final 2 seconds of each recording (~21 frames at 1024 samples/frame).
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from accordion_tuner.accordion import AccordionDetector, DetectorType
from accordion_tuner.constants import SAMPLE_RATE


def find_fundamental_frequency(
    freqs: np.ndarray,
    mags: np.ndarray,
    threshold_ratio: float = 0.1,
    min_freq: float = 60.0,
    max_freq: float = 2000.0
) -> float:
    """Find the fundamental frequency from spectrum, handling strong harmonics.
    
    Strategy:
    1. Find all significant peaks (above threshold)
    2. Look for harmonic series (integer multiples)
    3. Return the lowest frequency with strong harmonics (fundamental)
    4. Fall back to strongest peak if no clear fundamental
    
    Args:
        freqs: Frequency bins
        mags: Magnitudes
        threshold_ratio: Minimum peak magnitude as ratio of max
        min_freq: Minimum frequency to consider
        max_freq: Maximum frequency to consider
        
    Returns:
        Fundamental frequency in Hz
    """
    # Find all peaks
    threshold = np.max(mags) * threshold_ratio
    peaks = []
    
    for i in range(1, len(mags) - 1):
        if mags[i] > threshold and mags[i] > mags[i-1] and mags[i] > mags[i+1]:
            # Parabolic interpolation for sub-bin accuracy
            y1, y2, y3 = mags[i-1], mags[i], mags[i+1]
            denom = y1 - 2*y2 + y3
            if abs(denom) > 1e-10:
                delta = 0.5 * (y1 - y3) / denom
                freq_step = freqs[1] - freqs[0]
                freq = freqs[i] + delta * freq_step
                mag = y2 - 0.25 * (y1 - y3) * delta
            else:
                freq = freqs[i]
                mag = y2
            
            if min_freq <= freq <= max_freq:
                peaks.append((freq, mag))
    
    if not peaks:
        return 0.0
    
    # Sort by frequency (lowest first)
    peaks.sort(key=lambda x: x[0])
    
    # Look for harmonic series
    # The fundamental should have strong peaks at integer multiples
    best_fundamental = peaks[0][0]  # Default to lowest peak
    best_score = 0.0
    
    for candidate_idx, (candidate_freq, candidate_mag) in enumerate(peaks[:5]):  # Check top 5 lowest peaks
        score = 1.0  # Base score for the candidate itself
        
        # Check for harmonics (2x, 3x, 4x, 5x)
        for harmonic_num in range(2, 6):
            expected_harmonic = candidate_freq * harmonic_num
            # Look for a peak near this harmonic (±2% tolerance)
            tolerance = expected_harmonic * 0.02
            
            for peak_freq, peak_mag in peaks:
                if abs(peak_freq - expected_harmonic) < tolerance:
                    # Found a harmonic - weight by magnitude ratio
                    harmonic_strength = peak_mag / candidate_mag
                    score += harmonic_strength
                    break
        
        # Prefer lower frequencies slightly (within 10% of max score)
        if score > best_score * 0.9 and candidate_freq < best_fundamental * 1.5:
            if score > best_score:
                best_score = score
                best_fundamental = candidate_freq
    
    return float(best_fundamental)


def extract_ground_truth_temporal(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> dict:
    """Extract ground truth frequencies from final 2 seconds using frame-based analysis.
    
    Uses harmonic detection to find the fundamental frequency rather than
    just the strongest spectral peak (which may be a harmonic).
    
    Args:
        audio: Audio recording
        sample_rate: Sample rate
        
    Returns:
        Dictionary with frame-by-frame frequencies and statistics
    """
    hop_size = 1024
    frames_per_2sec = int(2.0 * sample_rate / hop_size)  # ~21 frames
    
    # Use final 2 seconds
    start_idx = max(0, len(audio) - frames_per_2sec * hop_size)
    audio_segment = audio[start_idx:]
    
    frequencies = []
    
    for i in range(0, len(audio_segment) - hop_size, hop_size):
        chunk = audio_segment[i:i + hop_size]
        
        # High-resolution FFT
        signal = chunk.astype(np.float64)
        signal_max = np.max(np.abs(signal))
        if signal_max < 1e-6:
            continue
        signal = signal / signal_max
        
        window = np.hamming(len(signal))
        windowed = signal * window
        n_fft = len(signal) * 4
        
        spectrum = np.fft.rfft(windowed, n=n_fft)
        mags = np.abs(spectrum)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        
        # Find fundamental using harmonic detection
        freq = find_fundamental_frequency(freqs, mags)
        
        if freq > 0:
            frequencies.append(freq)
    
    if not frequencies:
        return {"frequencies": [], "mean": 0.0, "std": 0.0, "median": 0.0, "count": 0}
    
    return {
        "frequencies": frequencies,
        "mean": float(np.mean(frequencies)),
        "std": float(np.std(frequencies)),
        "median": float(np.median(frequencies)),
        "count": len(frequencies)
    }


TOLERANCE_HZ = 0.25
TOLERANCE_HZ_TIGHT = 0.1


def analyze_detector_temporal(
    audio: np.ndarray,
    detector_type: DetectorType,
    ground_truth_reed1: float,
    ground_truth_reed2: float,
    sample_rate: int = SAMPLE_RATE
) -> dict:
    """Analyze detector performance frame-by-frame on final 2 seconds.
    
    Args:
        audio: Combined recording
        detector_type: FFT or ESPRIT
        ground_truth_reed1: Expected frequency of reed 1 (lower)
        ground_truth_reed2: Expected frequency of reed 2 (higher)
        sample_rate: Sample rate
        
    Returns:
        Dictionary with frame-by-frame results and statistics
    """
    hop_size = 1024
    frames_per_2sec = int(2.0 * sample_rate / hop_size)
    fft_size = 16384
    
    # Use final 2 seconds
    start_idx = max(0, len(audio) - frames_per_2sec * hop_size)
    audio_segment = audio[start_idx:]
    
    detector = AccordionDetector(detector_type=detector_type)
    if detector_type == DetectorType.ESPRIT:
        detector.set_esprit_min_separation(0.1)
    
    # Warmup: fill the detector buffer before collecting results
    warmup_samples = audio[max(0, start_idx - fft_size):start_idx]
    if len(warmup_samples) > 0:
        for i in range(0, len(warmup_samples) - hop_size, hop_size):
            detector.process(warmup_samples[i:i + hop_size])
    
    frame_results = []
    reed1_detections = []
    reed2_detections = []
    fundamental_detections = []
    
    for i in range(0, len(audio_segment) - hop_size, hop_size):
        chunk = audio_segment[i:i + hop_size]
        result = detector.process(chunk)
        
        frame_data = {
            "frame": i // hop_size,
            "valid": result.valid,
            "reed_count": result.reed_count,
            "detected_frequencies": [],
            "errors_hz": [],
            "errors_cents": [],
            "fundamental_frequency": None,
            "fundamental_error_hz": None
        }
        
        if result.valid and result.reeds:
            # Sort detected frequencies
            freqs = sorted([r.frequency for r in result.reeds])
            frame_data["detected_frequencies"] = freqs
            
            # Track fundamental (lowest detected frequency)
            if freqs:
                frame_data["fundamental_frequency"] = freqs[0]
                frame_data["fundamental_error_hz"] = abs(freqs[0] - ground_truth_reed1)
                fundamental_detections.append(freqs[0])
            
            # Assign to reed 1 and reed 2 (lowest and highest)
            if len(freqs) >= 1:
                # Determine which ground truth is closer
                err1 = abs(freqs[0] - ground_truth_reed1)
                err2 = abs(freqs[0] - ground_truth_reed2)
                
                if err1 <= err2:
                    # First freq is reed 1
                    frame_data["errors_hz"].append(err1)
                    frame_data["errors_cents"].append(
                        1200 * np.log2(freqs[0] / ground_truth_reed1) if ground_truth_reed1 > 0 else 0
                    )
                    reed1_detections.append(freqs[0])
                    
                    if len(freqs) >= 2:
                        frame_data["errors_hz"].append(abs(freqs[-1] - ground_truth_reed2))
                        frame_data["errors_cents"].append(
                            1200 * np.log2(freqs[-1] / ground_truth_reed2) if ground_truth_reed2 > 0 else 0
                        )
                        reed2_detections.append(freqs[-1])
                else:
                    # First freq is reed 2 (unexpected, but handle it)
                    frame_data["errors_hz"].append(err2)
                    frame_data["errors_cents"].append(
                        1200 * np.log2(freqs[0] / ground_truth_reed2) if ground_truth_reed2 > 0 else 0
                    )
                    reed2_detections.append(freqs[0])
                    
                    if len(freqs) >= 2:
                        frame_data["errors_hz"].append(abs(freqs[-1] - ground_truth_reed1))
                        frame_data["errors_cents"].append(
                            1200 * np.log2(freqs[-1] / ground_truth_reed1) if ground_truth_reed1 > 0 else 0
                        )
                        reed1_detections.append(freqs[-1])
        
        frame_results.append(frame_data)
    
    # Calculate statistics
    stats = {
        "total_frames": len(frame_results),
        "valid_frames": sum(1 for f in frame_results if f["valid"]),
        "frames_with_reed1": len(reed1_detections),
        "frames_with_reed2": len(reed2_detections),
        "frames_with_both": sum(1 for f in frame_results if f["reed_count"] >= 2),
        "detection_rate_reed1": len(reed1_detections) / len(frame_results) if frame_results else 0,
        "detection_rate_reed2": len(reed2_detections) / len(frame_results) if frame_results else 0,
    }
    
    if reed1_detections:
        reed1_errors = [abs(f - ground_truth_reed1) for f in reed1_detections]
        reed1_correct_025 = sum(1 for e in reed1_errors if e <= TOLERANCE_HZ)
        reed1_correct_010 = sum(1 for e in reed1_errors if e <= TOLERANCE_HZ_TIGHT)
        stats["reed1"] = {
            "mean_hz": float(np.mean(reed1_detections)),
            "std_hz": float(np.std(reed1_detections)),
            "median_hz": float(np.median(reed1_detections)),
            "mean_error_hz": float(np.mean(reed1_errors)),
            "mean_error_cents": float(np.mean([
                abs(1200 * np.log2(f / ground_truth_reed1)) for f in reed1_detections
            ])) if ground_truth_reed1 > 0 else 0,
            "correct_within_0.25hz": reed1_correct_025,
            "correct_within_0.1hz": reed1_correct_010,
            "correct_rate_0.25": reed1_correct_025 / len(reed1_detections),
            "correct_rate_0.1": reed1_correct_010 / len(reed1_detections),
            "count": len(reed1_detections)
        }
    
    if reed2_detections:
        reed2_errors = [abs(f - ground_truth_reed2) for f in reed2_detections]
        reed2_correct_025 = sum(1 for e in reed2_errors if e <= TOLERANCE_HZ)
        reed2_correct_010 = sum(1 for e in reed2_errors if e <= TOLERANCE_HZ_TIGHT)
        stats["reed2"] = {
            "mean_hz": float(np.mean(reed2_detections)),
            "std_hz": float(np.std(reed2_detections)),
            "median_hz": float(np.median(reed2_detections)),
            "mean_error_hz": float(np.mean(reed2_errors)),
            "mean_error_cents": float(np.mean([
                abs(1200 * np.log2(f / ground_truth_reed2)) for f in reed2_detections
            ])) if ground_truth_reed2 > 0 else 0,
            "correct_within_0.25hz": reed2_correct_025,
            "correct_within_0.1hz": reed2_correct_010,
            "correct_rate_0.25": reed2_correct_025 / len(reed2_detections),
            "correct_rate_0.1": reed2_correct_010 / len(reed2_detections),
            "count": len(reed2_detections)
        }
    
    # Fundamental statistics (lowest detected frequency vs lower reed)
    if fundamental_detections:
        fundamental_errors = [abs(f - ground_truth_reed1) for f in fundamental_detections]
        fundamental_correct_025 = sum(1 for e in fundamental_errors if e <= TOLERANCE_HZ)
        fundamental_correct_010 = sum(1 for e in fundamental_errors if e <= TOLERANCE_HZ_TIGHT)
        stats["fundamental"] = {
            "mean_hz": float(np.mean(fundamental_detections)),
            "std_hz": float(np.std(fundamental_detections)),
            "median_hz": float(np.median(fundamental_detections)),
            "mean_error_hz": float(np.mean(fundamental_errors)),
            "correct_within_0.25hz": fundamental_correct_025,
            "correct_within_0.1hz": fundamental_correct_010,
            "correct_rate_0.25": fundamental_correct_025 / len(fundamental_detections),
            "correct_rate_0.1": fundamental_correct_010 / len(fundamental_detections),
            "count": len(fundamental_detections)
        }

    frames_both_correct_025 = 0
    frames_both_correct_010 = 0
    for f in frame_results:
        if f["reed_count"] >= 2 and len(f["errors_hz"]) >= 2:
            if f["errors_hz"][0] <= TOLERANCE_HZ and f["errors_hz"][1] <= TOLERANCE_HZ:
                frames_both_correct_025 += 1
            if f["errors_hz"][0] <= TOLERANCE_HZ_TIGHT and f["errors_hz"][1] <= TOLERANCE_HZ_TIGHT:
                frames_both_correct_010 += 1
    stats["frames_both_correct_0.25"] = frames_both_correct_025
    stats["frames_both_correct_0.1"] = frames_both_correct_010
    
    return {
        "frame_results": frame_results,
        "statistics": stats
    }


def reanalyze_note(note_name: str, timestamp: str) -> dict:
    """Re-analyze all recordings for a single note.
    
    Args:
        note_name: Note name (e.g., "C4")
        timestamp: Recording timestamp
        
    Returns:
        Complete analysis dictionary
    """
    scripts_dir = Path("scripts")
    
    # Load files
    reed1_file = scripts_dir / f"reed1_{note_name}_{timestamp}.npy"
    reed2_file = scripts_dir / f"reed2_{note_name}_{timestamp}.npy"
    combined_file = scripts_dir / f"combined_{note_name}_{timestamp}.npy"
    
    if not all(f.exists() for f in [reed1_file, reed2_file, combined_file]):
        print(f"Warning: Missing files for {note_name}")
        return {}
    
    reed1_audio = np.load(reed1_file)
    reed2_audio = np.load(reed2_file)
    combined_audio = np.load(combined_file)
    
    print(f"\nAnalyzing {note_name}...")
    
    # Re-extract ground truth
    print("  Extracting ground truth...")
    gt_reed1 = extract_ground_truth_temporal(reed1_audio)
    gt_reed2 = extract_ground_truth_temporal(reed2_audio)
    
    print(f"    Reed 1: {gt_reed1['mean']:.3f} ± {gt_reed1['std']:.3f} Hz")
    print(f"    Reed 2: {gt_reed2['mean']:.3f} ± {gt_reed2['std']:.3f} Hz")
    
    # Analyze detectors
    print("  Analyzing FFT detector...")
    fft_results = analyze_detector_temporal(
        combined_audio,
        DetectorType.FFT,
        gt_reed1["mean"],
        gt_reed2["mean"]
    )
    
    print("  Analyzing ESPRIT detector...")
    esprit_results = analyze_detector_temporal(
        combined_audio,
        DetectorType.ESPRIT,
        gt_reed1["mean"],
        gt_reed2["mean"]
    )
    
    print("  Analyzing SIMPLE_FFT detector...")
    simple_fft_results = analyze_detector_temporal(
        combined_audio,
        DetectorType.SIMPLE_FFT,
        gt_reed1["mean"],
        gt_reed2["mean"]
    )
    
    # Generate plots
    print("  Generating plots...")
    plot_reed_analysis(
        note_name, 1, combined_audio,
        fft_results, esprit_results, simple_fft_results,
        gt_reed1["mean"]
    )
    plot_reed_analysis(
        note_name, 2, combined_audio,
        fft_results, esprit_results, simple_fft_results,
        gt_reed2["mean"]
    )
    
    # Build report
    report = {
        "note": note_name,
        "timestamp": timestamp,
        "sample_rate": SAMPLE_RATE,
        "analysis_duration_sec": 2.0,
        "files": {
            "reed1": str(reed1_file),
            "reed2": str(reed2_file),
            "combined": str(combined_file)
        },
        "ground_truth": {
            "reed1": gt_reed1,
            "reed2": gt_reed2,
            "separation_hz": abs(gt_reed2["mean"] - gt_reed1["mean"]),
            "separation_cents": 1200 * np.log2(gt_reed2["mean"] / gt_reed1["mean"]) if gt_reed1["mean"] > 0 else 0
        },
        "fft_detection": fft_results,
        "esprit_detection": esprit_results,
        "simple_fft_detection": simple_fft_results
    }
    
    return report


def print_summary(all_reports: list[dict]) -> None:
    """Print a summary table of all analyses."""
    print("\n" + "=" * 220)
    print("CLOSE REED TEMPORAL ANALYSIS SUMMARY")
    print(f"{'Note':<6} {'Sep(Hz)':<7} {'Sep(¢)':<6} {'Det':<4} {'Fund@0.25':<10} {'R1@0.25':<10} {'R2@0.25':<10} {'Both@0.25':<10} {'Fund@0.1':<10} {'R1@0.1':<10} {'R2@0.1':<10} {'Both@0.1':<10} {'R1 Err':<9} {'R2 Err':<9}")
    print("-" * 220)
    
    for report in all_reports:
        if not report:
            continue
        
        note = report["note"]
        sep_hz = report["ground_truth"]["separation_hz"]
        sep_cents = report["ground_truth"]["separation_cents"]
        
        # FFT stats
        fft_stats = report["fft_detection"]["statistics"]
        
        if "fundamental" in fft_stats:
            fft_fund_025 = f"{fft_stats['fundamental']['correct_within_0.25hz']}/{fft_stats['fundamental']['count']}"
            fft_fund_010 = f"{fft_stats['fundamental']['correct_within_0.1hz']}/{fft_stats['fundamental']['count']}"
        else:
            fft_fund_025 = fft_fund_010 = "N/A"
        
        if "reed1" in fft_stats:
            fft_r1_025 = f"{fft_stats['reed1']['correct_within_0.25hz']}/{fft_stats['reed1']['count']}"
            fft_r2_025 = f"{fft_stats['reed2']['correct_within_0.25hz']}/{fft_stats['reed2']['count']}" if "reed2" in fft_stats else "N/A"
            fft_r1_010 = f"{fft_stats['reed1']['correct_within_0.1hz']}/{fft_stats['reed1']['count']}"
            fft_r2_010 = f"{fft_stats['reed2']['correct_within_0.1hz']}/{fft_stats['reed2']['count']}" if "reed2" in fft_stats else "N/A"
            fft_both_025 = f"{fft_stats.get('frames_both_correct_0.25', 0)}"
            fft_both_010 = f"{fft_stats.get('frames_both_correct_0.1', 0)}"
            fft_r1_err = f"{fft_stats['reed1'].get('mean_error_hz', 0):.2f}"
            fft_r2_err = f"{fft_stats['reed2'].get('mean_error_hz', 0):.2f}" if "reed2" in fft_stats else "N/A"
        else:
            fft_r1_025 = fft_r2_025 = fft_r1_010 = fft_r2_010 = "N/A"
            fft_both_025 = fft_both_010 = "0"
            fft_r1_err = fft_r2_err = "N/A"
        
        print(f"{note:<6} {sep_hz:<7.3f} {sep_cents:<6.1f} {'FFT':<4} {fft_fund_025:<10} {fft_r1_025:<10} {fft_r2_025:<10} {fft_both_025:<10} {fft_fund_010:<10} {fft_r1_010:<10} {fft_r2_010:<10} {fft_both_010:<10} {fft_r1_err:<9} {fft_r2_err:<9}")
        
        # ESPRIT stats
        esprit_stats = report["esprit_detection"]["statistics"]
        
        if "fundamental" in esprit_stats:
            esp_fund_025 = f"{esprit_stats['fundamental']['correct_within_0.25hz']}/{esprit_stats['fundamental']['count']}"
            esp_fund_010 = f"{esprit_stats['fundamental']['correct_within_0.1hz']}/{esprit_stats['fundamental']['count']}"
        else:
            esp_fund_025 = esp_fund_010 = "N/A"
        
        if "reed1" in esprit_stats:
            esp_r1_025 = f"{esprit_stats['reed1']['correct_within_0.25hz']}/{esprit_stats['reed1']['count']}"
            esp_r2_025 = f"{esprit_stats['reed2']['correct_within_0.25hz']}/{esprit_stats['reed2']['count']}" if "reed2" in esprit_stats else "N/A"
            esp_r1_010 = f"{esprit_stats['reed1']['correct_within_0.1hz']}/{esprit_stats['reed1']['count']}"
            esp_r2_010 = f"{esprit_stats['reed2']['correct_within_0.1hz']}/{esprit_stats['reed2']['count']}" if "reed2" in esprit_stats else "N/A"
            esp_both_025 = f"{esprit_stats.get('frames_both_correct_0.25', 0)}"
            esp_both_010 = f"{esprit_stats.get('frames_both_correct_0.1', 0)}"
            esp_r1_err = f"{esprit_stats['reed1'].get('mean_error_hz', 0):.2f}"
            esp_r2_err = f"{esprit_stats['reed2'].get('mean_error_hz', 0):.2f}" if "reed2" in esprit_stats else "N/A"
        else:
            esp_r1_025 = esp_r2_025 = esp_r1_010 = esp_r2_010 = "N/A"
            esp_both_025 = esp_both_010 = "0"
            esp_r1_err = esp_r2_err = "N/A"
        
        print(f"{'':6} {'':7} {'':6} {'ESP':<4} {esp_fund_025:<10} {esp_r1_025:<10} {esp_r2_025:<10} {esp_both_025:<10} {esp_fund_010:<10} {esp_r1_010:<10} {esp_r2_010:<10} {esp_both_010:<10} {esp_r1_err:<9} {esp_r2_err:<9}")
        
        # SIMPLE_FFT stats
        simple_fft_stats = report.get("simple_fft_detection", {}).get("statistics", {})
        
        if "fundamental" in simple_fft_stats:
            sff_fund_025 = f"{simple_fft_stats['fundamental']['correct_within_0.25hz']}/{simple_fft_stats['fundamental']['count']}"
            sff_fund_010 = f"{simple_fft_stats['fundamental']['correct_within_0.1hz']}/{simple_fft_stats['fundamental']['count']}"
        else:
            sff_fund_025 = sff_fund_010 = "N/A"
        
        if "reed1" in simple_fft_stats:
            sff_r1_025 = f"{simple_fft_stats['reed1']['correct_within_0.25hz']}/{simple_fft_stats['reed1']['count']}"
            sff_r2_025 = f"{simple_fft_stats['reed2']['correct_within_0.25hz']}/{simple_fft_stats['reed2']['count']}" if "reed2" in simple_fft_stats else "N/A"
            sff_r1_010 = f"{simple_fft_stats['reed1']['correct_within_0.1hz']}/{simple_fft_stats['reed1']['count']}"
            sff_r2_010 = f"{simple_fft_stats['reed2']['correct_within_0.1hz']}/{simple_fft_stats['reed2']['count']}" if "reed2" in simple_fft_stats else "N/A"
            sff_both_025 = f"{simple_fft_stats.get('frames_both_correct_0.25', 0)}"
            sff_both_010 = f"{simple_fft_stats.get('frames_both_correct_0.1', 0)}"
            sff_r1_err = f"{simple_fft_stats['reed1'].get('mean_error_hz', 0):.2f}"
            sff_r2_err = f"{simple_fft_stats['reed2'].get('mean_error_hz', 0):.2f}" if "reed2" in simple_fft_stats else "N/A"
        else:
            sff_r1_025 = sff_r2_025 = sff_r1_010 = sff_r2_010 = "N/A"
            sff_both_025 = sff_both_010 = "0"
            sff_r1_err = sff_r2_err = "N/A"
        
        print(f"{'':6} {'':7} {'':6} {'SFF':<4} {sff_fund_025:<10} {sff_r1_025:<10} {sff_r2_025:<10} {sff_both_025:<10} {sff_fund_010:<10} {sff_r1_010:<10} {sff_r2_010:<10} {sff_both_010:<10} {sff_r1_err:<9} {sff_r2_err:<9}")
        print()
    
    print("=" * 220)


def plot_reed_analysis(
    note_name: str,
    reed_num: int,
    audio: np.ndarray,
    fft_results: dict,
    esprit_results: dict,
    simple_fft_results: dict,
    ground_truth_freq: float,
    sample_rate: int = SAMPLE_RATE
) -> None:
    """Generate histogram + spectrum plot for a single reed.
    
    Args:
        note_name: Note name (e.g., "A4")
        reed_num: Reed number (1 or 2)
        audio: Combined audio recording
        fft_results: FFT detector frame results
        esprit_results: ESPRIT detector frame results
        simple_fft_results: Simple FFT detector frame results
        ground_truth_freq: Ground truth frequency for this reed
        sample_rate: Audio sample rate
    """
    fft_freqs = []
    esprit_freqs = []
    simple_fft_freqs = []
    
    for frame in fft_results.get("frame_results", []):
        if frame["valid"] and frame["detected_frequencies"]:
            fft_freqs.extend(frame["detected_frequencies"])
    
    for frame in esprit_results.get("frame_results", []):
        if frame["valid"] and frame["detected_frequencies"]:
            esprit_freqs.extend(frame["detected_frequencies"])
    
    for frame in simple_fft_results.get("frame_results", []):
        if frame["valid"] and frame["detected_frequencies"]:
            simple_fft_freqs.extend(frame["detected_frequencies"])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    all_freqs = fft_freqs + esprit_freqs + simple_fft_freqs
    if all_freqs:
        freq_min = min(all_freqs) - 5
        freq_max = max(all_freqs) + 5
    else:
        freq_min = ground_truth_freq - 20
        freq_max = ground_truth_freq + 20
    
    bins = np.linspace(freq_min, freq_max, 50)
    
    if fft_freqs:
        ax1.hist(fft_freqs, bins=bins, alpha=0.5, label=f'FFT (n={len(fft_freqs)})', color='blue', edgecolor='black')
    if esprit_freqs:
        ax1.hist(esprit_freqs, bins=bins, alpha=0.5, label=f'ESPRIT (n={len(esprit_freqs)})', color='orange', edgecolor='black')
    if simple_fft_freqs:
        ax1.hist(simple_fft_freqs, bins=bins, alpha=0.5, label=f'SimpleFFT (n={len(simple_fft_freqs)})', color='red', edgecolor='black')
    
    ax1.axvline(ground_truth_freq, color='green', linestyle='--', linewidth=2, label=f'Ground Truth: {ground_truth_freq:.2f} Hz')
    ax1.axvline(ground_truth_freq + 0.25, color='green', linestyle=':', linewidth=1, alpha=0.7)
    ax1.axvline(ground_truth_freq - 0.25, color='green', linestyle=':', linewidth=1, alpha=0.7, label='±0.25 Hz')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{note_name} Reed {reed_num}: Detected Frequencies Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    signal = audio.astype(np.float64)
    signal_max = np.max(np.abs(signal))
    if signal_max > 1e-6:
        signal = signal / signal_max
    
    window = np.hamming(len(signal))
    windowed = signal * window
    n_fft = len(signal) * 4
    
    spectrum = np.fft.rfft(windowed, n=n_fft)
    mags = np.abs(spectrum)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    
    valid_mask = (freqs >= 50) & (freqs <= 2000)
    freqs_plot = freqs[valid_mask]
    mags_plot = mags[valid_mask]
    
    ax2.semilogy(freqs_plot, mags_plot, 'b-', linewidth=0.5)
    ax2.axvline(ground_truth_freq, color='green', linestyle='--', linewidth=2, label=f'Ground Truth: {ground_truth_freq:.2f} Hz')
    ax2.axvline(ground_truth_freq + 0.25, color='green', linestyle=':', linewidth=1, alpha=0.7)
    ax2.axvline(ground_truth_freq - 0.25, color='green', linestyle=':', linewidth=1, alpha=0.7, label='±0.25 Hz')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude (log)')
    ax2.set_title(f'{note_name} Reed {reed_num}: Spectrum')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(freq_min, freq_max)
    
    plt.tight_layout()
    
    filename = f"scripts/plot_{note_name}_reed{reed_num}.png"
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"  Saved: {filename}")


def main():
    """Main re-analysis workflow."""
    print("=" * 100)
    print("RE-ANALYZING CLOSE REED RECORDINGS")
    print("=" * 100)
    print("\nAnalyzing final 2 seconds of each recording with frame-by-frame detection")
    
    # Find all recording triplets
    scripts_dir = Path("scripts")
    
    # Look for combined files to get note/timestamp pairs
    combined_files = sorted(scripts_dir.glob("combined_*.npy"))
    
    all_reports = []
    
    for combined_file in combined_files:
        # Parse filename: combined_<note>_<timestamp>.npy
        parts = combined_file.stem.split("_")
        if len(parts) >= 3:
            note_name = parts[1]
            timestamp = "_".join(parts[2:])
            
            report = reanalyze_note(note_name, timestamp)
            if report:
                all_reports.append(report)
                
                # Save individual report
                report_file = scripts_dir / f"report_{note_name}_{timestamp}_temporal.json"
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"  Saved: {report_file}")
    
    # Print summary
    print_summary(all_reports)
    
    # Save aggregate report
    if all_reports:
        aggregate = {
            "timestamp": datetime.now().isoformat(),
            "notes_analyzed": len(all_reports),
            "notes": [r["note"] for r in all_reports if r],
            "summary": all_reports
        }
        
        aggregate_file = scripts_dir / f"report_aggregate_temporal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(aggregate_file, 'w') as f:
            json.dump(aggregate, f, indent=2)
        print(f"\nAggregate report saved: {aggregate_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
