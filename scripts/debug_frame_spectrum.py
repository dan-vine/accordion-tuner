"""
Debug script: Visualize what SimpleFFT sees at each frame for B4.

This shows the spectrum computed from the detector's buffer at each timestep,
overlaid with what SimpleFFT actually detects.
"""

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from accordion_tuner.simple_fft_detector import SimpleFftPeakDetector


def plot_frame_spectrum(audio, output_prefix, ground_truth_reed1, ground_truth_reed2):
    """Plot spectrum at each frame for SimpleFFT."""

    detector = SimpleFftPeakDetector()

    sample_rate = 11025
    hop_size = 1024
    fft_size = 16384

    start_idx = len(audio) - 22050  # final 2 seconds

    # Warmup: fill buffer before analysis period
    warmup_samples = audio[start_idx - fft_size:start_idx]
    for i in range(0, len(warmup_samples) - hop_size, hop_size):
        detector.process(warmup_samples[i:i + hop_size])

    # Compute full 5-second spectrum for comparison
    full_signal = audio.astype(np.float64)
    full_signal = full_signal / (np.max(np.abs(full_signal)) + 1e-6)
    full_window = np.hamming(len(full_signal))
    full_windowed = full_signal * full_window
    full_n_fft = len(full_signal) * 4
    full_spectrum = np.fft.rfft(full_windowed, n=full_n_fft)
    full_mags = np.abs(full_spectrum)
    full_freqs = np.fft.rfftfreq(full_n_fft, 1.0 / sample_rate)

    # Process each frame and save spectrum plots
    num_frames = 10

    for frame_num in range(num_frames):
        # Get current buffer content
        buffer = detector._buffer.copy()

        # Compute FFT (same as SimpleFFT does)
        signal = buffer * detector._window
        n_fft = fft_size * 4
        spectrum = np.fft.rfft(signal, n=n_fft)
        mags = np.abs(spectrum)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

        # Get detection result BEFORE processing next frame
        chunk = audio[start_idx + frame_num * hop_size:start_idx + (frame_num + 1) * hop_size]
        result = detector.process(chunk)

        detected_freqs = [r.frequency for r in result.maxima] if result.maxima else []

        # Create plot with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Calculate zoom range around ground truth
        freq_min = min(ground_truth_reed1, ground_truth_reed2) - 15
        freq_max = max(ground_truth_reed1, ground_truth_reed2) + 15

        # Top: Frame spectrum
        valid = (freqs >= freq_min) & (freqs <= freq_max)

        ax1.semilogy(freqs[valid], mags[valid], 'b-', linewidth=0.5, alpha=0.8)

        # Ground truth
        ax1.axvline(ground_truth_reed1, color='green', linestyle='--', linewidth=2, label=f'GT Reed1: {ground_truth_reed1:.2f} Hz')
        ax1.axvline(ground_truth_reed2, color='green', linestyle=':', linewidth=2, label=f'GT Reed2: {ground_truth_reed2:.2f} Hz')

        # Detected frequencies
        for _i, freq in enumerate(detected_freqs):
            if freq_min <= freq <= freq_max:
                ax1.axvline(freq, color='red', linestyle='-', linewidth=1.5, alpha=0.7)

        # Add detected frequencies to title
        detected_str = ', '.join([f'{f:.1f}' for f in detected_freqs[:5]])
        ax1.set_title(f'B4 Frame {frame_num} - Detected: [{detected_str}]')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Magnitude (log)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Bottom: Full 5-second spectrum
        full_valid = (full_freqs >= freq_min) & (full_freqs <= freq_max)
        ax2.semilogy(full_freqs[full_valid], full_mags[full_valid], 'purple', linewidth=0.5, alpha=0.8)
        ax2.axvline(ground_truth_reed1, color='green', linestyle='--', linewidth=2, label=f'GT Reed1: {ground_truth_reed1:.2f} Hz')
        ax2.axvline(ground_truth_reed2, color='green', linestyle=':', linewidth=2, label=f'GT Reed2: {ground_truth_reed2:.2f} Hz')
        ax2.set_title('Full 5-second recording spectrum')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude (log)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'{output_prefix}_frame_{frame_num:02d}.png'
        plt.savefig(filename, dpi=100)
        plt.close()
        print(f'Saved: {filename}')


if __name__ == '__main__':
    audio = np.load('scripts/combined_B4_20260213_161956.npy')
    plot_frame_spectrum(audio, 'scripts/debug_B4_spectrum', 499.4, 500.6)
    print('Done!')
