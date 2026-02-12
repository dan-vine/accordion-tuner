"""
FFT-ESPRIT based pitch detection for accordion tuning.

This module provides high-resolution frequency estimation using the FFT-ESPRIT
algorithm, which can resolve closely-spaced frequencies (like 1 Hz apart tremolo
reeds) that standard FFT cannot separate.

Based on:
    S. L. Kiser, et al., "Fast Kernel-based Signal Subspace Estimates for
    Line Spectral Estimation," PREPRINT, 2023.

Implementation inspired by:
    https://github.com/tam17aki/music-esprit-python
"""

import numpy as np
from scipy.linalg import pinv, qr
from scipy.signal import fftconvolve

from .constants import (
    A4_REFERENCE,
    A_OFFSET,
    C5_OFFSET,
    NOTE_NAMES,
    OCTAVE,
    SAMPLE_RATE,
)
from .multi_pitch_detector import Maximum, MultiPitchResult
from .temperaments import TEMPERAMENT_RATIOS, Temperament

# Detection constants
K_SAMPLES = 16384  # Analysis window size
K_STEP = 1024  # Hop size
K_MIN = 0.5  # Minimum magnitude threshold


def _safe_eigvals(matrix: np.ndarray) -> np.ndarray | None:
    """Safely compute eigenvalues of a matrix."""
    try:
        return np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None


class EspritPitchDetector:
    """
    High-resolution pitch detector using FFT-ESPRIT algorithm.

    FFT-ESPRIT provides super-resolution frequency estimation that can resolve
    frequencies closer than the FFT bin width. This is achieved by:
    1. Initial rough frequency estimation via iterative FFT
    2. Building a kernel from rough estimates
    3. Projecting data onto kernel via fast convolution
    4. QR decomposition for subspace estimation
    5. ESPRIT rotational invariance to extract frequencies

    This is particularly useful for accordion tremolo reeds which may be
    only 1-5 Hz apart.

    Merged Peak Detection
    ---------------------
    When two frequencies are very close (e.g., 440 Hz and 442 Hz), standard FFT
    shows them as a single merged peak rather than two distinct peaks. This
    detector identifies merged peaks by analyzing spectral width:

    - A pure sinusoid with Hamming window has predictable shoulder heights
      (~8% of peak at ±2 bins, ~2% at ±3 bins)
    - Merged frequencies produce wider peaks with higher shoulders
    - When width_threshold is exceeded, candidate frequencies are added around
      the peak to help ESPRIT resolve the close frequencies

    The candidates define a search neighborhood, but ESPRIT extracts actual
    frequencies from eigenvalues of the signal's subspace - so detected
    frequencies can be anywhere within ~5 Hz of candidates, not just at
    exact candidate positions.

    Key Parameters
    --------------
    width_threshold : float
        Shoulder-to-peak ratio threshold for detecting merged peaks.
        Default 0.25 means shoulders >25% of peak height trigger detection.
        Lower = more sensitive, higher = fewer false positives.

    candidate_offsets : list[float]
        Hz offsets added around merged peaks (e.g., [-0.8, -0.4, 0.4, 0.8]).
        These guide ESPRIT's search but don't constrain the output frequencies.

    min_separation : float
        Minimum Hz between detected frequencies. Frequencies closer than this
        are merged, keeping the one closest to a rough FFT estimate.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        fft_size: int = K_SAMPLES,
        hop_size: int = K_STEP,
        reference: float = A4_REFERENCE,
        num_sources: int = 4,
    ):
        """
        Initialize ESPRIT detector.

        Args:
            sample_rate: Audio sample rate in Hz
            fft_size: Analysis window size
            hop_size: Hop size between frames
            reference: Reference frequency for A4 in Hz
            num_sources: Expected number of signal sources (frequencies to detect)
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.reference = reference
        self.num_sources = num_sources
        self.temperament = Temperament.EQUAL
        self.key = 0

        # State buffer for accumulating samples
        self._buffer = np.zeros(fft_size, dtype=np.float64)
        self._dmax = 0.125  # Dynamic range normalization

        # Frequency range (accordion range: ~60 Hz to ~2000 Hz)
        self._freq_min = 60.0
        self._freq_max = 2000.0

        # Detection threshold
        self.min_magnitude = K_MIN

        # Window for FFT-based magnitude estimation
        self._window = np.hamming(fft_size)

        # Filters (for interface compatibility with MultiPitchDetector)
        self.fundamental_filter = False
        self.downsample = False
        self.octave_filter = True

        # Configurable parameters for close-frequency detection
        # Width threshold: ratio at ±2 bins to detect merged peaks (default 0.25)
        # Single sinusoid with Hamming drops to ~0.08 at ±2 bins
        # Higher values = more sensitive to merged peaks
        self._width_threshold = 0.25

        # Candidate offsets: Hz offsets to add around detected merged peaks
        # These help ESPRIT resolve close frequencies
        self._candidate_offsets = [-0.8, -0.4, 0.4, 0.8]

        # Minimum separation between detected frequencies (Hz)
        self._min_separation = 0.5

    def process(self, samples: np.ndarray) -> MultiPitchResult:
        """
        Process audio samples and detect multiple pitches using FFT-ESPRIT.

        Args:
            samples: Audio samples (will be added to internal buffer)

        Returns:
            MultiPitchResult with detected notes
        """
        # Ensure correct dtype
        if samples.dtype != np.float64:
            samples = samples.astype(np.float64)

        # Shift buffer and add new samples
        shift = min(len(samples), self.fft_size)
        self._buffer = np.roll(self._buffer, -shift)

        if len(samples) >= self.fft_size:
            self._buffer[:] = samples[-self.fft_size:]
        else:
            self._buffer[-len(samples):] = samples

        # Normalize
        dmax = np.max(np.abs(self._buffer))
        if dmax < 0.125:
            dmax = 0.125

        norm = self._dmax
        self._dmax = dmax

        input_signal = self._buffer / norm

        # Run FFT-ESPRIT algorithm to find frequencies
        detected_freqs = self._fft_esprit(input_signal)

        if not detected_freqs:
            return MultiPitchResult()

        # Estimate magnitudes using FFT at detected frequencies
        magnitudes = self._estimate_magnitudes(input_signal, detected_freqs)

        # Convert to Maximum objects with note information
        maxima = []
        for freq, mag in zip(detected_freqs, magnitudes):
            # Use lower threshold - AccordionDetector will filter by note proximity
            if mag < self.min_magnitude / 20:
                continue

            if freq <= 0:
                continue

            cf = -12.0 * np.log2(self.reference / freq)
            if np.isnan(cf):
                continue

            note = int(round(cf)) + C5_OFFSET
            if note < 0:
                continue

            # Fundamental filter: only allow harmonics of first detected note
            if self.fundamental_filter and len(maxima) > 0:
                if (note % OCTAVE) != (maxima[0].note % OCTAVE):
                    continue

            # Calculate reference frequency with temperament
            ref_freq = self._get_reference_frequency(note)

            # Calculate cents
            cents = 1200 * np.log2(freq / ref_freq) if ref_freq > 0 else 0.0

            maximum = Maximum(
                frequency=freq,
                ref_frequency=ref_freq,
                note=note,
                cents=cents,
                note_name=NOTE_NAMES[note % OCTAVE],
                octave=note // OCTAVE,
                magnitude=mag,
            )
            maxima.append(maximum)

        # Apply octave filter if enabled
        if self.octave_filter and len(maxima) > 1:
            fundamental_freq = maxima[0].frequency
            maxima = [m for m in maxima if m.frequency < fundamental_freq * 2.1]

        if not maxima:
            return MultiPitchResult()

        # Sort by frequency (lowest first)
        maxima.sort(key=lambda m: m.frequency)

        # Primary note is the first (lowest frequency) maximum
        primary = maxima[0]

        return MultiPitchResult(
            maxima=maxima,
            primary_frequency=primary.frequency,
            primary_note=primary.note,
            primary_cents=primary.cents,
            valid=True,
        )

    def _fft_esprit(self, signal: np.ndarray) -> list[float]:
        """
        FFT-ESPRIT algorithm for high-resolution frequency estimation.

        Args:
            signal: Normalized input signal

        Returns:
            List of detected frequencies in Hz
        """
        n_samples = len(signal)
        subspace_dim = n_samples // 3

        # Search for more frequencies than num_sources to account for
        # harmonics and spurious detections - let AccordionDetector filter
        search_count = max(self.num_sources * 2, self.num_sources + 4)

        # 1. Initial rough frequency estimation via iterative FFT
        rough_freqs = self._estimate_freqs_iterative(signal, search_count)
        if len(rough_freqs) == 0:
            return []

        # For real signals, include negative frequency pairs
        kernel_freqs = np.concatenate([rough_freqs, -rough_freqs])

        # 2. Build Vandermonde kernel matrix
        n_snapshots = n_samples - subspace_dim + 1
        t = np.arange(n_snapshots).reshape(-1, 1) / self.sample_rate
        freq_vec = kernel_freqs.reshape(1, -1)
        kernel_matrix = np.exp(2j * np.pi * t @ freq_vec)

        # 3. Fast Hankel-Vandermonde product via convolution
        n_components = kernel_matrix.shape[1]
        projected = np.zeros((subspace_dim, n_components), dtype=complex)
        for i in range(n_components):
            kernel_vec = kernel_matrix[:, i]
            conv_result = fftconvolve(signal, kernel_vec[::-1], mode="valid")
            projected[:, i] = conv_result[:subspace_dim]

        # 4. QR decomposition for orthonormalization
        try:
            q_matrix, _ = qr(projected, mode="economic")
        except np.linalg.LinAlgError:
            return list(rough_freqs[:self.num_sources])

        # 5. ESPRIT solver (Least Squares)
        q_upper = q_matrix[:-1, :]
        q_lower = q_matrix[1:, :]

        try:
            rotation_op = pinv(q_upper) @ q_lower
        except np.linalg.LinAlgError:
            return list(rough_freqs[:self.num_sources])

        eigenvalues = _safe_eigvals(rotation_op)
        if eigenvalues is None:
            return list(rough_freqs[:self.num_sources])

        # 6. Extract frequencies from eigenvalue angles
        omegas = np.angle(eigenvalues)
        freqs_hz = omegas * (self.sample_rate / (2 * np.pi))

        # Filter to positive frequencies in valid range and close to rough estimates
        # ESPRIT can produce spurious frequencies far from any actual signal
        valid_freqs = []
        for f in freqs_hz:
            if self._freq_min <= f <= self._freq_max:
                # Check if this frequency is reasonably close to any rough estimate
                # Allow up to 5 Hz deviation (ESPRIT refines the rough estimates)
                min_dist = min(abs(f - rf) for rf in rough_freqs)
                if min_dist <= 5.0:
                    valid_freqs.append(float(f))

        # Sort frequencies
        valid_freqs.sort()

        # Remove spurious frequencies that are too close together (< 0.5 Hz)
        # Keep the one closest to a rough_freqs estimate
        if len(valid_freqs) > 1:
            filtered = [valid_freqs[0]]
            for f in valid_freqs[1:]:
                if f - filtered[-1] >= 0.5:
                    filtered.append(f)
                else:
                    # Keep the one closer to a rough estimate
                    dist_new = min(abs(f - rf) for rf in rough_freqs)
                    dist_old = min(abs(filtered[-1] - rf) for rf in rough_freqs)
                    if dist_new < dist_old:
                        filtered[-1] = f
            valid_freqs = filtered

        return valid_freqs

    def _estimate_freqs_iterative(
        self, signal: np.ndarray, count: int | None = None
    ) -> np.ndarray:
        """
        Estimate frequencies by finding local maxima in the FFT spectrum.

        This provides rough initial estimates for FFT-ESPRIT. When close
        frequencies merge into a single wide peak, this method detects the
        merging and adds candidate frequencies to help ESPRIT resolve them.

        Merged Peak Detection Algorithm
        --------------------------------
        1. Find peaks in FFT spectrum using parabolic interpolation
        2. If only one dominant peak exists, check its spectral width:
           - Measure shoulder-to-peak ratio at ±2 and ±3 bins
           - Single Hamming-windowed sinusoid: ~0.08 at ±2, ~0.02 at ±3
           - Higher ratios indicate merged frequencies
        3. If width exceeds threshold, add candidates at configured offsets
           (e.g., peak ± 0.4 Hz, peak ± 0.8 Hz)
        4. These candidates guide ESPRIT but don't constrain its output -
           ESPRIT finds actual frequencies within ~5 Hz of any candidate

        Args:
            signal: Input signal
            count: Number of frequencies to search for (default: num_sources)

        Returns:
            Array of estimated frequencies in Hz, sorted ascending
        """
        if count is None:
            count = self.num_sources

        n_fft = len(signal)

        # Apply window for better spectral estimation
        window = np.hamming(n_fft)
        windowed = signal * window

        # FFT
        spectrum = np.abs(np.fft.fft(windowed, n=n_fft))
        freq_grid = np.fft.fftfreq(n_fft, d=1 / self.sample_rate)

        # Find indices in valid frequency range
        valid_mask = (freq_grid >= self._freq_min) & (freq_grid <= self._freq_max)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return np.array([])

        # Find the maximum magnitude for thresholding
        max_mag = np.max(spectrum[valid_indices])
        if max_mag < 1e-6:
            return np.array([])

        # Only consider peaks above 10% of the maximum (reject noise/leakage)
        mag_threshold = max_mag * 0.1
        freq_step = freq_grid[1] - freq_grid[0]

        # Find local maxima (peaks) in the valid range
        # A local maximum is higher than its neighbors
        peaks = []
        peak_indices = []  # Track indices for width analysis
        for i in valid_indices:
            if i < 3 or i >= len(spectrum) - 3:
                continue
            if spectrum[i] < mag_threshold:
                continue
            if spectrum[i] > spectrum[i - 1] and spectrum[i] > spectrum[i + 1]:
                # Parabolic interpolation for sub-bin accuracy
                y1, y2, y3 = spectrum[i - 1], spectrum[i], spectrum[i + 1]
                denom = y1 - 2 * y2 + y3
                if abs(denom) > 1e-10:
                    delta = 0.5 * (y1 - y3) / denom
                    peak_freq = freq_grid[i] + delta * freq_step
                else:
                    peak_freq = freq_grid[i]

                if peak_freq >= self._freq_min:
                    peaks.append((peak_freq, spectrum[i]))
                    peak_indices.append(i)

        if not peaks:
            return np.array([])

        # Check for merged close frequencies: if there's only ONE strong peak
        # but it's unusually wide, add candidate frequencies around it.
        # A single Hamming-windowed sinusoid drops to ~0.08 (-22dB) at ±2 bins.
        # Threshold is configurable via _width_threshold.
        if len(peaks) == 1 or (len(peaks) >= 1 and peaks[0][1] > peaks[1][1] * 3):
            # Only one dominant peak - check if it might be two merged frequencies
            i = peak_indices[0]
            peak_mag = spectrum[i]

            # Check spectral width: ratio of ±2 bin energy to peak
            width_ratio = max(spectrum[i - 2], spectrum[i + 2]) / peak_mag

            # Also check ±3 bins for very close frequencies
            width_ratio_3 = max(spectrum[i - 3], spectrum[i + 3]) / peak_mag

            # Single sinusoid with Hamming: ~0.08 at ±2 bins, ~0.02 at ±3 bins
            # Merged frequencies show higher ratios (configurable threshold)
            is_wide = width_ratio > self._width_threshold or width_ratio_3 > self._width_threshold * 0.32

            if is_wide and self._candidate_offsets:
                peak_freq = peaks[0][0]
                # Add candidates at configured offsets to help ESPRIT resolve close frequencies
                for offset in self._candidate_offsets:
                    candidate = peak_freq + offset
                    if candidate >= self._freq_min:
                        peaks.append((candidate, peak_mag * 0.8))

        # Sort by magnitude (strongest first)
        peaks.sort(key=lambda x: x[1], reverse=True)

        # Select top peaks with minimum separation (configurable)
        min_separation = self._min_separation
        freqs = []
        for freq, _mag in peaks:
            # Check if too close to any already-selected frequency
            too_close = False
            for selected_freq in freqs:
                if abs(freq - selected_freq) < min_separation:
                    too_close = True
                    break
            if not too_close:
                freqs.append(freq)
                if len(freqs) >= count:
                    break

        return np.sort(np.array(freqs))

    def _estimate_magnitudes(
        self, signal: np.ndarray, frequencies: list[float]
    ) -> list[float]:
        """Estimate magnitudes at detected frequencies using FFT."""
        windowed = signal * self._window
        spectrum = np.fft.rfft(windowed)
        fft_freqs = np.fft.rfftfreq(len(signal), 1.0 / self.sample_rate)

        magnitudes = []
        for freq in frequencies:
            idx = np.argmin(np.abs(fft_freqs - freq))
            if 0 < idx < len(spectrum) - 1:
                mag = np.abs(spectrum[idx - 1: idx + 2]).max()
            else:
                mag = np.abs(spectrum[idx])
            magnitudes.append(mag / 2048.0)

        return magnitudes

    def _get_reference_frequency(self, note: int) -> float:
        """Calculate temperament-adjusted reference frequency for a note."""
        note_in_octave = note % OCTAVE
        ratios = TEMPERAMENT_RATIOS[self.temperament]
        equal_ratios = TEMPERAMENT_RATIOS[Temperament.EQUAL]

        n = (note_in_octave - self.key + OCTAVE) % OCTAVE
        a = (A_OFFSET - self.key + OCTAVE) % OCTAVE

        temper_ratio = ratios[n] / ratios[a]
        equal_ratio = equal_ratios[n] / equal_ratios[a]
        temper_adjust = temper_ratio / equal_ratio

        semitones_from_a4 = note - C5_OFFSET
        equal_freq = self.reference * (2 ** (semitones_from_a4 / 12))

        return equal_freq * temper_adjust

    def set_reference(self, freq: float):
        """Set reference frequency for A4."""
        self.reference = freq

    def set_temperament(self, temperament: Temperament):
        """Set musical temperament."""
        self.temperament = temperament

    def set_key(self, key: int):
        """Set key for temperament (0=C, 1=C#, ..., 11=B)."""
        self.key = key % OCTAVE

    def set_num_sources(self, n: int):
        """Set expected number of signal sources."""
        self.num_sources = max(1, min(8, n))

    def set_fundamental_filter(self, enabled: bool):
        """Enable/disable fundamental filter."""
        self.fundamental_filter = enabled

    def set_downsample(self, enabled: bool):
        """Enable/disable downsampling."""
        self.downsample = enabled

    def set_octave_filter(self, enabled: bool):
        """Enable/disable octave filter."""
        self.octave_filter = enabled

    def set_min_magnitude(self, threshold: float):
        """Set minimum magnitude threshold."""
        self.min_magnitude = max(0.01, threshold)

    def set_width_threshold(self, threshold: float):
        """
        Set width threshold for detecting merged peaks.

        A single Hamming-windowed sinusoid has ~0.08 ratio at ±2 bins.
        Higher thresholds = less sensitive to merged peaks (fewer false positives).
        Lower thresholds = more sensitive (better close-freq detection but may hallucinate).

        Args:
            threshold: Ratio threshold (0.1 to 0.5, default 0.25)
        """
        self._width_threshold = max(0.1, min(0.5, threshold))

    def get_width_threshold(self) -> float:
        """Get current width threshold."""
        return self._width_threshold

    def set_candidate_offsets(self, offsets: list[float]):
        """
        Set frequency offsets for candidate generation around merged peaks.

        When a merged peak is detected, candidates are added at these offsets
        to help ESPRIT resolve the close frequencies.

        Args:
            offsets: List of Hz offsets (e.g., [-0.8, -0.4, 0.4, 0.8])
        """
        self._candidate_offsets = offsets

    def get_candidate_offsets(self) -> list[float]:
        """Get current candidate offsets."""
        return self._candidate_offsets

    def set_min_separation(self, separation: float):
        """
        Set minimum separation between detected frequencies.

        Args:
            separation: Minimum Hz between peaks (0.3 to 1.0, default 0.5)
        """
        self._min_separation = max(0.3, min(1.0, separation))

    def get_min_separation(self) -> float:
        """Get current minimum separation."""
        return self._min_separation

    def reset(self):
        """Reset internal state."""
        self._buffer.fill(0)
        self._dmax = 0.125
