# Accordion Tuner

A musical instrument tuner specialized for accordion reed tuning with multi-pitch detection, temperament support, and beat frequency measurement.

[![Latest Release](https://img.shields.io/github/v/release/dan-vine/accordion-tuner)](https://github.com/dan-vine/accordion-tuner/releases/latest)

![Accordion Tuner Screenshot](images/screenshot.png)

## Download

**[Download the latest release](https://github.com/dan-vine/accordion-tuner/releases/latest)**

- **Windows**: Download `accordion-tuner-windows.zip`, extract, and run `accordion-tuner.exe`
- **macOS**: Download `Accordion-Tuner.dmg`, drag to Applications (see [macOS installation notes](https://github.com/dan-vine/accordion-tuner/releases/latest))

## Features

- Multi-reed detection (1-4 simultaneous reeds) for accordion tremolo/musette tuning
- Beat frequency calculation for tremolo/musette tuning
- Custom tremolo tuning profiles (CSV/TSV)
- Multiple detection algorithms optimized for accordion reeds: FFT, SimpleFFT, and ESPRIT
- Hold mode for capturing best measurement
- Measurement log for recording and exporting tuning data

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Run the GUI
accordion-tuner

# Run CLI tuner
accordion-tuner-cli
```

## Settings

Click the "Settings" button to expand the settings panel with four tabs.

### Main Settings Bar

- **Reeds** - Number of reeds to detect (1-4). Set this to match your accordion's reed configuration for the register you're tuning.
- **Reference A4** - Reference frequency for A4 in Hz (default 440 Hz). Adjust if your accordion is tuned to a different concert pitch.

### Detection Tab

**Algorithm:**
- **FFT (Phase Vocoder)** - Default algorithm. Fast and reliable for typical accordion tuning. Uses FFT with phase vocoder for sub-bin frequency accuracy.
- **SimpleFFT** - Best for very close tremolo reeds (<1 Hz apart). Uses zero-padding for higher frequency resolution and bidirectional search to find reeds on both sides of the primary peak. Requires several seconds to stabilize for best accuracy.
- **ESPRIT** - Another method optimized for closely-spaced frequencies. Uses subspace methods to achieve super-resolution frequency estimation beyond the FFT bin width.

**General Settings:**
- **Octave Filter** - When enabled (default), restricts detection to one octave above the fundamental, filtering out harmonics. This does not affect detection of closely-spaced tremolo reeds (which differ by only a few Hz). Turn OFF only when detecting octave pairs (e.g., A3+A4 playing together).
- **Fundamental Filter** - Only detect harmonics of the fundamental frequency. Only accepts same-named notes (A, A, A...) 
- **Sensitivity** - Detection threshold (0.01-0.50). Lower values increase sensitivity but may detect more background noise.
- **Reed Spread** - Maximum cents deviation to group as the same note (20-100¢). Increase if your tremolo reeds have wide detuning.
- **Peak Threshold** - Relative threshold (5-50%). Peaks must be at least this percentage of the maximum peak to be detected. Lower values detect weaker reeds but may pick up noise or harmonics. Useful when one reed is much quieter than others.

**Measurement Stability:**

These features help achieve stable, accurate measurements by accumulating data over time. The stability bar on each reed panel fills as the reading stabilizes.

- **Temporal Smoothing** - Averages frequency measurements over time using weighted median filtering. This reduces jitter and provides more stable readings.
  - Window: Number of samples to average (0.5-4 seconds at ~10 Hz update rate)
  - Outliers are automatically rejected

- **Precision Mode** - Accumulates raw audio samples for high-resolution FFT analysis. This provides finer frequency resolution than the standard detection, especially useful for measuring very close frequencies or achieving sub-Hz accuracy.
  - Window: Audio accumulation duration (1-5 seconds)
  - Resolution improves with longer windows:
    - 2 seconds: ~0.5 Hz resolution
    - 3 seconds: ~0.37 Hz resolution
    - 4 seconds: ~0.28 Hz resolution
  - Buffer automatically resets when the detected note changes
  - Measurements become more accurate as the buffer fills

#### SimpleFFT Options

When SimpleFFT is selected, additional tuning options appear for fine-tuning close-frequency detection.

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Search Range** | 1.0-8.0 Hz | 3.0 Hz | Search range for additional reeds around the primary peak. |
| **Search Threshold** | 5-50% | 10% | Threshold for second-reed search relative to maximum magnitude. Lower values detect quieter reeds but may pick up noise. |

**How SimpleFFT detects multiple reeds:**

1. **Pass 1** - Uses FFT peak detection to find the primary peak above the threshold
2. **Pass 2** - Searches bidirectionally (both lower and higher frequencies) around the primary peak to find additional reeds within the search range

This two-pass approach makes SimpleFFT particularly effective for very close reeds (<1 Hz apart) and for 3-4 reed accordions where reeds are tuned on both sides of the reference reed. Best results require several seconds of stable signal to accumulate.

#### ESPRIT Options

When ESPRIT is selected, additional tuning options appear for fine-tuning close-frequency detection.

**How ESPRIT resolves close frequencies:**

Standard FFT has limited frequency resolution (about 0.67 Hz with the default settings). When two tremolo reeds are only 1-2 Hz apart, they appear as a single "merged" peak in the spectrum rather than two distinct peaks:

```
Two reeds 2 Hz apart in FFT:

Single reed:              Merged reeds:
      ▲                        ▲
     ███                    ██████
    █████                  ████████
   ███████                ██████████
  (narrow)               (wider shoulders)
```

ESPRIT detects merged peaks by checking if the "shoulders" of a peak are higher than expected for a single frequency. When detected, it adds candidate frequencies around the peak and uses subspace methods to resolve the actual frequencies. The candidates define a search neighborhood, but ESPRIT finds the true frequencies (within ~5 Hz of candidates) based on the signal itself - so detected frequencies are not limited to exact candidate positions.

**Parameters:**

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Width Threshold** | 0.05-0.50 | 0.25 | Threshold for detecting merged peaks. Lower values are more sensitive to close frequencies but may produce false detections. Values below 0.08 may trigger on single frequencies. |
| **Min Separation** | 0.30-1.00 Hz | 0.50 Hz | Minimum frequency separation between detected reeds. Lower values allow resolving closer frequencies. |
| **Candidate Offsets** | Presets | ±0.4, ±0.8 Hz | Frequency offsets added around merged peaks to help ESPRIT resolve close frequencies. |

**Candidate Offset Presets:**
- **±0.4, ±0.8 Hz (default)** - Good for typical tremolo (1-3 Hz beat)
- **±0.3, ±0.6 Hz (tighter)** - For very close frequencies (<1 Hz)
- **±0.5, ±1.0 Hz (wider)** - For wider tremolo (3-5 Hz beat)
- **±0.4, ±0.8, ±1.2 Hz (extended)** - Maximum range for variable tremolo
- **None (disable)** - Disable merged peak detection entirely

**Troubleshooting ESPRIT:**
- If close frequencies aren't detected: Lower Width Threshold (try 0.15-0.20) and/or lower Min Separation (try 0.35-0.40)
- If you get false/spurious frequencies: Raise Width Threshold (try 0.30-0.40) and/or raise Min Separation
- The theoretical resolution limit is approximately 0.6 Hz (frequencies closer than this cannot be reliably resolved)

### Tuning Tab

- **Temperament** - Select from 32 historical temperaments including Equal, Pythagorean, various Meantones, Werckmeister, Kirnberger, and more.
- **Key** - Root key for the temperament (C through B). Only affects non-equal temperaments.
- **Transpose** - Transpose the display by semitones (-6 to +6). Useful for transposing instruments.
- **Tremolo Profile** - Load a custom tremolo tuning profile. See [Tremolo Tuning Profiles](#tremolo-tuning-profiles).

### Display Tab

- **Lock Display** - Freeze the display values. Useful for examining a measurement without it updating.
- **Hold Mode** - Automatically captures and holds the best measurement when a note ends. The display freezes on the strongest signal detected during the note.
- **Zoom Spectrum** - When enabled, the spectrum view zooms to center on the detected note.

### Audio Tab

- **Input Device** - Select the audio input device. Use "Default" for the system default, or choose a specific microphone.

## Measurement Log

The measurement log window (accessible from the menu) allows you to record tuning measurements for later analysis or documentation.

**Recording Modes:**
- **Hold Mode** - Automatically records when a note is captured in hold mode. Best for measuring individual reeds one at a time.
- **Timed Mode** - Records at regular intervals (0.5-30 seconds). Useful for continuous monitoring.

**Data Recorded:**
- Timestamp
- Note name and octave
- Reference frequency
- Up to 4 reed frequencies with cents deviation

**Export:**
Click "Copy to Clipboard" to export all measurements in tab-separated format, ready for pasting into Excel or other spreadsheet applications.

## Tremolo Tuning Profiles

Load custom tremolo tuning profiles to show deviation from target beat frequencies rather than just deviation from reference pitch. This is useful when tuning to a specific tremolo specification.

**File Format:**
Tab or comma-separated values with note name and beat frequency in Hz:
```
G2,1.0
G#2,1.05
A2,1.1
# Comments start with #
A#2,1.15
```

**How It Works:**
When a profile is loaded, the cents display shows how far each reed is from its *target* position:

| Mode | Reed 1 | Reed 2 | Reed 3 | Reed 4 |
|------|--------|--------|--------|--------|
| 1 reed | deviation from nearest target* | - | - | - |
| 2 reed | reference (0¢) | deviation from (ref + beat) | - | - |
| 3 reed | deviation from (ref - beat) | reference (0¢) | deviation from (ref + beat) | - |
| 4 reed | deviation from (ref - beat) | reference (0¢) | reference (0¢) | deviation from (ref + beat) |

*For 1 reed mode, the target is whichever of (ref + beat) or (ref - beat) is closest to the detected frequency. This allows tuning individual tremolo reeds one at a time.

**Usage:**
1. Go to Settings > Tuning tab
2. Click "Load..." and select your profile CSV/TSV file
3. The profile appears in the dropdown and is automatically selected
4. Play a note - the display now shows deviation from target
5. Select "None" to return to standard cents display

A sample profile is included at `examples/tremolo_ora_13_cents.csv`.

## Acknowledgments

Originally inspired by [billthefarmer/ctuner](https://github.com/billthefarmer/ctuner), a cross-platform musical instrument strobe tuner.
