# Accordion Tuner

A musical instrument tuner specialized for accordion reed tuning with multi-pitch detection, temperament support, and beat frequency measurement.

![Accordion Tuner Screenshot](images/screenshot.png)

## Features

- Multi-reed detection (1-4 simultaneous reeds)
- Beat frequency calculation for tremolo/musette tuning
- Custom tremolo tuning profiles (CSV/TSV)
- Multiple detection algorithms: FFT and ESPRIT
- ESPRIT algorithm for super-resolution detection of close frequencies (<1 Hz apart)
- Phase vocoder for accurate pitch detection
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
- **FFT (Phase Vocoder)** - Default algorithm. Fast and reliable for most tuning scenarios. Uses FFT with phase vocoder for sub-bin frequency accuracy.
- **ESPRIT** - Best for closely-spaced frequencies (1-5 Hz tremolo reeds). Uses subspace methods to achieve super-resolution frequency estimation beyond the FFT bin width. Recommended when FFT cannot resolve your tremolo reeds.

**General Settings:**
- **Octave Filter** - When enabled, restricts detection to one octave above the fundamental. Keep OFF for accordion tuning to detect closely-spaced tremolo reeds.
- **Fundamental Filter** - Only detect harmonics of the fundamental frequency. Useful for filtering out non-harmonic noise.
- **Downsample** - Enables downsampling for better low frequency detection. Helpful for bass reeds. *(FFT only)*
- **Sensitivity** - Detection threshold (0.05-0.50). Lower values increase sensitivity but may detect more background noise.
- **Reed Spread** - Maximum cents deviation to group as the same note (20-100¢). Increase if your tremolo reeds have wide detuning.

#### ESPRIT Options

When ESPRIT is selected, additional tuning options appear for fine-tuning close-frequency detection:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Width Threshold** | 0.10-0.50 | 0.25 | Threshold for detecting merged peaks. Lower values are more sensitive to close frequencies but may produce false detections. |
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
