# Accordion Tuner

A musical instrument tuner specialized for accordion reed tuning with multi-pitch detection, temperament support, and beat frequency measurement.

![Accordion Tuner Screenshot](images/screenshot.png)

## Features

- Multi-reed detection (1-4 simultaneous reeds)
- Beat frequency calculation for tremolo/musette tuning
- Custom tremolo tuning profiles (CSV/TSV)
- 32 historical temperaments
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

- **Octave Filter** - When enabled, restricts detection to one octave above the fundamental. Keep OFF for accordion tuning to detect closely-spaced tremolo reeds.
- **Fundamental Filter** - Only detect harmonics of the fundamental frequency. Useful for filtering out non-harmonic noise.
- **Downsample** - Enables downsampling for better low frequency detection. Helpful for bass reeds.
- **Sensitivity** - Detection threshold (0.05-0.50). Lower values increase sensitivity but may detect more background noise.
- **Reed Spread** - Maximum cents deviation to group as the same note (20-100¢). Increase if your tremolo reeds have wide detuning.

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
| 2 reed | reference (0¢) | deviation from (ref + beat) | - | - |
| 3 reed | deviation from (ref - beat) | reference (0¢) | deviation from (ref + beat) | - |
| 4 reed | deviation from (ref - beat) | reference (0¢) | reference (0¢) | deviation from (ref + beat) |

**Usage:**
1. Go to Settings > Tuning tab
2. Click "Load..." and select your profile CSV/TSV file
3. The profile appears in the dropdown and is automatically selected
4. Play a note - the display now shows deviation from target
5. Select "None" to return to standard cents display

A sample profile is included at `examples/tremolo_ora_13_cents.csv`.

## Acknowledgments

Originally inspired by [billthefarmer/ctuner](https://github.com/billthefarmer/ctuner), a cross-platform musical instrument strobe tuner.
