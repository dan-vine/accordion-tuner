# Accordion Tuner

A musical instrument tuner specialized for accordion reed tuning with multi-pitch detection, temperament support, and beat frequency measurement.

![Accordion Tuner Screenshot](images/screenshot.png)

## Features

- Multi-reed detection (1-4 simultaneous reeds)
- Beat frequency calculation for tremolo/musette tuning
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

## Acknowledgments

Originally inspired by [billthefarmer/ctuner](https://github.com/billthefarmer/ctuner), a cross-platform musical instrument strobe tuner.
