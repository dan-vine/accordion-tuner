# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run GUI
accordion-tuner

# Run CLI
accordion-tuner-cli

# Run all tests
pytest

# Run single test file
pytest tests/test_accordion.py

# Run single test
pytest tests/test_accordion.py::TestAccordionDetector::test_process_single_frequency -v

# Lint
ruff check .

# Lint and fix
ruff check --fix .
```

## Architecture

This is an accordion reed tuner with multi-pitch detection for tuning 1-4 simultaneous reeds (tremolo/musette).

### Core Detection Pipeline

```
Audio Input (sounddevice)
    ↓
MultiPitchDetector (multi_pitch_detector.py)
    - FFT + phase vocoder for accurate pitch
    - Detects up to 8 frequency peaks
    - Applies temperament-adjusted reference frequencies
    ↓
AccordionDetector (accordion.py)
    - Groups peaks into reeds for same note
    - Calculates beat frequencies between reeds
    - Optionally computes target_cents from TremoloProfile
    ↓
AccordionResult → GUI display
```

### Key Modules

- **multi_pitch_detector.py** - FFT-based pitch detection matching C++ ctuner algorithm. Uses 16384-sample FFT with 1024-sample hop, phase vocoder for sub-bin accuracy.
- **accordion.py** - Wraps MultiPitchDetector, groups detected peaks into reed sets, calculates beat frequencies. `ReedInfo` contains per-reed data, `AccordionResult` is the detection output.
- **tremolo_profile.py** - Loads CSV/TSV profiles mapping notes to target beat frequencies. When active, `target_cents` shows deviation from target rather than reference.
- **temperaments.py** - 32 historical temperaments as frequency ratios. `Temperament` enum indexes into `TEMPERAMENT_RATIOS`.
- **constants.py** - Audio settings (SAMPLE_RATE=11025), note names, reference values.

### GUI (PySide6)

- **accordion_window.py** - Main window, orchestrates audio capture and display updates
- **reed_panel.py** - Individual reed display (frequency, cents, beat)
- **spectrum_view.py** - FFT spectrum visualization with peak markers
- **tuning_meter.py** - Multi-reed visual tuning indicator
- **measurement_log.py** - Recording measurements for export

### Data Flow for Tremolo Profiles

When a profile is loaded:
1. `AccordionDetector.set_tremolo_profile(profile)` stores the profile
2. During `_group_reeds()`, if profile exists, `_compute_target_cents()` calculates deviation from target frequency (ref ± beat) for each reed position
3. `ReedInfo.target_cents` is set (None when no profile)
4. GUI displays `target_cents` if available, otherwise `cents`