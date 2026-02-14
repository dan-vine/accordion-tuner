# AGENTS.md

This file provides guidance for AI agents working in this codebase.

## Quick Reference

### Installation
```bash
pip install -e ".[dev]"
```

### Running the Application
```bash
# Run GUI
accordion-tuner

# Run CLI
accordion-tuner-cli
```

### Testing
```bash
# Run all tests
pytest

# Run single test file
pytest tests/test_accordion.py

# Run single test (with full path)
pytest tests/test_accordion.py::TestAccordionDetector::test_process_single_frequency -v

# Run tests with coverage
pytest --cov=accordion_tuner
```

### Linting
```bash
# Lint all files
ruff check .

# Lint and auto-fix
ruff check --fix .

# Format code
ruff format .
```

---

## Code Style Guidelines

### General Principles
- **Line length**: 100 characters max (configured in ruff)
- **Python version**: 3.10+
- **Type hints**: Use them for function parameters and return types
- **Docstrings**: Use Google-style docstrings for public APIs

### Imports
- **Standard library first**, then third-party, then project imports
- **Group by category**: stdlib, third-party, project
- **Use absolute imports** within the package (e.g., `from accordion_tuner.accordion import ...`)
- **Sort with ruff** (will auto-sort with `ruff check --fix`)

Example:
```python
import os
import sys
from typing import NamedTuple

import numpy as np
from scipy.linalg import pinv
from PySide6.QtCore import QSettings, Qt

from accordion_tuner.accordion import AccordionDetector, DetectorType
from accordion_tuner.constants import SAMPLE_RATE
```

### Naming Conventions
- **Classes**: `PascalCase` (e.g., `EspritPitchDetector`)
- **Functions/methods**: `snake_case` (e.g., `process()`, `_fft_esprit()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `SAMPLE_RATE`, `K_SAMPLES`)
- **Private members**: Leading underscore (e.g., `self._buffer`, `_private_method()`)
- **Protected members**: Double underscore for name mangling (avoid unless needed)

### Error Handling
- **Use specific exceptions** when possible
- **Return None** or empty collections rather than raising for expected cases
- **Log errors** or provide user feedback for unexpected failures
- **Never expose raw exceptions** to users in GUI

Example:
```python
def _safe_eigvals(matrix: np.ndarray) -> np.ndarray | None:
    """Safely compute eigenvalues of a matrix."""
    try:
        return np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None
```

### Type Annotations
- Use `|` instead of `Union` for Python 3.10+
- Use `X | None` instead of `Optional[X]`
- Be explicit with container types: `list[float]`, `dict[str, int]`

### Docstrings
```python
def process(samples: np.ndarray) -> MultiPitchResult:
    """
    Process audio samples and detect multiple pitches.

    Args:
        samples: Audio samples as numpy array

    Returns:
        MultiPitchResult with detected notes
    """
```

### GUI Development (PySide6)
- **Use explicit imports** from PySide6 (not `from PySide6 import *`)
- **Connect signals** using `.connect()` not old-style
- **Use layouts** (QVBoxLayout, QHBoxLayout) for widget positioning
- **Set object names** for styling (`setObjectName("myWidget")`)
- **Use Settings** for persistence (`QSettings`)

---

## Architecture Overview

### Core Detection Pipeline

```
Audio Input (sounddevice)
    ↓
MultiPitchDetector (multi_pitch_detector.py)
    - FFT + phase vocoder for accurate pitch
    - Detects up to 8 frequency peaks
    ↓
EspritPitchDetector (esprit_detector.py)
    - FFT-ESPRIT for super-resolution close-freq detection
    - Uses covariance matrix + eigenvalue decomposition
    ↓
AccordionDetector (accordion.py)
    - Groups peaks into reeds for same note
    - Calculates beat frequencies
    - Applies temporal smoothing
    ↓
AccordionResult → GUI display
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `multi_pitch_detector.py` | FFT-based pitch detection |
| `esprit_detector.py` | ESPRIT algorithm for close frequency resolution |
| `accordion.py` | Main detector orchestrating the pipeline |
| `measurement_smoother.py` | Temporal smoothing for stable readings |
| `tremolo_profile.py` | Tremolo beat frequency profiles |
| `temperaments.py` | Historical temperaments (32 types) |

### GUI Components

| Component | Purpose |
|----------|---------|
| `accordion_window.py` | Main window, audio capture |
| `reed_panel.py` | Individual reed display |
| `spectrum_view.py` | FFT spectrum visualization |
| `tuning_meter.py` | Visual tuning indicator |

---

## Testing Guidelines

### Test Structure
- Use `pytest` with class-based tests
- Name test classes `Test<ClassName>`
- Name test methods `test_<description>`
- Use `pytest.approx()` for floating-point comparisons

Example:
```python
class TestReedSmoother:
    def test_single_measurement(self):
        smoother = ReedSmoother(max_samples=10)
        smoother.add(440.0, 0.0, 1.0)
        result = smoother.get_smoothed()
        assert result.frequency == pytest.approx(440.0)
```

### Running Tests
- All tests: `pytest`
- Single file: `pytest tests/test_accordion.py`
- Single test: `pytest tests/test_file.py::TestClass::test_method -v`
- With verbose output: `pytest -v`
- Stop on first failure: `pytest -x`

---

## Common Tasks

### Adding a New Detector Setting
1. Add to `EspritPitchDetector.__init__()` with default value
2. Add getter/setter methods
3. Add to `AccordionDetector` wrapper if needed
4. Add GUI control in `accordion_window.py`
5. Add to settings persistence in GUI

### Modifying the GUI
1. Changes typically go in `accordion_window.py`
2. Use existing widgets as patterns
3. Connect signals to handler methods
4. Test with both mouse and keyboard

### Debugging Audio Issues
- Use `tremolo_test2.npy` for testing (located in project root)
- Load with `np.load("tremolo_test2.npy")`
- Process chunks: `chunk = audio[start:start+16384]`

---

## Dependencies

- **numpy** >= 1.24.0 - Numerical computing
- **scipy** >= 1.10.0 - Signal processing, linear algebra
- **sounddevice** >= 0.4.6 - Audio input
- **PySide6** >= 6.5.0 - GUI framework
- **pytest** >= 7.0.0 - Testing (dev)
- **ruff** >= 0.4.0 - Linting (dev)
