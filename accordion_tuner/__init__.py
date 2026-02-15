"""
accordion_tuner - Accordion reed tuner with multi-pitch detection and temperament support
"""

from .accordion import (
    AccordionDetector,
    AccordionResult,
    DetectionMode,
    DetectorType,
    NoteGroup,
    ReedInfo,
)
from .constants import A4_REFERENCE, BUFFER_SIZE, NOTE_NAMES, SAMPLE_RATE
from .esprit_detector import EspritPitchDetector
from .multi_pitch_detector import Maximum, MultiPitchDetector, MultiPitchResult
from .temperaments import TEMPERAMENTS, Temperament

__version__ = "0.1.6"
__all__ = [
    "MultiPitchDetector",
    "EspritPitchDetector",
    "MultiPitchResult",
    "Maximum",
    "Temperament",
    "TEMPERAMENTS",
    "SAMPLE_RATE",
    "BUFFER_SIZE",
    "A4_REFERENCE",
    "NOTE_NAMES",
    "AccordionDetector",
    "AccordionResult",
    "DetectorType",
    "DetectionMode",
    "NoteGroup",
    "ReedInfo",
]
