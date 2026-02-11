"""
accordion_tuner - Accordion reed tuner with multi-pitch detection and temperament support
"""

from .accordion import AccordionDetector, AccordionResult, ReedInfo
from .constants import A4_REFERENCE, BUFFER_SIZE, NOTE_NAMES, SAMPLE_RATE
from .multi_pitch_detector import Maximum, MultiPitchDetector, MultiPitchResult
from .temperaments import TEMPERAMENTS, Temperament

__version__ = "0.1.0"
__all__ = [
    "MultiPitchDetector",
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
    "ReedInfo",
]
