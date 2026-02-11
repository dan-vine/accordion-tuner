"""
Audio processing constants - matching the C++ implementation
"""

# Audio capture settings
SAMPLE_RATE = 11025
BUFFER_SIZE = 1024  # STEP in C++ code
FFT_SIZE = 4096

# Reference values
A4_REFERENCE = 440.0
C5_OFFSET = 57  # MIDI note number offset (A4 = 69, so C0 = 12, offset = 57)
A_OFFSET = 9    # A is 9 semitones above C
OCTAVE = 12

# Note names
NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Display variants
DISPLAY_NOTES = ["C", "C", "D", "E", "E", "F", "F", "G", "A", "A", "B", "B"]
DISPLAY_SHARPS = ["", "#", "", "b", "", "", "#", "", "b", "", "b", ""]
