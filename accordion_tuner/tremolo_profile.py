"""
Tremolo tuning profile support.

This module provides loading and management of custom tremolo tuning profiles
from CSV/TSV files. When a profile is active, the tuner shows how far each
reed is from its TARGET position based on the profile's beat frequency for
that note.
"""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TremoloProfile:
    """
    A tremolo tuning profile mapping notes to target beat frequencies.

    Attributes:
        name: Display name of the profile (usually filename without extension)
        path: Original file path the profile was loaded from
        beat_frequencies: Mapping of note name (e.g., "G2", "A#3") to beat frequency in Hz
    """
    name: str = ""
    path: str = ""
    beat_frequencies: dict[str, float] = field(default_factory=dict)

    def get_beat_frequency(self, note_name: str, octave: int) -> float | None:
        """
        Get the beat frequency for a specific note.

        Args:
            note_name: Note name (e.g., "C", "F#", "Bb")
            octave: Octave number (e.g., 2, 3, 4)

        Returns:
            Beat frequency in Hz, or None if not found in profile
        """
        # Normalize note name: convert flats to sharps for lookup
        normalized = self._normalize_note_name(note_name)
        key = f"{normalized}{octave}"

        if key in self.beat_frequencies:
            return self.beat_frequencies[key]

        return None

    def _normalize_note_name(self, note_name: str) -> str:
        """Normalize note name to use sharps instead of flats."""
        # Map flats to sharps
        flat_to_sharp = {
            "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
            "Ab": "G#", "Bb": "A#", "Cb": "B"
        }
        return flat_to_sharp.get(note_name, note_name)


def load_profile(path: str) -> TremoloProfile:
    """
    Load a tremolo profile from a CSV or TSV file.

    File format: Tab or comma-separated values with note name and beat frequency.
    Lines starting with # are treated as comments.

    Example:
        G2    1.231
        G#2   1.279
        A2    1.329
        # This is a comment
        A#2,1.381

    Args:
        path: Path to the profile file

    Returns:
        TremoloProfile with the loaded data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    profile = TremoloProfile(
        name=file_path.stem,
        path=str(file_path.absolute()),
        beat_frequencies={},
    )

    # Pattern to match note names like "C2", "F#3", "Bb4"
    note_pattern = re.compile(r'^([A-Ga-g][#b]?)(\d+)$')

    with open(file_path, 'r', encoding='utf-8') as f:
        # Try to detect delimiter
        content = f.read()
        f.seek(0)

        # Determine delimiter: tab or comma
        if '\t' in content:
            delimiter = '\t'
        else:
            delimiter = ','

        reader = csv.reader(f, delimiter=delimiter)
        line_num = 0

        for row in reader:
            line_num += 1

            # Skip empty lines
            if not row:
                continue

            # Skip comment lines
            first_cell = row[0].strip()
            if first_cell.startswith('#'):
                continue

            # Need at least note and beat frequency
            if len(row) < 2:
                # Try splitting by whitespace if only one column
                parts = first_cell.split()
                if len(parts) >= 2:
                    row = parts
                else:
                    continue

            note_str = row[0].strip()
            beat_str = row[1].strip()

            # Parse note name
            match = note_pattern.match(note_str)
            if not match:
                continue  # Skip invalid lines silently

            note_name = match.group(1).upper()
            # Handle lowercase 'b' for flat
            if len(note_name) > 1 and note_name[1] == 'B':
                note_name = note_name[0] + 'b'
            octave = match.group(2)

            # Normalize to sharps for storage
            flat_to_sharp = {
                "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
                "Ab": "G#", "Bb": "A#", "Cb": "B"
            }
            if note_name in flat_to_sharp:
                note_name = flat_to_sharp[note_name]

            # Parse beat frequency
            try:
                beat_freq = float(beat_str)
                if beat_freq < 0:
                    continue  # Skip negative values
            except ValueError:
                continue  # Skip invalid numbers

            key = f"{note_name}{octave}"
            profile.beat_frequencies[key] = beat_freq

    if not profile.beat_frequencies:
        raise ValueError(f"No valid entries found in profile: {path}")

    return profile
