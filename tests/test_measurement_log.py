"""
Tests for measurement log functionality.
"""


from accordion_tuner.accordion import AccordionResult, NoteGroup, ReedInfo
from accordion_tuner.gui.measurement_log import MeasurementEntry


class TestMeasurementEntry:
    """Test MeasurementEntry dataclass."""

    def test_default_values(self):
        """Test default values."""
        entry = MeasurementEntry(
            timestamp="12:00:00",
            note_name="A4",
            ref_frequency=440.0,
        )
        assert entry.timestamp == "12:00:00"
        assert entry.note_name == "A4"
        assert entry.ref_frequency == 440.0
        assert entry.reeds == []
        assert entry.notes == []

    def test_with_reeds(self):
        """Test with reed data."""
        entry = MeasurementEntry(
            timestamp="12:00:00",
            note_name="A4",
            ref_frequency=440.0,
            reeds=[(440.0, 0.5), (442.0, 2.1)],
        )
        assert len(entry.reeds) == 2
        assert entry.reeds[0] == (440.0, 0.5)
        assert entry.reeds[1] == (442.0, 2.1)

    def test_with_notes(self):
        """Test with chord notes data."""
        entry = MeasurementEntry(
            timestamp="12:00:00",
            note_name="C4|E4|G4",
            ref_frequency=261.63,
            notes=[("C4", 261.63, 0.5), ("E4", 329.63, -0.3), ("G4", 392.0, 1.2)],
        )
        assert len(entry.notes) == 3
        assert entry.notes[0] == ("C4", 261.63, 0.5)
        assert entry.notes[1] == ("E4", 329.63, -0.3)
        assert entry.notes[2] == ("G4", 392.0, 1.2)


class TestMeasurementEntryParsing:
    """Test parsing entries from AccordionResult."""

    def create_result_with_reeds(self):
        """Create a test result with reed data."""
        result = AccordionResult(
            valid=True,
            note_name="A",  # Just the note letter, not including octave
            octave=4,
            ref_frequency=440.0,
            reeds=[
                ReedInfo(frequency=440.0, cents=0.5, magnitude=0.8),
                ReedInfo(frequency=442.0, cents=2.1, magnitude=0.7),
            ],
        )
        return result

    def create_result_with_notes(self):
        """Create a test result with chord notes."""
        result = AccordionResult(
            valid=True,
            note_name="C4",
            octave=4,
            ref_frequency=261.63,
            reeds=[ReedInfo(frequency=261.63, cents=0.5, magnitude=0.8)],
            notes=[
                NoteGroup(
                    note_name="C",
                    octave=4,
                    ref_frequency=261.63,
                    reeds=[ReedInfo(frequency=261.63, cents=0.5, magnitude=0.8)],
                    beat_frequencies=[],
                ),
                NoteGroup(
                    note_name="E",
                    octave=4,
                    ref_frequency=329.63,
                    reeds=[ReedInfo(frequency=329.63, cents=-0.3, magnitude=0.7)],
                    beat_frequencies=[],
                ),
                NoteGroup(
                    note_name="G",
                    octave=4,
                    ref_frequency=392.0,
                    reeds=[ReedInfo(frequency=392.0, cents=1.2, magnitude=0.6)],
                    beat_frequencies=[],
                ),
            ],
        )
        return result

    def test_parse_reed_result(self):
        """Test parsing reed result into entry format."""
        result = self.create_result_with_reeds()

        # Simulate what add_entry does for reeds
        is_chord = hasattr(result, 'notes') and len(result.notes) > 1

        if is_chord:
            note_names = [f"{n.note_name}{n.octave}" for n in result.notes]
            note_name = "|".join(note_names)
            ref_frequency = result.notes[0].ref_frequency if result.notes else result.ref_frequency
            notes = [(f"{n.note_name}{n.octave}", n.reeds[0].frequency, n.reeds[0].cents) for n in result.notes if n.reeds]
            reeds = []
        else:
            note_name = f"{result.note_name}{result.octave}"
            ref_frequency = result.ref_frequency
            reeds = [(r.frequency, r.cents) for r in result.reeds]
            notes = []

        assert is_chord is False
        assert note_name == "A4"
        assert ref_frequency == 440.0
        assert len(reeds) == 2
        assert reeds[0] == (440.0, 0.5)
        assert reeds[1] == (442.0, 2.1)
        assert notes == []

    def test_parse_chord_result(self):
        """Test parsing chord result into entry format."""
        result = self.create_result_with_notes()

        # Simulate what add_entry does for chords
        is_chord = hasattr(result, 'notes') and len(result.notes) > 1

        if is_chord:
            note_names = [f"{n.note_name}{n.octave}" for n in result.notes]
            note_name = "|".join(note_names)
            ref_frequency = result.notes[0].ref_frequency if result.notes else result.ref_frequency
            notes = [(f"{n.note_name}{n.octave}", n.reeds[0].frequency, n.reeds[0].cents) for n in result.notes if n.reeds]
            reeds = []
        else:
            note_name = f"{result.note_name}{result.octave}"
            ref_frequency = result.ref_frequency
            reeds = [(r.frequency, r.cents) for r in result.reeds]
            notes = []

        assert is_chord is True
        assert note_name == "C4|E4|G4"
        assert ref_frequency == 261.63
        assert len(notes) == 3
        assert notes[0] == ("C4", 261.63, 0.5)
        assert notes[1] == ("E4", 329.63, -0.3)
        assert notes[2] == ("G4", 392.0, 1.2)
        assert reeds == []

    def test_invalid_result(self):
        """Test that invalid result returns False for is_chord."""
        result = AccordionResult(valid=False)
        is_chord = hasattr(result, 'notes') and len(result.notes) > 1
        assert is_chord is False
