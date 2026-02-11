"""
Note display widget - large note name with subscript octave.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from .styles import (
    BORDER_COLOR,
    PANEL_BACKGROUND,
    TEXT_COLOR,
    TEXT_SECONDARY,
)


class NoteDisplay(QFrame):
    """
    Large note name display with subscript octave.

    Shows:
    - Reference frequency (e.g., "A: 440.0 Hz") at the top
    - Large note name (e.g., "C") in the center
    - Subscript octave number (e.g., "4")
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("noteDisplay")
        self._setup_ui()
        self._apply_style()

    def _setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(5)

        # Reference frequency label
        self._ref_label = QLabel("A: 440.0 Hz")
        self._ref_label.setObjectName("refFrequency")
        self._ref_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._ref_label)

        # Note name container (note + octave)
        note_container = QHBoxLayout()
        note_container.setSpacing(2)
        note_container.setAlignment(Qt.AlignCenter)

        # Note name
        self._note_label = QLabel("--")
        self._note_label.setObjectName("noteName")
        self._note_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        note_container.addWidget(self._note_label)

        # Octave (subscript-style)
        self._octave_label = QLabel("")
        self._octave_label.setObjectName("octaveLabel")
        self._octave_label.setAlignment(Qt.AlignLeft | Qt.AlignBottom)
        note_container.addWidget(self._octave_label)

        layout.addLayout(note_container)
        layout.addStretch()

    def _apply_style(self):
        """Apply styling to the widget."""
        self.setStyleSheet(f"""
            QFrame#noteDisplay {{
                background-color: {PANEL_BACKGROUND};
                border: 1px solid {BORDER_COLOR};
                border-radius: 12px;
            }}
            QLabel#noteName {{
                font-size: 72px;
                font-weight: bold;
                color: {TEXT_COLOR};
            }}
            QLabel#octaveLabel {{
                font-size: 36px;
                color: {TEXT_SECONDARY};
                padding-bottom: 8px;
            }}
            QLabel#refFrequency {{
                font-size: 14px;
                color: {TEXT_SECONDARY};
            }}
        """)
        self.setMinimumSize(180, 140)

    def set_note(self, note_name: str, octave: int, ref_frequency: float):
        """
        Set the displayed note.

        Args:
            note_name: Note name (e.g., "C", "F#", "Bb")
            octave: Octave number (e.g., 4)
            ref_frequency: Reference frequency in Hz
        """
        self._note_label.setText(note_name)
        self._octave_label.setText(str(octave))
        self._ref_label.setText(f"{ref_frequency:.1f} Hz")

    def set_reference(self, reference: float):
        """Set the reference A4 frequency for display."""
        self._ref_label.setText(f"A: {reference:.1f} Hz")

    def set_inactive(self):
        """Set display to inactive state."""
        self._note_label.setText("--")
        self._octave_label.setText("")
