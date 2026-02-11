"""
Reed panel widget - complete panel for one reed.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QProgressBar, QVBoxLayout

from .styles import (
    ACCENT_GREEN,
    BORDER_COLOR,
    ERROR_RED,
    PANEL_BACKGROUND,
    TEXT_SECONDARY,
    WARNING_ORANGE,
)


class ReedPanel(QFrame):
    """
    Panel displaying information for a single reed.

    Shows:
    - Reed number/title
    - Frequency in Hz
    - Cents deviation with color coding
    - Visual tuning meter
    - Beat frequency (if applicable)
    """

    def __init__(self, reed_number: int = 1, parent=None):
        super().__init__(parent)
        self.setObjectName("reedPanel")
        self._reed_number = reed_number
        self._setup_ui()
        self._apply_style()

    def _setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 12, 15, 12)
        layout.setSpacing(8)

        # Reed title
        self._title_label = QLabel(f"Reed {self._reed_number}")
        self._title_label.setObjectName("reedTitle")
        self._title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._title_label)

        # Cents display (large, centered)
        self._cents_label = QLabel("--")
        self._cents_label.setObjectName("centsLabel")
        self._cents_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._cents_label)

        # Frequency display
        self._freq_label = QLabel("-- Hz")
        self._freq_label.setObjectName("frequencyLabel")
        self._freq_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._freq_label)

        # Beat frequency display
        self._beat_label = QLabel("")
        self._beat_label.setObjectName("beatLabel")
        self._beat_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._beat_label)

        # Stability indicator bar
        self._stability_bar = QProgressBar()
        self._stability_bar.setObjectName("stabilityBar")
        self._stability_bar.setRange(0, 100)
        self._stability_bar.setValue(0)
        self._stability_bar.setTextVisible(False)
        self._stability_bar.setMaximumHeight(6)
        self._stability_bar.setToolTip("Measurement stability (fills as reading stabilizes)")
        layout.addWidget(self._stability_bar)

        layout.addStretch()

    def _apply_style(self):
        """Apply styling to the widget."""
        self.setStyleSheet(f"""
            QFrame#reedPanel {{
                background-color: {PANEL_BACKGROUND};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
            }}
            QLabel#reedTitle {{
                font-size: 14px;
                font-weight: bold;
                color: {TEXT_SECONDARY};
            }}
            QLabel#centsLabel {{
                font-size: 28px;
                font-weight: bold;
            }}
            QLabel#frequencyLabel {{
                font-size: 14px;
                color: {TEXT_SECONDARY};
            }}
            QLabel#beatLabel {{
                font-size: 12px;
                color: {TEXT_SECONDARY};
            }}
            QProgressBar#stabilityBar {{
                background-color: #3a3a3a;
                border: none;
                border-radius: 3px;
            }}
            QProgressBar#stabilityBar::chunk {{
                background-color: {ACCENT_GREEN};
                border-radius: 3px;
            }}
        """)
        self.setMinimumWidth(160)
        self.setMinimumHeight(180)

    def set_data(
        self,
        frequency: float,
        cents: float,
        beat_frequency: float = None,
        target_cents: float | None = None,
        stability: float = 0.0,
        sample_count: int = 0,
    ):
        """
        Set reed data.

        Args:
            frequency: Detected frequency in Hz
            cents: Deviation from reference in cents
            beat_frequency: Beat frequency with next reed (optional)
            target_cents: Deviation from target when tremolo profile is active (optional)
            stability: Measurement stability (0.0-1.0)
            sample_count: Number of samples in the smoothed average
        """
        # Update frequency
        self._freq_label.setText(f"{frequency:.2f} Hz")

        # Use target_cents if provided, otherwise use regular cents
        display_cents = target_cents if target_cents is not None else cents

        # Update cents with sign and color
        sign = "+" if display_cents >= 0 else ""
        self._cents_label.setText(f"{sign}{display_cents:.1f}¢")
        self._cents_label.setStyleSheet(f"color: {self._get_cents_color(display_cents)};")

        # Update beat frequency
        if beat_frequency is not None:
            self._beat_label.setText(f"Beat: {beat_frequency:.2f} Hz")
        else:
            self._beat_label.setText("")

        # Update stability bar
        # Combine stability score with sample count for a "settling" indicator
        # Need at least 5 samples for meaningful stability, max out at 20
        sample_factor = min(1.0, sample_count / 10.0)  # Ramps up over first 10 samples
        combined_stability = stability * sample_factor
        self._stability_bar.setValue(int(combined_stability * 100))

    def set_inactive(self):
        """Set panel to inactive state."""
        self._cents_label.setText("--")
        self._cents_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self._freq_label.setText("-- Hz")
        self._beat_label.setText("")
        self._stability_bar.setValue(0)

    def _get_cents_color(self, cents: float) -> str:
        """Get color based on cents deviation."""
        abs_cents = abs(cents)
        if abs_cents <= 5:
            return ACCENT_GREEN
        elif abs_cents <= 15:
            return WARNING_ORANGE
        else:
            return ERROR_RED
