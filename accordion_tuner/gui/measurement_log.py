"""
Measurement log window for recording and exporting tuning measurements.
"""

from dataclasses import dataclass, field
from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..accordion import AccordionResult
from .styles import (
    BORDER_COLOR,
    MAIN_WINDOW_STYLE,
    PANEL_BACKGROUND,
    TEXT_SECONDARY,
)


@dataclass
class MeasurementEntry:
    """A single recorded measurement."""
    timestamp: str  # "HH:MM:SS"
    note_name: str  # "A4"
    ref_frequency: float  # 440.0
    reeds: list[tuple[float, float]] = field(default_factory=list)  # [(freq, cents), ...]


class MeasurementLogWindow(QWidget):
    """
    Window for recording and exporting tuning measurements.

    Supports two recording modes:
    - Hold mode: automatically record when a note is captured (enters HOLDING state)
    - Timed interval: record every N seconds
    """

    # Signal to request hold mode enable in main window
    request_hold_mode = Signal(bool)
    # Signal to request timed recording: (enabled, interval_sec)
    request_timed_recording = Signal(bool, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Measurement Log")
        self.setMinimumSize(600, 400)
        self.setWindowFlags(Qt.Window)  # Non-modal separate window

        self._entries: list[MeasurementEntry] = []
        self._recording = False

        self._setup_ui()
        self._apply_style()

    def _setup_ui(self):
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Recording mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Recording:"))

        self._hold_mode_rb = QRadioButton("Hold Mode")
        self._hold_mode_rb.setToolTip("Record when a note is captured in hold mode")
        self._hold_mode_rb.setChecked(True)
        mode_layout.addWidget(self._hold_mode_rb)

        self._timed_mode_rb = QRadioButton("Timed")
        self._timed_mode_rb.setToolTip("Record at regular intervals")
        mode_layout.addWidget(self._timed_mode_rb)

        # Button group for radio buttons
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._hold_mode_rb, 0)
        self._mode_group.addButton(self._timed_mode_rb, 1)

        mode_layout.addSpacing(20)

        # Interval spinner (for timed mode)
        mode_layout.addWidget(QLabel("Interval:"))
        self._interval_spin = QDoubleSpinBox()
        self._interval_spin.setRange(0.5, 30.0)
        self._interval_spin.setValue(2.0)
        self._interval_spin.setSuffix(" sec")
        self._interval_spin.setDecimals(1)
        self._interval_spin.setSingleStep(0.5)
        self._interval_spin.setEnabled(False)  # Disabled when hold mode selected
        mode_layout.addWidget(self._interval_spin)

        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        # Connect mode change to enable/disable interval spinner
        self._mode_group.buttonClicked.connect(self._on_mode_changed)

        # Recording controls
        controls_layout = QHBoxLayout()

        self._start_btn = QPushButton("Start Recording")
        self._start_btn.clicked.connect(self._on_start_recording)
        controls_layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop Recording")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_recording)
        controls_layout.addWidget(self._stop_btn)

        controls_layout.addStretch()

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        controls_layout.addWidget(self._status_label)

        layout.addLayout(controls_layout)

        # Table for measurements
        self._table = QTableWidget()
        self._table.setColumnCount(11)
        self._table.setHorizontalHeaderLabels([
            "Time", "Note", "Ref Hz",
            "Reed1 Hz", "Reed1 ¢",
            "Reed2 Hz", "Reed2 ¢",
            "Reed3 Hz", "Reed3 ¢",
            "Reed4 Hz", "Reed4 ¢",
        ])

        # Configure table appearance
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.verticalHeader().setVisible(False)

        # Set column resize modes
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Time
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Note
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Ref Hz
        for i in range(3, 11):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        layout.addWidget(self._table)

        # Bottom buttons
        buttons_layout = QHBoxLayout()

        self._copy_btn = QPushButton("Copy to Clipboard")
        self._copy_btn.setToolTip("Copy all measurements in tab-separated format")
        self._copy_btn.clicked.connect(self._copy_to_clipboard)
        buttons_layout.addWidget(self._copy_btn)

        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.clicked.connect(self._clear_all)
        buttons_layout.addWidget(self._clear_btn)

        buttons_layout.addStretch()

        self._count_label = QLabel("0 entries")
        self._count_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        buttons_layout.addWidget(self._count_label)

        layout.addLayout(buttons_layout)

    def _apply_style(self):
        """Apply styling to the window."""
        self.setStyleSheet(MAIN_WINDOW_STYLE + f"""
            QTableWidget {{
                background-color: {PANEL_BACKGROUND};
                border: 1px solid {BORDER_COLOR};
                gridline-color: {BORDER_COLOR};
            }}
            QTableWidget::item {{
                padding: 4px;
            }}
            QTableWidget::item:alternate {{
                background-color: #252525;
            }}
            QHeaderView::section {{
                background-color: {PANEL_BACKGROUND};
                border: 1px solid {BORDER_COLOR};
                padding: 4px;
                font-weight: bold;
            }}
        """)

    def _on_mode_changed(self):
        """Handle recording mode change."""
        timed_mode = self._timed_mode_rb.isChecked()
        self._interval_spin.setEnabled(timed_mode)

    def _on_start_recording(self):
        """Start recording measurements."""
        self._recording = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._hold_mode_rb.setEnabled(False)
        self._timed_mode_rb.setEnabled(False)
        self._interval_spin.setEnabled(False)

        if self._hold_mode_rb.isChecked():
            # Request hold mode be enabled in main window
            self.request_hold_mode.emit(True)
            self._status_label.setText("Recording (Hold Mode)")
        else:
            # Request timed recording
            interval = self._interval_spin.value()
            self.request_timed_recording.emit(True, interval)
            self._status_label.setText(f"Recording (every {interval:.1f}s)")

    def _on_stop_recording(self):
        """Stop recording measurements."""
        self._recording = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._hold_mode_rb.setEnabled(True)
        self._timed_mode_rb.setEnabled(True)
        self._interval_spin.setEnabled(self._timed_mode_rb.isChecked())

        # Emit signals to stop recording
        if self._hold_mode_rb.isChecked():
            self.request_hold_mode.emit(False)
        else:
            self.request_timed_recording.emit(False, 0.0)

        self._status_label.setText("Stopped")

    def add_entry(self, result: AccordionResult):
        """Add a measurement entry from an AccordionResult."""
        if not result.valid:
            return

        # Create entry
        timestamp = datetime.now().strftime("%H:%M:%S")
        note_name = f"{result.note_name}{result.octave}"
        reeds = [(r.frequency, r.cents) for r in result.reeds]

        entry = MeasurementEntry(
            timestamp=timestamp,
            note_name=note_name,
            ref_frequency=result.ref_frequency,
            reeds=reeds,
        )
        self._entries.append(entry)

        # Add row to table
        row = self._table.rowCount()
        self._table.insertRow(row)

        self._table.setItem(row, 0, QTableWidgetItem(entry.timestamp))
        self._table.setItem(row, 1, QTableWidgetItem(entry.note_name))
        self._table.setItem(row, 2, QTableWidgetItem(f"{entry.ref_frequency:.2f}"))

        # Add reed data (up to 4 reeds)
        for i, (freq, cents) in enumerate(entry.reeds):
            if i >= 4:
                break
            col_freq = 3 + i * 2
            col_cents = 4 + i * 2
            self._table.setItem(row, col_freq, QTableWidgetItem(f"{freq:.2f}"))
            cents_str = f"{cents:+.1f}" if cents != 0 else "0.0"
            self._table.setItem(row, col_cents, QTableWidgetItem(cents_str))

        # Scroll to bottom
        self._table.scrollToBottom()

        # Update count
        self._count_label.setText(f"{len(self._entries)} entries")

    def _copy_to_clipboard(self):
        """Copy all measurements to clipboard in tab-separated format."""
        if not self._entries:
            return

        lines = []

        # Header
        headers = [
            "Time", "Note", "Ref Hz",
            "Reed1 Hz", "Reed1 ¢",
            "Reed2 Hz", "Reed2 ¢",
            "Reed3 Hz", "Reed3 ¢",
            "Reed4 Hz", "Reed4 ¢",
        ]
        lines.append("\t".join(headers))

        # Data rows
        for entry in self._entries:
            row = [
                entry.timestamp,
                entry.note_name,
                f"{entry.ref_frequency:.2f}",
            ]

            # Add reed data (up to 4 reeds)
            for i in range(4):
                if i < len(entry.reeds):
                    freq, cents = entry.reeds[i]
                    row.append(f"{freq:.2f}")
                    cents_str = f"{cents:+.1f}" if cents != 0 else "0.0"
                    row.append(cents_str)
                else:
                    row.append("")
                    row.append("")

            lines.append("\t".join(row))

        # Copy to clipboard
        text = "\n".join(lines)
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

        self._status_label.setText(f"Copied {len(self._entries)} entries")

    def _clear_all(self):
        """Clear all measurements."""
        self._entries.clear()
        self._table.setRowCount(0)
        self._count_label.setText("0 entries")
        self._status_label.setText("Cleared")

    def is_recording(self) -> bool:
        """Return whether recording is active."""
        return self._recording

    def closeEvent(self, event):
        """Handle window close - stop recording if active."""
        if self._recording:
            self._on_stop_recording()
        event.accept()
