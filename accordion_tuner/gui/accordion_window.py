"""
Main application window for accordion reed tuning.
"""

import sys

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..accordion import AccordionDetector, AccordionResult
from ..constants import A4_REFERENCE, NOTE_NAMES, SAMPLE_RATE
from ..temperaments import Temperament
from ..tremolo_profile import TremoloProfile, load_profile
from .measurement_log import MeasurementLogWindow
from .note_display import NoteDisplay
from .reed_panel import ReedPanel
from .spectrum_view import SpectrumView
from .styles import (
    BORDER_COLOR,
    MAIN_WINDOW_STYLE,
    PANEL_BACKGROUND,
    SETTINGS_PANEL_STYLE,
    TEXT_SECONDARY,
    TOGGLE_BUTTON_STYLE,
)
from .tuning_meter import MultiReedMeter


class AccordionWindow(QMainWindow):
    """
    Main window for accordion reed tuning application.

    Features:
    - Spectrum display showing FFT of audio
    - Large note display with reference frequency
    - Individual reed panels (2-4)
    - Settings for reference frequency and number of reeds
    """

    # Default settings values
    DEFAULTS = {
        'num_reeds': 3,
        'reference': 440.0,
        'octave_filter': False,
        'fundamental_filter': False,
        'downsample': False,
        'sensitivity': 10,
        'reed_spread': 50,
        'temperament': 8,  # Equal
        'key': 0,  # C
        'transpose': 0,
        'zoom_spectrum': True,
        'hold_mode': False,
        'settings_expanded': False,
        'tremolo_profile_path': '',  # Path to last loaded profile
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Accordion Reed Tuner")
        self.setMinimumSize(800, 600)

        # Detection - create first to get hop_size
        self._reference = A4_REFERENCE
        self._num_reeds = 3
        self._detector = AccordionDetector(
            sample_rate=SAMPLE_RATE,
            reference=self._reference,
            max_reeds=self._num_reeds,
        )

        # Audio settings - buffer size must match detector's hop_size for accurate phase vocoder
        self._sample_rate = SAMPLE_RATE
        self._buffer_size = self._detector._detector.hop_size  # 1024

        # Audio stream
        self._stream = None
        self._audio_buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._input_device = None  # None = default device

        # Display settings
        self._lock_display = False
        self._zoom_spectrum = True

        # Hold mode state
        self._hold_mode = False
        self._hold_state = "WAITING"  # WAITING, TRACKING, HOLDING
        self._held_result: AccordionResult | None = None
        self._held_magnitude = 0.0

        # UI components
        self._reed_panels: list[ReedPanel] = []
        self._settings_expanded = False

        # Tremolo profile state
        self._tremolo_profile: TremoloProfile | None = None
        self._loaded_profiles: dict[str, TremoloProfile] = {}  # name -> profile

        # Measurement log window state
        self._log_window: MeasurementLogWindow | None = None
        self._log_recording_hold = False  # Recording in hold mode
        self._log_recording_timed = False  # Recording in timed mode
        self._log_timer = QTimer()
        self._log_timer.timeout.connect(self._on_log_timer)

        self._setup_ui()
        self._apply_style()

        # Update timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_display)
        self._timer.start(50)  # 20 Hz update rate

        # Last result for display
        self._last_result: AccordionResult | None = None

        # Load saved settings (must be after UI setup)
        self._load_settings()

        # Start audio
        self._start_audio()

    def _setup_ui(self):
        """Set up the UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # Spectrum view (top)
        self._spectrum_view = SpectrumView()
        self._spectrum_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._spectrum_view.setMinimumHeight(150)
        main_layout.addWidget(self._spectrum_view)

        # Note display (center)
        note_container = QHBoxLayout()
        note_container.addStretch()
        self._note_display = NoteDisplay()
        note_container.addWidget(self._note_display)
        note_container.addStretch()
        main_layout.addLayout(note_container)

        # Reed panels (horizontal row)
        self._reed_container = QHBoxLayout()
        self._reed_container.setSpacing(15)
        self._create_reed_panels()
        main_layout.addLayout(self._reed_container)

        # Unified multi-reed tuning meter (below reed panels)
        self._multi_meter = MultiReedMeter(max_reeds=4)
        self._multi_meter.set_num_reeds(self._num_reeds)
        main_layout.addWidget(self._multi_meter)

        # Settings bar (bottom)
        settings_frame = QFrame()
        settings_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {PANEL_BACKGROUND};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
            }}
        """)
        settings_layout = QHBoxLayout(settings_frame)
        settings_layout.setContentsMargins(15, 10, 15, 10)

        # Number of reeds
        reeds_label = QLabel("Reeds:")
        settings_layout.addWidget(reeds_label)

        self._reeds_combo = QComboBox()
        self._reeds_combo.addItems(["1", "2", "3", "4"])
        self._reeds_combo.setCurrentIndex(2)  # Default to 3
        self._reeds_combo.currentIndexChanged.connect(self._on_reeds_changed)
        settings_layout.addWidget(self._reeds_combo)

        settings_layout.addSpacing(30)

        # Reference frequency
        ref_label = QLabel("Reference A4:")
        settings_layout.addWidget(ref_label)

        self._ref_spinbox = QDoubleSpinBox()
        self._ref_spinbox.setRange(400.0, 480.0)
        self._ref_spinbox.setValue(self._reference)
        self._ref_spinbox.setSuffix(" Hz")
        self._ref_spinbox.setDecimals(1)
        self._ref_spinbox.setSingleStep(0.5)
        self._ref_spinbox.valueChanged.connect(self._on_reference_changed)
        settings_layout.addWidget(self._ref_spinbox)

        settings_layout.addSpacing(30)

        # Settings toggle button
        self._settings_toggle = QPushButton("▼ Settings")
        self._settings_toggle.setObjectName("settingsToggle")
        self._settings_toggle.setCheckable(True)
        self._settings_toggle.setStyleSheet(TOGGLE_BUTTON_STYLE)
        self._settings_toggle.clicked.connect(self._toggle_settings)
        settings_layout.addWidget(self._settings_toggle)

        settings_layout.addSpacing(15)

        # Log button
        self._log_btn = QPushButton("Log")
        self._log_btn.setToolTip("Open measurement log window")
        self._log_btn.clicked.connect(self._open_log_window)
        settings_layout.addWidget(self._log_btn)

        settings_layout.addStretch()

        # Status label
        self._status_label = QLabel("Listening...")
        self._status_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        settings_layout.addWidget(self._status_label)

        main_layout.addWidget(settings_frame)

        # Expandable settings panel (hidden by default)
        self._settings_panel = self._create_settings_panel()
        self._settings_panel.setVisible(False)
        main_layout.addWidget(self._settings_panel)

    def _create_reed_panels(self):
        """Create reed panels based on current number of reeds."""
        # Clear all items from layout (panels and stretches)
        while self._reed_container.count():
            item = self._reed_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._reed_panels.clear()

        # Add stretch at start for centering
        self._reed_container.addStretch()

        # Create new panels
        for i in range(self._num_reeds):
            panel = ReedPanel(reed_number=i + 1)
            self._reed_panels.append(panel)
            self._reed_container.addWidget(panel)

        # Add stretch at end for centering
        self._reed_container.addStretch()

    def _create_settings_panel(self) -> QFrame:
        """Create the expandable settings panel with tabbed categories."""
        panel = QFrame()
        panel.setObjectName("settingsPanel")
        panel.setStyleSheet(SETTINGS_PANEL_STYLE)

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Tab widget for settings categories
        tabs = QTabWidget()

        # === Detection tab ===
        detection_tab = QWidget()
        detection_layout = QVBoxLayout(detection_tab)
        detection_layout.setSpacing(8)
        detection_layout.setContentsMargins(10, 10, 10, 10)

        # Checkboxes row
        checkbox_row = QHBoxLayout()
        checkbox_row.setSpacing(20)

        self._octave_filter_cb = QCheckBox("Octave Filter")
        self._octave_filter_cb.setToolTip("Restrict detection to one octave (OFF for chords)")
        self._octave_filter_cb.setChecked(False)
        self._octave_filter_cb.stateChanged.connect(self._on_octave_filter_changed)
        checkbox_row.addWidget(self._octave_filter_cb)

        self._fundamental_filter_cb = QCheckBox("Fundamental Filter")
        self._fundamental_filter_cb.setToolTip("Only detect harmonics of fundamental")
        self._fundamental_filter_cb.setChecked(False)
        self._fundamental_filter_cb.stateChanged.connect(self._on_fundamental_filter_changed)
        checkbox_row.addWidget(self._fundamental_filter_cb)

        self._downsample_cb = QCheckBox("Downsample")
        self._downsample_cb.setToolTip("Better low frequency detection")
        self._downsample_cb.setChecked(False)
        self._downsample_cb.stateChanged.connect(self._on_downsample_changed)
        checkbox_row.addWidget(self._downsample_cb)

        checkbox_row.addStretch()
        detection_layout.addLayout(checkbox_row)

        # Sliders row
        sliders_row = QHBoxLayout()
        sliders_row.setSpacing(30)

        # Sensitivity slider
        sens_layout = QHBoxLayout()
        sens_layout.addWidget(QLabel("Sensitivity:"))
        self._sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self._sensitivity_slider.setRange(5, 50)
        self._sensitivity_slider.setValue(10)
        self._sensitivity_slider.setMinimumWidth(100)
        self._sensitivity_slider.setToolTip("Detection threshold (lower = more sensitive)")
        self._sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        sens_layout.addWidget(self._sensitivity_slider)
        self._sensitivity_value = QLabel("0.10")
        sens_layout.addWidget(self._sensitivity_value)
        sliders_row.addLayout(sens_layout)

        # Reed Spread slider
        spread_layout = QHBoxLayout()
        spread_layout.addWidget(QLabel("Reed Spread:"))
        self._reed_spread_slider = QSlider(Qt.Orientation.Horizontal)
        self._reed_spread_slider.setRange(20, 100)
        self._reed_spread_slider.setValue(50)
        self._reed_spread_slider.setMinimumWidth(100)
        self._reed_spread_slider.setToolTip("Max cents to group as same note")
        self._reed_spread_slider.valueChanged.connect(self._on_reed_spread_changed)
        spread_layout.addWidget(self._reed_spread_slider)
        self._reed_spread_value = QLabel("50¢")
        spread_layout.addWidget(self._reed_spread_value)
        sliders_row.addLayout(spread_layout)

        sliders_row.addStretch()
        detection_layout.addLayout(sliders_row)

        tabs.addTab(detection_tab, "Detection")

        # === Tuning tab ===
        tuning_tab = QWidget()
        tuning_layout = QVBoxLayout(tuning_tab)
        tuning_layout.setSpacing(10)
        tuning_layout.setContentsMargins(10, 10, 10, 10)

        # Row 1: Temperament, Key, Transpose
        row1 = QHBoxLayout()
        row1.setSpacing(20)

        # Temperament
        row1.addWidget(QLabel("Temperament:"))
        self._temperament_combo = QComboBox()
        self._temperament_combo.setFixedWidth(140)
        temperament_names = [
            "Kirnberger I", "Kirnberger II", "Kirnberger III",
            "Werckmeister III", "Werckmeister IV", "Werckmeister V", "Werckmeister VI",
            "Bach-Lehman", "Equal", "Pythagorean", "Just",
            "Meantone", "Meantone 1/4", "Meantone 1/5", "Meantone 1/6",
            "Silbermann", "Salinas", "Zarlino", "Rossi", "Rossi 2",
            "Vallotti", "Young", "Kellner", "Held",
            "Neidhardt I", "Neidhardt II", "Neidhardt III",
            "Bruder 1829", "Barnes", "Prelleur", "Chaumont", "Rameau"
        ]
        self._temperament_combo.addItems(temperament_names)
        self._temperament_combo.setCurrentIndex(Temperament.EQUAL)
        self._temperament_combo.currentIndexChanged.connect(self._on_temperament_changed)
        row1.addWidget(self._temperament_combo)

        row1.addSpacing(20)

        # Key
        row1.addWidget(QLabel("Key:"))
        self._key_combo = QComboBox()
        self._key_combo.setFixedWidth(60)
        self._key_combo.addItems(NOTE_NAMES)
        self._key_combo.setCurrentIndex(0)
        self._key_combo.currentIndexChanged.connect(self._on_key_changed)
        row1.addWidget(self._key_combo)

        row1.addSpacing(20)

        # Transpose
        row1.addWidget(QLabel("Transpose:"))
        self._transpose_spin = QDoubleSpinBox()
        self._transpose_spin.setFixedWidth(70)
        self._transpose_spin.setRange(-6, 6)
        self._transpose_spin.setValue(0)
        self._transpose_spin.setDecimals(0)
        self._transpose_spin.setSuffix(" st")
        self._transpose_spin.setToolTip("Transpose display for transposing instruments")
        row1.addWidget(self._transpose_spin)

        row1.addStretch()
        tuning_layout.addLayout(row1)

        # Row 2: Tremolo Profile
        row2 = QHBoxLayout()
        row2.setSpacing(10)

        row2.addWidget(QLabel("Tremolo Profile:"))
        self._profile_combo = QComboBox()
        self._profile_combo.setFixedWidth(180)
        self._profile_combo.addItem("None")
        self._profile_combo.setToolTip("Select a tremolo tuning profile")
        self._profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        row2.addWidget(self._profile_combo)

        self._load_profile_btn = QPushButton("Load...")
        self._load_profile_btn.setToolTip("Load a tremolo profile from CSV/TSV file")
        self._load_profile_btn.clicked.connect(self._on_load_profile)
        row2.addWidget(self._load_profile_btn)

        self._clear_profile_btn = QPushButton("Clear")
        self._clear_profile_btn.setToolTip("Remove all loaded profiles")
        self._clear_profile_btn.clicked.connect(self._on_clear_profiles)
        row2.addWidget(self._clear_profile_btn)

        row2.addStretch()
        tuning_layout.addLayout(row2)

        tabs.addTab(tuning_tab, "Tuning")

        # === Display tab ===
        display_tab = QWidget()
        display_layout = QHBoxLayout(display_tab)
        display_layout.setSpacing(20)
        display_layout.setContentsMargins(10, 10, 10, 10)

        self._lock_display_cb = QCheckBox("Lock Display")
        self._lock_display_cb.setToolTip("Freeze display values")
        self._lock_display_cb.setChecked(False)
        self._lock_display_cb.stateChanged.connect(self._on_lock_display_changed)
        display_layout.addWidget(self._lock_display_cb)

        self._hold_mode_cb = QCheckBox("Hold Mode")
        self._hold_mode_cb.setToolTip("Freeze display when note ends, showing best measurement")
        self._hold_mode_cb.setChecked(False)
        self._hold_mode_cb.stateChanged.connect(self._on_hold_mode_changed)
        display_layout.addWidget(self._hold_mode_cb)

        self._zoom_spectrum_cb = QCheckBox("Zoom Spectrum")
        self._zoom_spectrum_cb.setToolTip("Zoom spectrum to detected note")
        self._zoom_spectrum_cb.setChecked(True)
        self._zoom_spectrum_cb.stateChanged.connect(self._on_zoom_spectrum_changed)
        display_layout.addWidget(self._zoom_spectrum_cb)

        display_layout.addStretch()

        tabs.addTab(display_tab, "Display")

        # === Audio tab ===
        audio_tab = QWidget()
        audio_layout = QHBoxLayout(audio_tab)
        audio_layout.setSpacing(15)
        audio_layout.setContentsMargins(10, 10, 10, 10)

        audio_layout.addWidget(QLabel("Input Device:"))
        self._input_combo = QComboBox()
        self._input_combo.setMinimumWidth(200)
        self._populate_audio_devices()
        self._input_combo.currentIndexChanged.connect(self._on_input_device_changed)
        audio_layout.addWidget(self._input_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._populate_audio_devices)
        audio_layout.addWidget(refresh_btn)

        audio_layout.addStretch()

        tabs.addTab(audio_tab, "Audio")

        layout.addWidget(tabs, 1)

        # Reset All button outside tabs
        reset_btn = QPushButton("Reset All")
        reset_btn.setToolTip("Reset all settings to their default values")
        reset_btn.clicked.connect(self._reset_to_defaults)
        layout.addWidget(reset_btn)

        return panel

    def _populate_audio_devices(self):
        """Populate the audio input device combo box."""
        self._input_combo.clear()
        self._input_combo.addItem("Default", None)

        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    name = device['name']
                    self._input_combo.addItem(name, i)
        except Exception:
            pass

    def _toggle_settings(self, checked: bool):
        """Toggle the expanded settings panel visibility."""
        self._settings_expanded = checked
        self._settings_panel.setVisible(checked)
        self._settings_toggle.setText("▲ Settings" if checked else "▼ Settings")

        # Resize window to accommodate settings panel (tabs are more compact)
        if checked:
            self.setMinimumHeight(680)
            self.resize(self.width(), 680)
        else:
            self.setMinimumHeight(600)
            self.resize(self.width(), 600)

    def _on_octave_filter_changed(self, state):
        """Handle octave filter checkbox change."""
        self._detector.set_octave_filter(self._octave_filter_cb.isChecked())

    def _on_fundamental_filter_changed(self, state):
        """Handle fundamental filter checkbox change."""
        self._detector.set_fundamental_filter(self._fundamental_filter_cb.isChecked())

    def _on_downsample_changed(self, state):
        """Handle downsample checkbox change."""
        self._detector.set_downsample(self._downsample_cb.isChecked())

    def _on_sensitivity_changed(self, value: int):
        """Handle sensitivity slider change."""
        threshold = value / 100.0
        self._detector.set_sensitivity(threshold)
        self._sensitivity_value.setText(f"{threshold:.2f}")

    def _on_reed_spread_changed(self, value: int):
        """Handle reed spread slider change."""
        self._detector.set_reed_spread(float(value))
        self._reed_spread_value.setText(f"{value}¢")

    def _on_temperament_changed(self, index: int):
        """Handle temperament combo change."""
        self._detector.set_temperament(Temperament(index))

    def _on_key_changed(self, index: int):
        """Handle key combo change."""
        self._detector.set_key(index)

    def _on_profile_changed(self, index: int):
        """Handle tremolo profile selection change."""
        if index == 0:
            # "None" selected
            self._tremolo_profile = None
            self._detector.set_tremolo_profile(None)
        else:
            profile_name = self._profile_combo.currentText()
            if profile_name in self._loaded_profiles:
                self._tremolo_profile = self._loaded_profiles[profile_name]
                self._detector.set_tremolo_profile(self._tremolo_profile)

    def _on_load_profile(self):
        """Handle load profile button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Tremolo Profile",
            "",
            "Profile Files (*.csv *.tsv *.txt);;All Files (*.*)",
        )
        if not file_path:
            return

        try:
            profile = load_profile(file_path)

            # Add to loaded profiles
            self._loaded_profiles[profile.name] = profile

            # Add to combo if not already present
            existing_items = [self._profile_combo.itemText(i) for i in range(self._profile_combo.count())]
            if profile.name not in existing_items:
                self._profile_combo.addItem(profile.name)

            # Select the newly loaded profile
            index = self._profile_combo.findText(profile.name)
            if index >= 0:
                self._profile_combo.setCurrentIndex(index)

        except FileNotFoundError as e:
            QMessageBox.warning(self, "Load Error", f"File not found: {e}")
        except ValueError as e:
            QMessageBox.warning(self, "Load Error", f"Invalid profile format: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load profile: {e}")

    def _on_clear_profiles(self):
        """Handle clear profiles button click."""
        # Reset to "None"
        self._profile_combo.setCurrentIndex(0)

        # Remove all profiles except "None"
        while self._profile_combo.count() > 1:
            self._profile_combo.removeItem(1)

        self._loaded_profiles.clear()
        self._tremolo_profile = None
        self._detector.set_tremolo_profile(None)

    def _on_lock_display_changed(self, state):
        """Handle lock display checkbox change."""
        self._lock_display = self._lock_display_cb.isChecked()

    def _on_hold_mode_changed(self, state):
        """Handle hold mode checkbox change."""
        self._hold_mode = self._hold_mode_cb.isChecked()
        # Reset hold state when toggling
        self._hold_state = "WAITING"
        self._held_result = None
        self._held_magnitude = 0.0
        if not self._hold_mode:
            self._status_label.setText("Listening...")

    def _on_zoom_spectrum_changed(self, state):
        """Handle zoom spectrum checkbox change."""
        # Use isChecked() directly for reliable state detection
        self._zoom_spectrum = self._zoom_spectrum_cb.isChecked()
        # Apply zoom change immediately using last known frequency
        center_freq = None
        if self._last_result and self._last_result.valid and self._last_result.reeds:
            center_freq = self._last_result.reeds[0].frequency
        self._spectrum_view.set_zoom(center_freq, self._zoom_spectrum)

    def _open_log_window(self):
        """Open or show the measurement log window."""
        if self._log_window is None:
            self._log_window = MeasurementLogWindow(self)
            self._log_window.request_hold_mode.connect(self._on_log_request_hold_mode)
            self._log_window.request_timed_recording.connect(self._on_log_request_timed)
        self._log_window.show()
        self._log_window.raise_()

    def _on_log_request_hold_mode(self, enabled: bool):
        """Handle log window request to enable/disable hold mode recording."""
        self._log_recording_hold = enabled
        if enabled:
            # Auto-enable hold mode in main window
            self._hold_mode_cb.setChecked(True)

    def _on_log_request_timed(self, enabled: bool, interval: float):
        """Handle log window request for timed recording."""
        self._log_recording_timed = enabled
        if enabled:
            self._log_timer.start(int(interval * 1000))
        else:
            self._log_timer.stop()

    def _on_log_timer(self):
        """Timer callback for timed recording mode."""
        if self._log_window and self._last_result and self._last_result.valid:
            self._log_window.add_entry(self._last_result)

    def _on_input_device_changed(self, index: int):
        """Handle input device combo change."""
        device_id = self._input_combo.itemData(index)
        if device_id != self._input_device:
            self._input_device = device_id
            self._restart_audio()

    def _restart_audio(self):
        """Restart audio stream with new device."""
        self._stop_audio()
        self._start_audio()

    def _apply_style(self):
        """Apply styling to the window."""
        self.setStyleSheet(MAIN_WINDOW_STYLE)

    def _start_audio(self):
        """Start audio capture."""
        try:
            self._stream = sd.InputStream(
                device=self._input_device,
                samplerate=self._sample_rate,
                blocksize=self._buffer_size,
                channels=1,
                dtype=np.float32,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._status_label.setText("Listening...")
        except Exception as e:
            self._status_label.setText(f"Audio error: {e}")

    def _stop_audio(self):
        """Stop audio capture."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _audio_callback(self, indata, frames, time, status):
        """Audio callback - process incoming audio."""
        if status:
            pass  # Ignore status messages

        # Get audio data and apply fixed gain boost for typical microphone levels
        audio = indata[:, 0].copy() * 10.0  # Fixed 10x gain

        self._audio_buffer = audio

        # Process with detector
        self._last_result = self._detector.process(self._audio_buffer)

    def _update_display(self):
        """Update the display with the latest detection result."""
        # Skip update if display is locked
        if self._lock_display:
            return

        result = self._last_result

        # Handle hold mode state machine
        if self._hold_mode:
            display_result = self._handle_hold_mode(result)
            if display_result is None:
                # Update spectrum with live data while waiting
                if result and result.spectrum_data:
                    freqs, mags = result.spectrum_data
                    self._spectrum_view.set_spectrum(freqs, mags)
                return
            # Use the result from hold mode (could be held result)
            result = display_result
        else:
            # Normal mode - handle invalid result
            if result is None or not result.valid:
                self._note_display.set_inactive()
                for panel in self._reed_panels:
                    panel.set_inactive()
                self._multi_meter.set_all_inactive()
                if result and result.spectrum_data:
                    freqs, mags = result.spectrum_data
                    self._spectrum_view.set_spectrum(freqs, mags)
                    self._spectrum_view.set_peaks([])
                return

        # Display the result (either live or held) - includes spectrum
        self._display_result(result)

    def _handle_hold_mode(self, result: AccordionResult | None) -> AccordionResult | None:
        """Handle hold mode state machine. Returns result to display, or None to skip display."""
        is_valid = result is not None and result.valid

        if self._hold_state == "WAITING":
            if is_valid:
                # Note started - begin tracking
                self._hold_state = "TRACKING"
                self._held_result = result
                self._held_magnitude = self._calculate_magnitude(result)
                self._status_label.setText("Tracking...")
                return result
            else:
                # Still waiting for a note
                self._note_display.set_inactive()
                for panel in self._reed_panels:
                    panel.set_inactive()
                self._multi_meter.set_all_inactive()
                if result and result.spectrum_data:
                    self._spectrum_view.set_peaks([])
                return None

        elif self._hold_state == "TRACKING":
            if is_valid:
                # Note still playing - update best if stronger
                current_mag = self._calculate_magnitude(result)
                if current_mag > self._held_magnitude:
                    self._held_result = result
                    self._held_magnitude = current_mag
                # Display current result (real-time while playing)
                return result
            else:
                # Note ended - freeze on best result
                self._hold_state = "HOLDING"
                self._status_label.setText("Held")
                # Add to log if hold mode recording is active
                if self._log_recording_hold and self._log_window and self._held_result:
                    self._log_window.add_entry(self._held_result)
                return self._held_result

        elif self._hold_state == "HOLDING":
            if is_valid:
                # New note started - reset and begin tracking
                self._hold_state = "TRACKING"
                self._held_result = result
                self._held_magnitude = self._calculate_magnitude(result)
                self._status_label.setText("Tracking...")
                return result
            else:
                # Keep showing held result
                return self._held_result

        return None

    def _calculate_magnitude(self, result: AccordionResult) -> float:
        """Calculate total magnitude from result's reeds."""
        if not result or not result.reeds:
            return 0.0
        return sum(r.magnitude for r in result.reeds)

    def _display_result(self, result: AccordionResult):
        """Display the given result on all UI components."""
        # Update note display
        self._note_display.set_note(
            result.note_name,
            result.octave,
            result.ref_frequency,
        )

        # Update spectrum view with data, peaks and zoom
        if result.spectrum_data:
            freqs, mags = result.spectrum_data
            self._spectrum_view.set_spectrum(freqs, mags)
            self._spectrum_view.set_peaks([r.frequency for r in result.reeds])
            # Zoom to detected note if enabled
            if result.reeds:
                self._spectrum_view.set_zoom(result.reeds[0].frequency, self._zoom_spectrum)
            else:
                self._spectrum_view.set_zoom(None, self._zoom_spectrum)

        # Update reed panels with beat frequencies on detuned reeds
        # beat_frequencies contains: [|f1-f2|, |f2-f3|, ...]
        # Display logic:
        #   2 reeds: beat on Reed 2 (detuned reed beats with Reed 1)
        #   3 reeds: beat on Reed 1 and Reed 3 (outer reeds beat with center Reed 2)
        #   4 reeds: beat on Reed 1 and Reed 4 (outer reeds beat with inner reeds)
        num_reeds = len(result.reeds)
        for i, panel in enumerate(self._reed_panels):
            if i < num_reeds:
                reed = result.reeds[i]
                beat_freq = None

                if num_reeds == 2:
                    # Show beat on Reed 2 (index 1)
                    if i == 1 and len(result.beat_frequencies) > 0:
                        beat_freq = result.beat_frequencies[0]
                elif num_reeds == 3:
                    # Show beat on Reed 1 and Reed 3 (indices 0 and 2)
                    if i == 0 and len(result.beat_frequencies) > 0:
                        beat_freq = result.beat_frequencies[0]  # |f1-f2|
                    elif i == 2 and len(result.beat_frequencies) > 1:
                        beat_freq = result.beat_frequencies[1]  # |f2-f3|
                elif num_reeds >= 4:
                    # Show beat on outer reeds (first and last)
                    if i == 0 and len(result.beat_frequencies) > 0:
                        beat_freq = result.beat_frequencies[0]
                    elif i == num_reeds - 1 and len(result.beat_frequencies) > i - 1:
                        beat_freq = result.beat_frequencies[-1]

                panel.set_data(reed.frequency, reed.cents, beat_freq, reed.target_cents)
                # Update unified meter with this reed's cents (or target_cents if available)
                display_cents = reed.target_cents if reed.target_cents is not None else reed.cents
                self._multi_meter.set_reed_data(i, display_cents)
            else:
                panel.set_inactive()
                self._multi_meter.set_reed_data(i, None)

    def _on_reeds_changed(self, index: int):
        """Handle number of reeds change."""
        self._num_reeds = index + 1  # 0=1, 1=2, 2=3, 3=4
        self._detector.set_max_reeds(self._num_reeds)
        self._create_reed_panels()
        self._multi_meter.set_num_reeds(self._num_reeds)

    def _on_reference_changed(self, value: float):
        """Handle reference frequency change."""
        self._reference = value
        self._detector.set_reference(self._reference)
        self._note_display.set_reference(self._reference)

    def closeEvent(self, event):
        """Handle window close."""
        self._save_settings()
        self._stop_audio()
        self._timer.stop()
        self._log_timer.stop()
        if self._log_window:
            self._log_window.close()
        event.accept()

    def _load_settings(self):
        """Load saved settings from QSettings."""
        settings = QSettings("accordion-tuner", "AccordionTuner")

        # Number of reeds
        num_reeds = settings.value("num_reeds", self.DEFAULTS['num_reeds'], type=int)
        self._reeds_combo.setCurrentIndex(num_reeds - 1)

        # Reference frequency
        reference = settings.value("reference", self.DEFAULTS['reference'], type=float)
        self._ref_spinbox.setValue(reference)

        # Detection settings
        octave_filter = settings.value("octave_filter", self.DEFAULTS['octave_filter'], type=bool)
        self._octave_filter_cb.setChecked(octave_filter)

        fundamental_filter = settings.value("fundamental_filter", self.DEFAULTS['fundamental_filter'], type=bool)
        self._fundamental_filter_cb.setChecked(fundamental_filter)

        downsample = settings.value("downsample", self.DEFAULTS['downsample'], type=bool)
        self._downsample_cb.setChecked(downsample)

        sensitivity = settings.value("sensitivity", self.DEFAULTS['sensitivity'], type=int)
        self._sensitivity_slider.setValue(sensitivity)

        reed_spread = settings.value("reed_spread", self.DEFAULTS['reed_spread'], type=int)
        self._reed_spread_slider.setValue(reed_spread)

        # Tuning settings
        temperament = settings.value("temperament", self.DEFAULTS['temperament'], type=int)
        self._temperament_combo.setCurrentIndex(temperament)

        key = settings.value("key", self.DEFAULTS['key'], type=int)
        self._key_combo.setCurrentIndex(key)

        transpose = settings.value("transpose", self.DEFAULTS['transpose'], type=int)
        self._transpose_spin.setValue(transpose)

        # Display settings
        zoom_spectrum = settings.value("zoom_spectrum", self.DEFAULTS['zoom_spectrum'], type=bool)
        self._zoom_spectrum_cb.setChecked(zoom_spectrum)

        hold_mode = settings.value("hold_mode", self.DEFAULTS['hold_mode'], type=bool)
        self._hold_mode_cb.setChecked(hold_mode)

        settings_expanded = settings.value("settings_expanded", self.DEFAULTS['settings_expanded'], type=bool)
        if settings_expanded:
            self._settings_toggle.setChecked(True)
            self._toggle_settings(True)

        # Input device (by name)
        device_name = settings.value("input_device_name", "", type=str)
        if device_name:
            for i in range(self._input_combo.count()):
                if self._input_combo.itemText(i) == device_name:
                    self._input_combo.setCurrentIndex(i)
                    break

        # Tremolo profile
        profile_path = settings.value("tremolo_profile_path", self.DEFAULTS['tremolo_profile_path'], type=str)
        if profile_path:
            try:
                profile = load_profile(profile_path)
                self._loaded_profiles[profile.name] = profile
                self._profile_combo.addItem(profile.name)
                self._profile_combo.setCurrentIndex(self._profile_combo.findText(profile.name))
            except Exception:
                pass  # Silently ignore if profile can't be loaded

        # Window geometry
        geometry = settings.value("window_geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def _save_settings(self):
        """Save current settings to QSettings."""
        settings = QSettings("accordion-tuner", "AccordionTuner")

        # Number of reeds
        settings.setValue("num_reeds", self._reeds_combo.currentIndex() + 1)

        # Reference frequency
        settings.setValue("reference", self._ref_spinbox.value())

        # Detection settings
        settings.setValue("octave_filter", self._octave_filter_cb.isChecked())
        settings.setValue("fundamental_filter", self._fundamental_filter_cb.isChecked())
        settings.setValue("downsample", self._downsample_cb.isChecked())
        settings.setValue("sensitivity", self._sensitivity_slider.value())
        settings.setValue("reed_spread", self._reed_spread_slider.value())

        # Tuning settings
        settings.setValue("temperament", self._temperament_combo.currentIndex())
        settings.setValue("key", self._key_combo.currentIndex())
        settings.setValue("transpose", int(self._transpose_spin.value()))

        # Display settings
        settings.setValue("zoom_spectrum", self._zoom_spectrum_cb.isChecked())
        settings.setValue("hold_mode", self._hold_mode_cb.isChecked())
        settings.setValue("settings_expanded", self._settings_expanded)

        # Input device (by name, not ID which may change)
        settings.setValue("input_device_name", self._input_combo.currentText())

        # Tremolo profile (save path of currently selected profile)
        if self._tremolo_profile is not None:
            settings.setValue("tremolo_profile_path", self._tremolo_profile.path)
        else:
            settings.setValue("tremolo_profile_path", "")

        # Window geometry
        settings.setValue("window_geometry", self.saveGeometry())

    def _reset_to_defaults(self):
        """Reset all settings to their default values."""
        reply = QMessageBox.question(
            self,
            "Reset to Defaults",
            "Are you sure you want to reset all settings to their default values?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Clear stored settings
        settings = QSettings("accordion-tuner", "AccordionTuner")
        settings.clear()

        # Reset all widgets to defaults (this triggers their change handlers)
        self._reeds_combo.setCurrentIndex(self.DEFAULTS['num_reeds'] - 1)
        self._ref_spinbox.setValue(self.DEFAULTS['reference'])
        self._octave_filter_cb.setChecked(self.DEFAULTS['octave_filter'])
        self._fundamental_filter_cb.setChecked(self.DEFAULTS['fundamental_filter'])
        self._downsample_cb.setChecked(self.DEFAULTS['downsample'])
        self._sensitivity_slider.setValue(self.DEFAULTS['sensitivity'])
        self._reed_spread_slider.setValue(self.DEFAULTS['reed_spread'])
        self._temperament_combo.setCurrentIndex(self.DEFAULTS['temperament'])
        self._key_combo.setCurrentIndex(self.DEFAULTS['key'])
        self._transpose_spin.setValue(self.DEFAULTS['transpose'])
        self._zoom_spectrum_cb.setChecked(self.DEFAULTS['zoom_spectrum'])
        self._hold_mode_cb.setChecked(self.DEFAULTS['hold_mode'])
        self._input_combo.setCurrentIndex(0)  # Default device

        # Clear tremolo profiles
        self._on_clear_profiles()

        # Collapse settings panel if expanded
        if self._settings_expanded:
            self._settings_toggle.setChecked(False)
            self._toggle_settings(False)


def main():
    """Main entry point for the accordion tuner GUI."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for consistent cross-platform look

    window = AccordionWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
