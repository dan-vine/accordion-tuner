"""
Qt stylesheets for modern dark theme.

Color scheme:
- Background: #1e1e1e (dark gray)
- Panel background: #2d2d2d
- Text: #ffffff
- Accent (in tune): #4caf50 (green)
- Warning (slightly off): #ff9800 (orange)
- Error (very off): #f44336 (red)
- Meter background: #3d3d3d
"""

# Colors
BACKGROUND = "#1e1e1e"
PANEL_BACKGROUND = "#2d2d2d"
TEXT_COLOR = "#ffffff"
TEXT_SECONDARY = "#b0b0b0"
ACCENT_GREEN = "#4caf50"
WARNING_ORANGE = "#ff9800"
ERROR_RED = "#f44336"
METER_BACKGROUND = "#3d3d3d"
BORDER_COLOR = "#404040"

# Main window stylesheet
MAIN_WINDOW_STYLE = f"""
QMainWindow {{
    background-color: {BACKGROUND};
}}
QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT_COLOR};
    font-family: "Segoe UI", "SF Pro Display", "Helvetica Neue", sans-serif;
}}
QLabel {{
    color: {TEXT_COLOR};
}}
QComboBox {{
    background-color: {PANEL_BACKGROUND};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    padding: 5px 10px;
    min-width: 80px;
    color: {TEXT_COLOR};
}}
QComboBox:hover {{
    border-color: {ACCENT_GREEN};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid {TEXT_COLOR};
    margin-right: 5px;
}}
QComboBox QAbstractItemView {{
    background-color: {PANEL_BACKGROUND};
    border: 1px solid {BORDER_COLOR};
    selection-background-color: {ACCENT_GREEN};
    color: {TEXT_COLOR};
}}
QSpinBox, QDoubleSpinBox {{
    background-color: {PANEL_BACKGROUND};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    padding: 5px 10px;
    color: {TEXT_COLOR};
}}
QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {ACCENT_GREEN};
}}
QPushButton {{
    background-color: {PANEL_BACKGROUND};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    padding: 8px 16px;
    color: {TEXT_COLOR};
}}
QPushButton:hover {{
    background-color: {METER_BACKGROUND};
    border-color: {ACCENT_GREEN};
}}
QPushButton:pressed {{
    background-color: {ACCENT_GREEN};
}}
QGroupBox {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {TEXT_SECONDARY};
}}
"""

# Reed panel stylesheet
REED_PANEL_STYLE = f"""
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
"""

# Note display stylesheet
NOTE_DISPLAY_STYLE = f"""
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
}}
QLabel#refFrequency {{
    font-size: 14px;
    color: {TEXT_SECONDARY};
}}
"""

# Spectrum view stylesheet
SPECTRUM_VIEW_STYLE = f"""
QFrame#spectrumView {{
    background-color: {PANEL_BACKGROUND};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
}}
"""


def get_cents_color(cents: float) -> str:
    """Get color based on cents deviation."""
    abs_cents = abs(cents)
    if abs_cents <= 5:
        return ACCENT_GREEN
    elif abs_cents <= 15:
        return WARNING_ORANGE
    else:
        return ERROR_RED


def get_tuning_gradient(cents: float) -> str:
    """Get gradient for tuning meter based on cents position."""
    color = get_cents_color(cents)
    return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {METER_BACKGROUND}, stop:0.5 {color}, stop:1 {METER_BACKGROUND})"


# Settings panel stylesheet
SETTINGS_PANEL_STYLE = f"""
QFrame#settingsPanel {{
    background-color: {PANEL_BACKGROUND};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
}}
QFrame#settingsPanel QCheckBox {{
    color: {TEXT_COLOR};
    spacing: 8px;
    background-color: transparent;
}}
QFrame#settingsPanel QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid {BORDER_COLOR};
    background-color: {METER_BACKGROUND};
}}
QFrame#settingsPanel QCheckBox::indicator:checked {{
    background-color: {ACCENT_GREEN};
    border-color: {ACCENT_GREEN};
}}
QFrame#settingsPanel QCheckBox::indicator:hover {{
    border-color: {ACCENT_GREEN};
}}
QFrame#settingsPanel QSlider::groove:horizontal {{
    border: 1px solid {BORDER_COLOR};
    height: 6px;
    background: {METER_BACKGROUND};
    border-radius: 3px;
}}
QFrame#settingsPanel QSlider::handle:horizontal {{
    background: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QFrame#settingsPanel QSlider::handle:horizontal:hover {{
    background: {ACCENT_GREEN};
    border-color: {ACCENT_GREEN};
}}
QFrame#settingsPanel QSlider::sub-page:horizontal {{
    background: {ACCENT_GREEN};
    border-radius: 3px;
}}
QFrame#settingsPanel QLabel {{
    background-color: transparent;
}}
QFrame#settingsPanel QComboBox {{
    min-width: 140px;
    background-color: {PANEL_BACKGROUND};
}}
QFrame#settingsPanel QSlider {{
    background-color: transparent;
    min-width: 80px;
}}
QFrame#settingsPanel QDoubleSpinBox {{
    background-color: {PANEL_BACKGROUND};
}}
"""

# Toggle button style for settings expansion
TOGGLE_BUTTON_STYLE = f"""
QPushButton#settingsToggle {{
    background-color: transparent;
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    padding: 5px 12px;
    color: {TEXT_SECONDARY};
    font-size: 12px;
}}
QPushButton#settingsToggle:hover {{
    background-color: {METER_BACKGROUND};
    border-color: {ACCENT_GREEN};
    color: {TEXT_COLOR};
}}
QPushButton#settingsToggle:checked {{
    background-color: {METER_BACKGROUND};
    color: {TEXT_COLOR};
}}
"""
