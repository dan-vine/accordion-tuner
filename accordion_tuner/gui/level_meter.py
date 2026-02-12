"""
Audio level meter widget - vertical bar showing input signal strength.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .styles import (
    ACCENT_GREEN,
    BORDER_COLOR,
    ERROR_RED,
    METER_BACKGROUND,
    PANEL_BACKGROUND,
    TEXT_SECONDARY,
    WARNING_ORANGE,
)


class LevelMeter(QWidget):
    """
    Vertical audio level meter showing input signal strength.

    Displays:
    - A vertical bar that fills from bottom to top based on RMS level
    - Color gradient: green (normal) -> orange (medium) -> red (loud/clipping)
    - Peak hold indicator that slowly decays
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rms = 0.0  # 0.0 to 1.0
        self._peak = 0.0  # 0.0 to 1.0
        self._peak_hold = 0.0  # Peak hold value (decays over time)
        self._active = False

        # Thresholds for color zones (as fraction of full scale)
        self._green_threshold = 0.5  # Below 50% = green
        self._orange_threshold = 0.8  # 50-80% = orange, above 80% = red

        # Peak hold decay rate (per update, ~20 Hz)
        self._peak_decay = 0.02

        self.setMinimumSize(24, 80)
        self.setMaximumWidth(30)

    def set_level(self, rms: float, peak: float):
        """
        Set the current level values.

        Args:
            rms: RMS level (0.0 to 1.0 scale)
            peak: Peak level (0.0 to 1.0 scale)
        """
        self._rms = max(0.0, min(1.0, rms))
        self._peak = max(0.0, min(1.0, peak))

        # Update peak hold (only goes up instantly, decays slowly)
        if self._peak > self._peak_hold:
            self._peak_hold = self._peak
        else:
            self._peak_hold = max(0.0, self._peak_hold - self._peak_decay)

        self._active = True
        self.update()

    def set_inactive(self):
        """Set meter to inactive state (no audio)."""
        self._active = False
        self._rms = 0.0
        self._peak = 0.0
        self._peak_hold = 0.0
        self.update()

    def paintEvent(self, event):
        """Paint the level meter."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        margin = 2

        bar_width = width - 2 * margin
        bar_height = height - 2 * margin
        bar_x = margin
        bar_y = margin

        # Draw background with border
        painter.setPen(QPen(QColor(BORDER_COLOR), 1))
        painter.setBrush(QBrush(QColor(PANEL_BACKGROUND)))
        painter.drawRoundedRect(bar_x, bar_y, bar_width, bar_height, 3, 3)

        # Inner area for the meter bar
        inner_margin = 2
        inner_x = bar_x + inner_margin
        inner_y = bar_y + inner_margin
        inner_width = bar_width - 2 * inner_margin
        inner_height = bar_height - 2 * inner_margin

        # Draw meter background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(METER_BACKGROUND)))
        painter.drawRect(int(inner_x), int(inner_y), int(inner_width), int(inner_height))

        if self._active and self._rms > 0.001:
            # Calculate filled height based on RMS level
            fill_height = int(inner_height * self._rms)
            fill_y = inner_y + inner_height - fill_height

            # Create gradient for the filled portion
            gradient = QLinearGradient(0, inner_y + inner_height, 0, inner_y)

            # Green at bottom, orange in middle, red at top
            gradient.setColorAt(0.0, QColor(ACCENT_GREEN))
            gradient.setColorAt(self._green_threshold, QColor(ACCENT_GREEN))
            gradient.setColorAt(self._green_threshold + 0.01, QColor(WARNING_ORANGE))
            gradient.setColorAt(self._orange_threshold, QColor(WARNING_ORANGE))
            gradient.setColorAt(self._orange_threshold + 0.01, QColor(ERROR_RED))
            gradient.setColorAt(1.0, QColor(ERROR_RED))

            painter.setBrush(QBrush(gradient))
            painter.drawRect(int(inner_x), int(fill_y), int(inner_width), int(fill_height))

            # Draw peak hold indicator
            if self._peak_hold > 0.01:
                peak_y = inner_y + inner_height - int(inner_height * self._peak_hold)
                peak_color = self._get_level_color(self._peak_hold)
                painter.setPen(QPen(peak_color, 2))
                painter.drawLine(int(inner_x), int(peak_y), int(inner_x + inner_width), int(peak_y))
        elif not self._active:
            # Draw inactive state - gray bar
            painter.setBrush(QBrush(QColor(TEXT_SECONDARY).darker(200)))
            painter.drawRect(int(inner_x), int(inner_y + inner_height - 3), int(inner_width), 3)

        # Draw segment lines for visual reference
        painter.setPen(QPen(QColor(BORDER_COLOR), 1))
        num_segments = 8
        for i in range(1, num_segments):
            seg_y = inner_y + int(inner_height * i / num_segments)
            painter.drawLine(int(inner_x), int(seg_y), int(inner_x + inner_width), int(seg_y))

    def _get_level_color(self, level: float) -> QColor:
        """Get color based on level value."""
        if level <= self._green_threshold:
            return QColor(ACCENT_GREEN)
        elif level <= self._orange_threshold:
            return QColor(WARNING_ORANGE)
        else:
            return QColor(ERROR_RED)
