"""
Tuning meter widget - horizontal bar showing cents deviation.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .styles import (
    ACCENT_GREEN,
    BORDER_COLOR,
    ERROR_RED,
    METER_BACKGROUND,
    PANEL_BACKGROUND,
    TEXT_COLOR,
    TEXT_SECONDARY,
    WARNING_ORANGE,
)


class TuningMeter(QWidget):
    """
    Horizontal tuning meter showing cents deviation from -50 to +50.

    The meter displays:
    - A center line at 0 cents
    - A moving indicator showing current deviation
    - Color coding: green (in tune), orange (slightly off), red (very off)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cents = 0.0
        self._active = False

        # Appearance settings
        self._min_cents = -50.0
        self._max_cents = 50.0
        self._green_zone = 5.0   # ±5 cents = green
        self._orange_zone = 15.0  # ±15 cents = orange

        self.setMinimumSize(200, 30)
        self.setMaximumHeight(40)

    def set_cents(self, cents: float):
        """Set the current cents deviation."""
        self._cents = max(self._min_cents, min(self._max_cents, cents))
        self._active = True
        self.update()

    def set_inactive(self):
        """Set meter to inactive state (no signal)."""
        self._active = False
        self.update()

    def paintEvent(self, event):
        """Paint the tuning meter."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        margin = 4
        bar_height = height - 2 * margin
        bar_width = width - 2 * margin

        # Draw background
        painter.setPen(QPen(QColor(BORDER_COLOR), 1))
        painter.setBrush(QBrush(QColor(METER_BACKGROUND)))
        painter.drawRoundedRect(margin, margin, bar_width, bar_height, 4, 4)

        # Draw colored zones
        zone_height = bar_height - 4
        zone_y = margin + 2
        zone_width = bar_width - 4
        zone_x = margin + 2

        # Green zone (center)
        green_start = self._cents_to_x(0 - self._green_zone, zone_x, zone_width)
        green_end = self._cents_to_x(0 + self._green_zone, zone_x, zone_width)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(ACCENT_GREEN).darker(200)))
        painter.drawRect(int(green_start), int(zone_y), int(green_end - green_start), int(zone_height))

        # Orange zones
        painter.setBrush(QBrush(QColor(WARNING_ORANGE).darker(200)))
        # Left orange
        orange_left_start = self._cents_to_x(0 - self._orange_zone, zone_x, zone_width)
        painter.drawRect(int(orange_left_start), int(zone_y), int(green_start - orange_left_start), int(zone_height))
        # Right orange
        orange_right_end = self._cents_to_x(0 + self._orange_zone, zone_x, zone_width)
        painter.drawRect(int(green_end), int(zone_y), int(orange_right_end - green_end), int(zone_height))

        # Red zones (outer)
        painter.setBrush(QBrush(QColor(ERROR_RED).darker(200)))
        # Left red
        painter.drawRect(int(zone_x), int(zone_y), int(orange_left_start - zone_x), int(zone_height))
        # Right red
        painter.drawRect(int(orange_right_end), int(zone_y), int(zone_x + zone_width - orange_right_end), int(zone_height))

        # Draw center line
        center_x = self._cents_to_x(0, margin, bar_width)
        painter.setPen(QPen(QColor(TEXT_COLOR), 2))
        painter.drawLine(int(center_x), int(margin + 2), int(center_x), int(margin + bar_height - 2))

        # Draw tick marks at ±10, ±20, ±30, ±40 cents
        painter.setPen(QPen(QColor(TEXT_COLOR).darker(150), 1))
        for tick in [-40, -30, -20, -10, 10, 20, 30, 40]:
            tick_x = self._cents_to_x(tick, margin, bar_width)
            painter.drawLine(int(tick_x), int(margin + bar_height - 6), int(tick_x), int(margin + bar_height - 2))

        # Draw indicator if active
        if self._active:
            indicator_x = self._cents_to_x(self._cents, margin, bar_width)
            indicator_color = self._get_indicator_color()

            # Draw indicator triangle/pointer
            painter.setBrush(QBrush(indicator_color))
            painter.setPen(QPen(QColor(TEXT_COLOR), 1))

            # Triangle pointing down
            triangle_size = 8
            points = [
                (indicator_x - triangle_size, margin - 2),
                (indicator_x + triangle_size, margin - 2),
                (indicator_x, margin + 6),
            ]
            from PySide6.QtCore import QPointF
            from PySide6.QtGui import QPolygonF
            polygon = QPolygonF([QPointF(x, y) for x, y in points])
            painter.drawPolygon(polygon)

            # Vertical line
            painter.setPen(QPen(indicator_color, 3))
            painter.drawLine(int(indicator_x), int(margin + 4), int(indicator_x), int(margin + bar_height - 2))

    def _cents_to_x(self, cents: float, start: float, width: float) -> float:
        """Convert cents value to x coordinate."""
        normalized = (cents - self._min_cents) / (self._max_cents - self._min_cents)
        return start + normalized * width

    def _get_indicator_color(self) -> QColor:
        """Get color for indicator based on cents deviation."""
        abs_cents = abs(self._cents)
        if abs_cents <= self._green_zone:
            return QColor(ACCENT_GREEN)
        elif abs_cents <= self._orange_zone:
            return QColor(WARNING_ORANGE)
        else:
            return QColor(ERROR_RED)


class MultiReedMeter(QWidget):
    """
    Unified tuning meter showing all reeds simultaneously.

    Displays multiple vertical indicators on a single horizontal scale
    from -50 to +50 cents, with color-coded zones and labeled tick marks.
    """

    def __init__(self, max_reeds: int = 4, parent=None):
        super().__init__(parent)
        self._max_reeds = max_reeds
        self._num_active_reeds = max_reeds
        self._reed_cents: list[float | None] = [None] * max_reeds

        # Appearance settings
        self._min_cents = -50.0
        self._max_cents = 50.0
        self._green_zone = 5.0   # ±5 cents = green
        self._orange_zone = 15.0  # ±15 cents = orange

        # Reed indicator colors (distinct colors for each reed)
        self._reed_colors = [
            QColor("#4fc3f7"),  # Light blue (Reed 1)
            QColor("#81c784"),  # Light green (Reed 2)
            QColor("#ffb74d"),  # Orange (Reed 3)
            QColor("#f06292"),  # Pink (Reed 4)
        ]

        self.setMinimumSize(400, 70)
        self.setMaximumHeight(80)

    def set_reed_data(self, reed_index: int, cents: float | None):
        """Set cents for a specific reed (None = inactive)."""
        if 0 <= reed_index < self._max_reeds:
            if cents is not None:
                cents = max(self._min_cents, min(self._max_cents, cents))
            self._reed_cents[reed_index] = cents
            self.update()

    def set_num_reeds(self, count: int):
        """Set number of active reeds to display."""
        self._num_active_reeds = min(count, self._max_reeds)
        # Clear inactive reeds
        for i in range(self._num_active_reeds, self._max_reeds):
            self._reed_cents[i] = None
        self.update()

    def set_all_inactive(self):
        """Set all reeds to inactive state."""
        for i in range(self._max_reeds):
            self._reed_cents[i] = None
        self.update()

    def paintEvent(self, event):
        """Paint the multi-reed tuning meter."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()

        # Layout dimensions
        margin_x = 30  # Side margins for scale labels
        margin_top = 20  # Top margin for labels
        bar_height = 24
        bar_y = margin_top + 16  # Position after labels

        bar_width = width - 2 * margin_x
        bar_x = margin_x

        # Draw background bar with rounded corners
        painter.setPen(QPen(QColor(BORDER_COLOR), 1))
        painter.setBrush(QBrush(QColor(PANEL_BACKGROUND)))
        painter.drawRoundedRect(int(bar_x), int(bar_y), int(bar_width), int(bar_height), 4, 4)

        # Draw colored zones inside the bar
        zone_margin = 2
        zone_x = bar_x + zone_margin
        zone_width = bar_width - 2 * zone_margin
        zone_y = bar_y + zone_margin
        zone_height = bar_height - 2 * zone_margin

        # Red zones (outer)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(ERROR_RED).darker(200)))
        # Left red
        red_left_end = self._cents_to_x(-self._orange_zone, zone_x, zone_width)
        painter.drawRect(int(zone_x), int(zone_y), int(red_left_end - zone_x), int(zone_height))
        # Right red
        red_right_start = self._cents_to_x(self._orange_zone, zone_x, zone_width)
        painter.drawRect(int(red_right_start), int(zone_y), int(zone_x + zone_width - red_right_start), int(zone_height))

        # Orange zones
        painter.setBrush(QBrush(QColor(WARNING_ORANGE).darker(200)))
        # Left orange
        orange_left_end = self._cents_to_x(-self._green_zone, zone_x, zone_width)
        painter.drawRect(int(red_left_end), int(zone_y), int(orange_left_end - red_left_end), int(zone_height))
        # Right orange
        orange_right_start = self._cents_to_x(self._green_zone, zone_x, zone_width)
        painter.drawRect(int(orange_right_start), int(zone_y), int(red_right_start - orange_right_start), int(zone_height))

        # Green zone (center)
        painter.setBrush(QBrush(QColor(ACCENT_GREEN).darker(200)))
        painter.drawRect(int(orange_left_end), int(zone_y), int(orange_right_start - orange_left_end), int(zone_height))

        # Draw center line
        center_x = self._cents_to_x(0, bar_x, bar_width)
        painter.setPen(QPen(QColor(TEXT_COLOR), 2))
        painter.drawLine(int(center_x), int(bar_y + 2), int(center_x), int(bar_y + bar_height - 2))

        # Draw tick marks and labels
        tick_values = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        for tick in tick_values:
            tick_x = self._cents_to_x(tick, bar_x, bar_width)

            # Draw tick mark above the bar
            painter.setPen(QPen(QColor(TEXT_SECONDARY), 1))
            painter.drawLine(int(tick_x), int(bar_y - 3), int(tick_x), int(bar_y))

            # Draw label (show absolute value for symmetry, with 0 at center)
            label = str(abs(tick))
            painter.setPen(QColor(TEXT_SECONDARY))
            text_width = painter.fontMetrics().horizontalAdvance(label)
            painter.drawText(int(tick_x - text_width / 2), int(bar_y - 6), label)

        # Draw indicators for each active reed
        active_reeds = []
        for i in range(self._num_active_reeds):
            if self._reed_cents[i] is not None:
                active_reeds.append((i, self._reed_cents[i]))

        # Draw reed indicators with slight offset to avoid overlap
        indicator_width = 4
        for reed_idx, cents in active_reeds:
            indicator_x = self._cents_to_x(cents, bar_x, bar_width)

            # Offset indicators slightly if multiple reeds at similar positions
            offset = 0
            if len(active_reeds) > 1:
                # Spread indicators by reed index
                offset = (reed_idx - (self._num_active_reeds - 1) / 2) * 2

            x = indicator_x + offset

            # Get color based on cents deviation
            indicator_color = self._get_indicator_color(cents)
            reed_color = self._reed_colors[reed_idx % len(self._reed_colors)]

            # Draw vertical indicator line
            painter.setPen(QPen(indicator_color, indicator_width))
            painter.drawLine(int(x), int(bar_y + 2), int(x), int(bar_y + bar_height - 2))

            # Draw reed number indicator below the bar
            painter.setPen(QPen(reed_color, 2))
            marker_y = bar_y + bar_height + 4
            painter.setBrush(QBrush(reed_color))
            painter.drawEllipse(int(x - 4), int(marker_y), 8, 8)

            # Draw reed number in the marker
            painter.setPen(QColor(PANEL_BACKGROUND))
            font_small = QFont()
            font_small.setPointSize(6)
            font_small.setBold(True)
            painter.setFont(font_small)
            painter.drawText(int(x - 3), int(marker_y + 7), str(reed_idx + 1))

    def _cents_to_x(self, cents: float, start: float, width: float) -> float:
        """Convert cents value to x coordinate."""
        normalized = (cents - self._min_cents) / (self._max_cents - self._min_cents)
        return start + normalized * width

    def _get_indicator_color(self, cents: float) -> QColor:
        """Get color for indicator based on cents deviation."""
        abs_cents = abs(cents)
        if abs_cents <= self._green_zone:
            return QColor(ACCENT_GREEN)
        elif abs_cents <= self._orange_zone:
            return QColor(WARNING_ORANGE)
        else:
            return QColor(ERROR_RED)
