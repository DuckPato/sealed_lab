"""Application icon: drawn with QPainter (three fanned cards and a rising bar chart), shipped as assets/ files."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPen

ASSETS = Path(__file__).resolve().parent / "assets"
ICO = ASSETS / "sealed_lab.ico"
PNG = ASSETS / "sealed_lab.png"


def draw_icon(size: int = 256) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 256.0
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#1d2740"))
    p.drawRoundedRect(QRectF(8 * s, 8 * s, 240 * s, 240 * s), 48 * s, 48 * s)
    card = QRectF(-48 * s, -68 * s, 96 * s, 136 * s)
    for angle, dx, col in ((-16, -34, "#4a90d9"), (16, 34, "#3f9d62"), (0, 0, "#e8c15a")):
        p.save()
        p.translate(128 * s + dx * s, 136 * s)
        p.rotate(angle)
        p.setPen(QPen(QColor("#f4f1e8"), 5 * s))
        p.setBrush(QColor(col))
        p.drawRoundedRect(card, 10 * s, 10 * s)
        p.restore()
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor("#1d2740"))
    for i, h in enumerate((28, 46, 68)):        # rising bars on the front card
        p.drawRoundedRect(QRectF((98 + i * 22) * s, (178 - h) * s, 16 * s, h * s), 3 * s, 3 * s)
    p.setPen(QPen(QColor("#1d2740"), 6 * s, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    p.drawLine(QPointF(98 * s, 96 * s), QPointF(158 * s, 84 * s))
    p.end()
    return img


def app_icon() -> QIcon:
    if PNG.exists():
        return QIcon(str(PNG))
    from PySide6.QtGui import QPixmap

    return QIcon(QPixmap.fromImage(draw_icon()))


def write_assets() -> None:
    """Regenerate the shipped icon files (run once after changing the drawing)."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    draw_icon(256).save(str(PNG))
    draw_icon(256).save(str(ICO))


if __name__ == "__main__":  # python -m sealed_lab.gui.icon
    from PySide6.QtGui import QGuiApplication

    _app = QGuiApplication([])
    write_assets()
