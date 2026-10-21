"""Bundled GNU Unifont for all game text."""

from pathlib import Path
import unicodedata

import pygame

from constants import UNICODE_FONT_FILES


def text_units(text):
    """Keep combining marks and joined sequences with their preceding letter."""
    unit = ""
    for char in text:
        if unit and not (unicodedata.category(char).startswith("M")
                         or char == "\u200d" or unit.endswith("\u200d")):
            yield unit
            unit = ""
        unit += char
    if unit:
        yield unit


class UnicodeFont:
    """The render/size/line-spacing interface used by all game text controls."""

    def __init__(self, size):
        root = Path(__file__).resolve().parent.parent
        self._primary, self._upper = (
            pygame.font.Font(str(root / filename), size) for filename in UNICODE_FONT_FILES
        )

    def _font_for(self, unit):
        return self._upper if ord(unit[0]) > 0xFFFF else self._primary

    def _runs(self, text):
        if text.isascii():
            if text:
                yield self._primary, text
            return
        current = self._primary
        run = []
        for unit in text_units(text):
            font = self._font_for(unit)
            if font is not current and run:
                yield current, "".join(run)
                run = []
            current = font
            run.append(unit)
        if run:
            yield current, "".join(run)

    def _layout(self, text):
        x = 0
        height = self.get_height()
        runs = []
        for font, run in self._runs(text):
            width, run_height = font.size(run)
            y = self._primary.get_ascent() - font.get_ascent()
            runs.append((font, run, x, y))
            x += width
            height = max(height, y + run_height)
        return runs, (x, height)

    def size(self, text):
        return self._layout(text)[1]

    def render(self, text, antialias, color, background=None):
        runs, dimensions = self._layout(text)
        surface = pygame.Surface(dimensions, pygame.SRCALPHA)
        if background is not None:
            surface.fill(background)
        for font, run, x, y in runs:
            surface.blit(font.render(run, antialias, color), (x, y))
        return surface

    def get_height(self):
        return self._primary.get_height()

    def get_linesize(self):
        return self._primary.get_linesize()
