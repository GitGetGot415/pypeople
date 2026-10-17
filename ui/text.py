import re

import pygame

from constants import DIALOGUE_PAGE_LINES, DIALOGUE_PART_MAX_CHARACTERS, INK


def draw_centered_text(surface, text, font, y, color=INK):
    rendered = font.render(text, True, color)
    surface.blit(rendered, rendered.get_rect(midtop=(surface.get_width() // 2, y)))


def wrap_text(text, font, width):
    """Wrap words, including words longer than the available width."""
    lines = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}" if line else word
        if font.size(candidate)[0] <= width:
            line = candidate
            continue
        if line:
            lines.append(line)
            line = ""
        # A single unbroken word must fit too.
        for char in word:
            if line and font.size(line + char)[0] > width:
                lines.append(line)
                line = ""
            line += char
    if line:
        lines.append(line)
    return lines


def paginate_text(text, font, width, max_lines=DIALOGUE_PAGE_LINES):
    """Fit dialogue pages, preferring sentence breaks and preserving all words."""
    def fits(text):
        return len(text) <= DIALOGUE_PART_MAX_CHARACTERS and len(wrap_text(text, font, width)) <= max_lines

    pages = []
    page = ""
    for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
        candidate = f"{page} {sentence}" if page else sentence
        if fits(candidate):
            page = candidate
            continue
        if page:
            pages.append(page)
            page = ""
        for word in sentence.split():
            candidate = f"{page} {word}" if page else word
            if fits(candidate):
                page = candidate
                continue
            if page:
                pages.append(page)
                page = ""
            # A very long unbroken word also needs to fit a single page.
            for char in word:
                if page and not fits(page + char):
                    pages.append(page)
                    page = ""
                page += char
    if page:
        pages.append(page)
    return pages


def draw_wrapped_text(surface, text, font, color, rect):
    """Wrap words (and very long words), clipping all drawing to the rectangle."""
    rect = pygame.Rect(rect)
    lines = wrap_text(text, font, rect.width)
    previous_clip = surface.get_clip()
    surface.set_clip(rect.clip(previous_clip))
    y = rect.y
    for line in lines:
        if y + font.get_linesize() > rect.bottom:
            break
        surface.blit(font.render(line, True, color), (rect.x, y))
        y += font.get_linesize()
    surface.set_clip(previous_clip)
