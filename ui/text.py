import pygame

from constants import DIALOGUE_PAGE_LINES, INK
from dialogue_text import split_sentences


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
    """Group complete sentences; a long sentence stays together and can scroll."""
    pages = []
    page = ""
    for sentence in split_sentences(text):
        candidate = f"{page} {sentence}" if page else sentence
        if page and len(wrap_text(candidate, font, width)) > max_lines:
            pages.append(page)
            page = sentence
        else:
            page = candidate
    if page:
        pages.append(page)
    return pages


def paginate_parts(parts, font, width, max_lines=DIALOGUE_PAGE_LINES):
    """Rejoin one speaker's part fragments, retaining the source of each page."""
    texts, sources, offset = [], [], 0
    for text, source in parts:
        text = " ".join(text.split())
        if text:
            texts.append(text)
            offset += len(text) + 1
            sources.append((offset, source))
    pages, offset, source_index = [], 0, 0
    for page in paginate_text(" ".join(texts), font, width, max_lines):
        while offset >= sources[source_index][0]:
            source_index += 1
        pages.append((sources[source_index][1], page))
        offset += len(page) + 1
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
