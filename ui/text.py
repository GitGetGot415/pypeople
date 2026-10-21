import pygame

from constants import DIALOGUE_PAGE_LINES, DIALOGUE_PAGE_WIDTH, INK, SCROLLBAR_WIDTH, SCROLLBAR_MIN_THUMB
from dialogue_text import split_sentences
from .fonts import text_units


def scrollbar_rects(rect, scroll, limit):
    track = pygame.Rect(rect.right - SCROLLBAR_WIDTH, rect.top, SCROLLBAR_WIDTH, rect.height)
    height = max(SCROLLBAR_MIN_THUMB, round(rect.height * rect.height / (rect.height + limit)))
    top = track.top + round((track.height - height) * scroll / limit) if limit else track.top
    return track, pygame.Rect(track.x, top, track.width, height)


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
        for char in text_units(word):
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


def paginate_parts(parts, font, width, max_lines=DIALOGUE_PAGE_LINES, *, condensed=False):
    """Keep complete sentences and part emotions on pages shared by both views."""
    texts, sources, offset = [], [], 0
    for text, source in parts:
        text = " ".join(text.split())
        if text:
            texts.append(text)
            offset += len(text) + 1
            sources.append((offset, source))
    if condensed:
        return [(sources[0][1], " ".join(texts))] if texts else []
    sections, offset, source_index = [], 0, 0
    for sentence in split_sentences(" ".join(texts)):
        while offset >= sources[source_index][0]:
            source_index += 1
        # A part may begin halfway through a sentence. Finish that sentence
        # with its original emotion, then use the next sentence's source.
        if sections and sections[-1][0] == source_index:
            sections[-1][1].append(sentence)
        else:
            sections.append((source_index, [sentence]))
        offset += len(sentence) + 1
    return [
        (sources[index][1], page)
        for index, sentences in sections
        for page in paginate_text(" ".join(sentences), font, min(width, DIALOGUE_PAGE_WIDTH), max_lines)
    ]


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
