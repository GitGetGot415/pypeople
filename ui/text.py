import pygame

from .theme import INK


def draw_centered_text(surface, text, font, y, color=INK):
    rendered = font.render(text, True, color)
    surface.blit(rendered, rendered.get_rect(midtop=(surface.get_width() // 2, y)))


def draw_wrapped_text(surface, text, font, color, rect):
    """Wrap words (and very long words), clipping all drawing to the rectangle."""
    rect = pygame.Rect(rect)
    lines = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}" if line else word
        if font.size(candidate)[0] <= rect.width:
            line = candidate
            continue
        if line:
            lines.append(line)
            line = ""
        # A single unbroken word must fit too.
        for char in word:
            if line and font.size(line + char)[0] > rect.width:
                lines.append(line)
                line = ""
            line += char
    if line:
        lines.append(line)
    previous_clip = surface.get_clip()
    surface.set_clip(rect.clip(previous_clip))
    y = rect.y
    for line in lines:
        if y + font.get_linesize() > rect.bottom:
            break
        surface.blit(font.render(line, True, color), (rect.x, y))
        y += font.get_linesize()
    surface.set_clip(previous_clip)
