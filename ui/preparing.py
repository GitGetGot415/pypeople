"""Background preparation keeps incomplete residents out of the active game."""

import pygame

from .button import Button
from .text import draw_centered_text, draw_wrapped_text
from .theme import GRASS, PAPER, INK, MUTED


class PreparingScreen:
    def __init__(self):
        self.retry = Button((640, 510, 150, 44), "Retry")
        self.cancel = Button((450, 510, 150, 44), "Cancel")

    def handle_event(self, app, event):
        if self.cancel.clicked(event) or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            app.cancel_preparation()
        elif self.retry.clicked(event):
            for npc in app.npcs:
                npc.background_error = ""

    def draw(self, app):
        app.window.fill(GRASS)
        pygame.draw.rect(app.window, PAPER, (230, 100, 820, 500))
        draw_centered_text(app.window, "Remembering former lives", app.heading_font, 150)
        ready = sum(n.background is not None for n in app.npcs)
        draw_centered_text(app.window, f"Prepared {ready} of {len(app.npcs)} residents", app.font, 225)
        current = next((n for n in app.npcs if n.background is None), None)
        if current:
            draw_centered_text(app.window, current.name, app.font, 280, INK)
            message = current.background_error or "Generating a persistent background using your selected provider..."
            draw_wrapped_text(app.window, message, app.font, MUTED, (280, 335, 720, 140))
        self.cancel.draw(app.window, app.font)
        self.retry.draw(app.window, app.font, enabled=bool(current and current.background_error))
