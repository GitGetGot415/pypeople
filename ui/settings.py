import pygame

from .button import Button
from .text import draw_centered_text
from .text_input import TextInput
from .theme import BORDER_RADIUS, GRASS, INK, MUTED, PAPER


class SettingsScreen:
    def __init__(self):
        self.save_button = Button((340, 400, 280, 54), "Save & Back")
        self.key_input = TextInput(
            (180, 280, 600, 52),
            max_length=512,
            secret=True,
            strip_paste=True,
            placeholder="Paste your key here",
        )

    def open(self, api_key):
        self.key_input.set_value(api_key)
        self.key_input.focus()

    def handle_event(self, app, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.key_input.set_value(app.api_key)
            self.key_input.active = False
            app.screen = "menu"
        elif self.save_button.clicked(event):
            app.api_key = self.key_input.text.strip()
            self.key_input.set_value(app.api_key)
            self.key_input.active = False
            app.screen = "menu"
        else:
            self.key_input.handle_event(event)

    def draw(self, app):
        app.window.fill(GRASS)
        pygame.draw.rect(app.window, PAPER, (120, 120, 720, 405), border_radius=BORDER_RADIUS)
        draw_centered_text(app.window, "Settings", app.heading_font, 160)
        app.window.blit(app.font.render("Gemini API key", True, INK), (180, 243))
        self.key_input.draw(app.window, app.font)
        app.window.blit(app.small_font.render("Stored only for this session.", True, MUTED), (180, 345))
        self.save_button.draw(app.window, app.font)
        draw_centered_text(app.window, "Get a key at Google AI Studio: aistudio.google.com/apikey", app.small_font, 475, MUTED)
        draw_centered_text(app.window, self.key_input.error or "Ctrl+V to paste. Escape discards your edits.", app.small_font, 553)
