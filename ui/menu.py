import pygame

from .button import Button
from .text import draw_centered_text, draw_wrapped_text
from .theme import BORDER_RADIUS, GRASS, MUTED, PAPER


class MenuScreen:
    def __init__(self):
        self.start_button = Button((340, 240, 280, 54), "Start")
        self.character_button = Button((340, 300, 280, 54), "Character Creator")
        self.settings_button = Button((340, 360, 280, 54), "Settings")
        self.exit_button = Button((340, 420, 280, 54), "Exit")
        self.message = ""

    def handle_event(self, app, event):
        if self.start_button.clicked(event):
            app.screen = "setup"
        elif self.character_button.clicked(event):
            app.screen = "character"
        elif self.settings_button.clicked(event):
            app.open_settings()
        elif self.exit_button.clicked(event):
            app.running = False

    def draw(self, app):
        app.window.fill(GRASS)
        pygame.draw.rect(app.window, PAPER, (240, 55, 480, 530), border_radius=BORDER_RADIUS)
        draw_centered_text(app.window, "Pypeople", app.title_font, 105)
        draw_centered_text(app.window, "say whatever you want", app.font, 185, MUTED)
        for button in (self.start_button, self.character_button, self.settings_button, self.exit_button):
            button.draw(app.window, app.font)
        draw_centered_text(app.window, "(no you can't flirt with the ai yet)", app.small_font, 510, MUTED)
        if self.message:
            draw_wrapped_text(app.window, self.message, app.small_font, MUTED, (260, 540, 440, 40))
