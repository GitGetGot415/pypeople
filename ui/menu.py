import pygame

from .button import Button
from .text import draw_centered_text
from .theme import BORDER_RADIUS, GRASS, MUTED, PAPER


class MenuScreen:
    def __init__(self):
        self.start_button = Button((340, 300, 280, 54), "Start")
        self.settings_button = Button((340, 372, 280, 54), "Settings")
        self.exit_button = Button((340, 444, 280, 54), "Exit")

    def handle_event(self, app, event):
        if self.start_button.clicked(event):
            app.start_game()
        elif self.settings_button.clicked(event):
            app.open_settings()
        elif self.exit_button.clicked(event):
            app.running = False

    def draw(self, app):
        app.window.fill(GRASS)
        pygame.draw.rect(app.window, PAPER, (240, 110, 480, 470), border_radius=BORDER_RADIUS)
        draw_centered_text(app.window, "Pypeople", app.title_font, 155)
        draw_centered_text(app.window, "say whatever you want", app.font, 245, MUTED)
        for button in (self.start_button, self.settings_button, self.exit_button):
            button.draw(app.window, app.font)
        draw_centered_text(app.window, "(no you can't flirt with the ai yet)", app.small_font, 541, MUTED)
