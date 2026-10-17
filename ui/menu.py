import pygame
from llm import MODEL
from .button import Button
from .text import draw_centered_text, draw_wrapped_text
from constants import BORDER_RADIUS, GRASS, MUTED, PAPER, INK

class MenuScreen:
    def __init__(self):
        self.start_button = Button((410, 290, 220, 54), "New Sim")
        self.load_button = Button((650, 290, 220, 54), "Load Sim")
        self.character_button = Button((410, 360, 460, 54), "Character Creator")
        self.settings_button = Button((410, 430, 220, 54), "Settings")
        self.exit_button = Button((650, 430, 220, 54), "Exit")
        self.message = ""

    def handle_event(self, app, event):
        if self.start_button.clicked(event):
            app.screen = "setup"

        elif self.load_button.clicked(event):
            app.open_load_simulations()

        elif self.character_button.clicked(event):
            app.screen = "character"

        elif self.settings_button.clicked(event):
            app.open_settings()

        elif self.exit_button.clicked(event):
            app.running = False

    def draw(self, app):
        app.window.fill(GRASS)
        pygame.draw.rect(app.window, PAPER, (370, 95, 540, 530), border_radius=BORDER_RADIUS)
        if app.provider == "gemini":
            model_name = f"Gemini ({MODEL})"
        elif app.provider == "ollama":
            model_name = f"Ollama ({app.ollama_model})"
        else:
            model_name = "No AI selected"

        draw_centered_text(
            app.window,
            "Pypeople",
            app.title_font,
            145,
            INK,
        )

        draw_centered_text(
            app.window,
            model_name,
            app.small_font,
            205,
            MUTED,
        )
        draw_centered_text(app.window, "say whatever you want", app.font, 225, MUTED)
        for button in (self.start_button, self.load_button, self.character_button, self.settings_button, self.exit_button):
            button.draw(app.window, app.font)
        draw_centered_text(app.window, "(no you can't flirt with the ai yet)", app.small_font, 550, MUTED)
        if self.message:
            draw_wrapped_text(app.window, self.message, app.small_font, MUTED, (420, 580, 440, 40))
