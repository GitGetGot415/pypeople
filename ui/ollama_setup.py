
import pygame

from ollama_manager import open_ollama_download, check_ollama, download_model
from .button import Button
from .text import draw_centered_text
from .theme import GRASS, PAPER, MUTED, BORDER_RADIUS


class OllamaSetupScreen:
    def __init__(self):
        self.manual_button = Button((340, 220, 280, 48), "Install Ollama")
        self.continue_button = Button((340, 325, 280, 40), "I've Installed Ollama")
        self.download_button = Button((340, 455, 280, 48), "Download Llama 3")
        self.back_button = Button((50, 580, 60, 30), "Back")

        self.ollama_ready = False
        self.message = ""

    def open(self):
        self.ollama_ready = False
        self.message = ""

    def handle_event(self, app, event):
        if self.manual_button.clicked(event):
            open_ollama_download()

        elif self.continue_button.clicked(event):
            self.ollama_ready = check_ollama(force_refresh=True)

            if self.ollama_ready:
                self.message = "Ollama is running! You can download Llama 3."
            else:
                self.message = "Cannot connect to Ollama. Make sure it is running."

        elif self.download_button.clicked(event, enabled=self.ollama_ready):
            if download_model():
                self.message = "When the terminal says complete, you can close it."
            else:
                self.message = "Could not launch download. Try restarting PyPeople."

        elif self.back_button.clicked(event):
            app.settings.refresh_models()
            app.screen = "settings"

    def draw(self, app):
        surface = app.window
        surface.fill(GRASS)

        pygame.draw.rect(
            surface,
            PAPER,
            (120, 65, 720, 515),
            border_radius=BORDER_RADIUS
        )

        draw_centered_text(surface, "Ollama Setup", app.heading_font, 90)

        # Step 1: Install Ollama
        draw_centered_text(surface, "Step 1: Install Ollama", app.font, 155)
        draw_centered_text(
            surface,
            "Download Ollama from its official website.",
            app.small_font, 188, MUTED
        )

        self.manual_button.draw(surface, app.font)

        draw_centered_text(
            surface,
            "Already installed? Make sure Ollama is running.",
            app.small_font, 290, MUTED
        )

        self.continue_button.draw(surface, app.font)

        # Step 2: Download Llama 3
        draw_centered_text(surface, "Step 2: Download Llama 3", app.font, 390)
        draw_centered_text(
            surface,
            "A terminal will open. Internet connection required.",
            app.small_font, 425, MUTED
        )

        self.download_button.draw(
            surface, app.font, enabled=self.ollama_ready
        )

        if self.message:
            draw_centered_text(
                surface, self.message,
                app.small_font, 530, MUTED
            )

        self.back_button.draw(surface, app.font)
