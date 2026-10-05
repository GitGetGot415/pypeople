import pygame

from .button import Button
from .text import draw_centered_text
from .text_input import TextInput
from .theme import BORDER_RADIUS, GRASS, INK, MUTED, PAPER


class SettingsScreen:
    def __init__(self):
        self.gemini_button = Button((180, 210, 280, 48), "Gemini")
        self.ollama_button = Button((500, 210, 280, 48), "Ollama (llama3)")
        self.save_button = Button((340, 410, 280, 54), "Save & Back")
        self.key_input = TextInput(
            (180, 310, 600, 52),
            max_length=512,
            secret=True,
            strip_paste=True,
            placeholder="Paste your key here",
        )
        self.provider = "gemini"

    def open(self, provider, api_key):
        self.provider = provider
        self.key_input.set_value(api_key)
        if provider == "gemini":
            self.key_input.focus()
        else:
            self.key_input.active = False

    def handle_event(self, app, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.key_input.set_value(app.api_key)
            self.key_input.active = False
            app.screen = "menu"
        elif self.gemini_button.clicked(event):
            self.provider = "gemini"
            self.key_input.focus()
        elif self.ollama_button.clicked(event):
            self.provider = "ollama"
            self.key_input.active = False
        elif self.save_button.clicked(event):
            app.provider = self.provider
            app.api_key = self.key_input.text.strip()
            self.key_input.set_value(app.api_key)
            self.key_input.active = False
            app.screen = "menu"
        elif self.provider == "gemini":
            self.key_input.handle_event(event)

    def draw(self, app):
        app.window.fill(GRASS)
        pygame.draw.rect(app.window, PAPER, (120, 120, 720, 405), border_radius=BORDER_RADIUS)
        draw_centered_text(app.window, "Settings", app.heading_font, 160)
        self.gemini_button.draw(app.window, app.font, enabled=self.provider != "gemini")
        self.ollama_button.draw(app.window, app.font, enabled=self.provider != "ollama")
        if self.provider == "gemini":
            app.window.blit(app.font.render("Gemini API key", True, INK), (180, 276))
            self.key_input.draw(app.window, app.font)
            app.window.blit(app.small_font.render("Stored only for this session.", True, MUTED), (180, 375))
        else:
            app.window.blit(app.font.render("Local Ollama model: llama3", True, INK), (180, 310))
            app.window.blit(app.small_font.render("No API key needed!", True, MUTED), (180, 375))
        self.save_button.draw(app.window, app.font)
        if self.provider == "gemini":
            draw_centered_text(app.window, "Get a key at Google AI Studio: aistudio.google.com/apikey", app.small_font, 475, MUTED)
            hint = self.key_input.error or "Ctrl+V to paste. Escape discards your edits."
        else:
            draw_centered_text(app.window, "Install Ollama, then run: ollama pull llama3", app.small_font, 475, MUTED)
            hint = "Ctrl+V to paste. Escape discards your edits."
        draw_centered_text(app.window, hint, app.small_font, 553)
