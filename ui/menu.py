import pygame
from model_transport import MODEL
from ollama_manager import check_ollama
from .button import Button
from .text import draw_centered_text, draw_wrapped_text
from constants import BORDER_RADIUS, GRASS, MUTED, PAPER, INK, MODAL_OVERLAY_COLOR

class MenuScreen:
    def __init__(self):
        self.start_button = Button((410, 290, 220, 54), "New Sim")
        self.load_button = Button((650, 290, 220, 54), "Load Sim")
        self.character_button = Button((410, 360, 460, 54), "Character Creator")
        self.settings_button = Button((410, 430, 220, 54), "Settings")
        self.exit_button = Button((650, 430, 220, 54), "Exit")
        self.message = ""
        self.ai_warning = ""
        self.pending_screen = None
        self.configure_button = Button((350, 415, 180, 44), "Settings")
        self.continue_button = Button((550, 415, 180, 44), "Continue anyway")
        self.cancel_button = Button((750, 415, 180, 44), "Cancel")

    def handle_event(self, app, event):
        if self.ai_warning:
            if self.cancel_button.clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.ai_warning = ""
                self.pending_screen = None
            elif self.configure_button.clicked(event):
                self.ai_warning = ""
                self.pending_screen = None
                app.open_settings()
            elif self.continue_button.clicked(event):
                self._open_requested_screen(app)
            return

        if self.start_button.clicked(event) or self.load_button.clicked(event):
            self.pending_screen = "load_sim" if self.load_button.clicked(event) else "setup"
            if app.provider == "gemini":
                self.ai_warning = "" if app.api_key.strip() else "No Gemini API key is configured."
            elif app.provider == "ollama":
                self.ai_warning = "" if check_ollama() else "Ollama is not running or could not be reached."
            else:
                self.ai_warning = "Choose an AI provider in Settings."
            if not self.ai_warning:
                self._open_requested_screen(app)

        elif self.character_button.clicked(event):
            app.screen = "character"

        elif self.settings_button.clicked(event):
            app.open_settings()

        elif self.exit_button.clicked(event):
            app.running = False

    def _open_requested_screen(self, app):
        screen = self.pending_screen
        self.pending_screen = None
        self.ai_warning = ""
        if screen == "load_sim":
            app.open_load_simulations()
        else:
            app.screen = screen

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
        if self.ai_warning:
            overlay = pygame.Surface(app.window.get_size(), pygame.SRCALPHA)
            overlay.fill(MODAL_OVERLAY_COLOR)
            app.window.blit(overlay, (0, 0))
            pygame.draw.rect(app.window, PAPER, (315, 240, 650, 240), border_radius=BORDER_RADIUS)
            draw_centered_text(app.window, "AI setup needed", app.heading_font, 265, INK)
            draw_wrapped_text(
                app.window,
                self.ai_warning + " Open Settings to configure AI, or continue without generated dialogue.",
                app.font, INK, (350, 320, 580, 80),
            )
            for button in (self.configure_button, self.continue_button, self.cancel_button):
                button.draw(app.window, app.small_font)
