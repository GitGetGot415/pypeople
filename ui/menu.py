import pygame
from llm import MODEL, OLLAMA_MODEL
from .button import Button
from .text import draw_centered_text, draw_wrapped_text
from .theme import BORDER_RADIUS, GRASS, MUTED, PAPER, INK
from ollama_manager import check_ollama

class MenuScreen:
    def __init__(self):
        self.start_button = Button((340, 240, 280, 54), "Start")
        self.character_button = Button((340, 300, 280, 54), "Character Creator")
        self.settings_button = Button((340, 360, 280, 54), "Settings")
        self.exit_button = Button((340, 420, 280, 54), "Exit")
        self.message = ""

        self.confirm_start = False

        self.yes_button = Button(
            (315, 350, 145, 48),
            "Yes",
        )

        self.no_button = Button(
            (500, 350, 145, 48),
            "No",
        )

    def handle_event(self, app, event):

        # While the popup is open, ONLY its buttons work.
        if self.confirm_start:

            if self.yes_button.clicked(event):
                self.confirm_start = False

                # Original Start button behavior
                app.screen = "setup"

            elif self.no_button.clicked(event):
                self.confirm_start = False

            # Block all underlying menu interactions
            return

        # Check the currently selected AI provider
        if self.start_button.clicked(event):

            if app.provider == "gemini":
                configured = bool(app.api_key.strip())

            elif app.provider == "ollama":
                configured = check_ollama()

            else:
                configured = False

            if configured:
                # Original Start behavior
                app.screen = "setup"

            else:
                # No configured AI: ask for confirmation
                self.confirm_start = True

        elif self.character_button.clicked(event):
            app.screen = "character"

        elif self.settings_button.clicked(event):
            app.open_settings()

        elif self.exit_button.clicked(event):
            app.running = False

    def draw(self, app):
        app.window.fill(GRASS)
        pygame.draw.rect(app.window, PAPER, (240, 55, 480, 530), border_radius=BORDER_RADIUS)
        if app.provider == "gemini":
            model_name = f"Gemini ({MODEL})"
        elif app.provider == "ollama":
            model_name = f"Ollama ({OLLAMA_MODEL})"
        else:
            model_name = "No AI selected"

        draw_centered_text(
            app.window,
            "Pypeople",
            app.title_font,
            105,
            INK,
        )

        draw_centered_text(
            app.window,
            model_name,
            app.small_font,
            165,
            MUTED,
        )
        draw_centered_text(app.window, "say whatever you want", app.font, 185, MUTED)
        for button in (self.start_button, self.character_button, self.settings_button, self.exit_button):
            button.draw(app.window, app.font)
        draw_centered_text(app.window, "(no you can't flirt with the ai yet)", app.small_font, 510, MUTED)
        if self.message:
            draw_wrapped_text(app.window, self.message, app.small_font, MUTED, (260, 540, 440, 40))
        if self.confirm_start:

            # Darken the background
            overlay = pygame.Surface(
                app.window.get_size(),
                pygame.SRCALPHA,
            )

            overlay.fill((0, 0, 0, 140))

            app.window.blit(overlay, (0, 0))

            # Popup background
            popup = pygame.Rect(
                245, 235, 470, 195
            )

            pygame.draw.rect(
                app.window,
                PAPER,
                popup,
                border_radius=BORDER_RADIUS,
            )

            # Popup text
            draw_centered_text(
                app.window,
                "API keys not configured,",
                app.heading_font,
                285,
                INK,
            )

            draw_centered_text(
                app.window,
                "continue anyways?",
                app.font,
                323,
                INK,
            )

            # Confirmation buttons
            self.yes_button.draw(
                app.window,
                app.font,
            )

            self.no_button.draw(
                app.window,
                app.font,
            )