import queue
import threading

import pygame

from ollama_manager import DEFAULT_OLLAMA_MODEL, list_models, save_model

from .button import Button
from .text import draw_centered_text, draw_wrapped_text
from .text_input import TextInput
from .theme import BORDER_RADIUS, GRASS, INK, MUTED, PAPER


class SettingsScreen:
    def __init__(self):
        self.gemini_button = Button((180, 210, 280, 48), "Gemini")
        self.ollama_button = Button((500, 210, 280, 48), "Ollama")
        self.previous_model_button = Button((180, 310, 48, 52), "<")
        self.next_model_button = Button((732, 310, 48, 52), ">")
        self.refresh_button = Button((590, 370, 190, 32), "Refresh models")
        self.save_button = Button((340, 410, 280, 54), "Save & Back")
        self.setup_button = Button((340, 470, 280, 48), "Setup Ollama")
        self.key_input = TextInput(
            (180, 310, 600, 52),
            max_length=512,
            secret=True,
            strip_paste=True,
            placeholder="Paste your key here",
        )
        self.provider = "gemini"
        self.ollama_model = DEFAULT_OLLAMA_MODEL
        self.models = [DEFAULT_OLLAMA_MODEL]
        self.installed_models = []
        self.model_worker = None
        self.model_results = queue.Queue()
        self.model_message = ""
        self.error = ""

    def open(self, provider, api_key, ollama_model):
        self.provider = provider
        self.ollama_model = ollama_model
        self.error = ""
        self.update_model_choices()
        self.key_input.set_value(api_key)
        if provider == "gemini":
            self.key_input.focus()
        else:
            self.key_input.active = False
            self.refresh_models()

    def update_model_choices(self):
        # llama3 and llama3:latest refer to the same default model in Ollama.
        names = [DEFAULT_OLLAMA_MODEL, self.ollama_model]
        names.extend(name for name in self.installed_models if name != "llama3:latest")
        self.models = list(dict.fromkeys(names))

    def refresh_models(self):
        if self.model_worker is not None:
            return
        self.model_message = "Loading installed models..."

        def fetch():
            try:
                self.model_results.put((list_models(), ""))
            except ValueError as error:
                self.model_results.put(([], str(error)))

        self.model_worker = threading.Thread(target=fetch, daemon=True, name="Ollama-model-list")
        self.model_worker.start()

    def update(self):
        try:
            names, error = self.model_results.get_nowait()
        except queue.Empty:
            return
        self.model_worker = None
        self.installed_models = names
        self.update_model_choices()
        self.model_message = error
        if not error and not names:
            self.model_message = "No models installed. Use Setup Ollama, then Refresh models."

    def change_model(self, direction):
        index = self.models.index(self.ollama_model)
        self.ollama_model = self.models[(index + direction) % len(self.models)]
        self.error = ""

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
            self.refresh_models()
        elif self.provider == "ollama" and self.previous_model_button.clicked(event):
            self.change_model(-1)
        elif self.provider == "ollama" and self.next_model_button.clicked(event):
            self.change_model(1)
        elif self.provider == "ollama" and self.refresh_button.clicked(event):
            self.refresh_models()
        elif self.provider == "ollama" and self.setup_button.clicked(event):
            app.ollama_setup.open()
            app.screen = "ollama_setup"
        elif self.save_button.clicked(event):
            if self.ollama_model != app.ollama_model:
                try:
                    save_model(self.ollama_model)
                except OSError:
                    self.error = "Could not save your model choice. Check folder permissions and try again."
                    return
            app.ollama_model = self.ollama_model
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
            index = self.models.index(self.ollama_model) + 1
            label = f"Local Ollama model ({index}/{len(self.models)})"
            app.window.blit(app.font.render(label, True, INK), (180, 276))
            self.previous_model_button.draw(app.window, app.font, enabled=len(self.models) > 1)
            self.next_model_button.draw(app.window, app.font, enabled=len(self.models) > 1)
            draw_wrapped_text(app.window, self.ollama_model, app.font, INK, (244, 310, 472, 52))
            app.window.blit(app.small_font.render("Your model choice is saved between launches.", True, MUTED), (180, 375))
            self.refresh_button.draw(app.window, app.small_font, enabled=self.model_worker is None)
            self.setup_button.draw(app.window, app.font)
        self.save_button.draw(app.window, app.font)
        if self.provider == "gemini":
            draw_centered_text(app.window, "Get a key at Google AI Studio: aistudio.google.com/apikey", app.small_font, 475, MUTED)
            hint = self.key_input.error or "Ctrl+V to paste. Escape discards your edits."
        else:
            hint = self.model_message or "Use < and > to choose a model. Escape discards your edits."
            installed = (self.ollama_model in self.installed_models
                         or f"{self.ollama_model}:latest" in self.installed_models)
            if not self.model_message and not installed:
                hint = "Selected model is not installed. Download it in Ollama, then Refresh models."
        draw_wrapped_text(app.window, self.error or hint, app.small_font, INK, (120, 553, 720, 65))
