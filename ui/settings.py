import queue
import threading

import pygame

from app_settings import save_settings
from character_data import NPC_TRAIT_MODES
from ollama_manager import DEFAULT_OLLAMA_MODEL, list_models, save_model

from .button import Button
from .text import draw_centered_text, draw_wrapped_text
from .text_input import TextInput
from constants import BORDER_RADIUS, GRASS, INK, MUTED, PAPER


def format_model_size(size):
    if size is None:
        return "unavailable"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1000


class SettingsScreen:
    def __init__(self):
        self.gemini_button = Button((340, 160, 280, 42), "Gemini")
        self.ollama_button = Button((660, 160, 280, 42), "Ollama")
        self.previous_model_button = Button((340, 242, 48, 48), "<")
        self.next_model_button = Button((892, 242, 48, 48), ">")
        self.refresh_button = Button((750, 297, 190, 30), "Refresh models")
        self.save_button = Button((340, 580, 280, 48), "Save & Back")
        self.setup_button = Button((660, 580, 280, 48), "Setup Ollama")
        self.voice_rewrite_button = Button((340, 435, 600, 38), "")
        self.validation_button = Button((340, 503, 600, 38), "")
        self.key_input = TextInput(
            (340, 242, 600, 48),
            max_length=512,
            secret=True,
            strip_paste=True,
            placeholder="Paste your key here",
        )
        self.provider = "gemini"
        self.npc_trait_mode = "normal"
        self.voice_rewrites = True
        self.dialogue_validation = True
        self.trait_mode_buttons = {
            mode: Button((340 + index * 205, 365, 190, 38), mode.title())
            for index, mode in enumerate(NPC_TRAIT_MODES)
        }
        self.ollama_model = DEFAULT_OLLAMA_MODEL
        self.models = [DEFAULT_OLLAMA_MODEL]
        self.installed_models = {}
        self.model_worker = None
        self.model_results = queue.Queue()
        self.model_message = ""
        self.error = ""

    def open(self, provider, api_key, ollama_model, npc_trait_mode="normal", *,
             voice_rewrites=True, dialogue_validation=True):
        self.provider = provider
        self.ollama_model = ollama_model
        self.npc_trait_mode = npc_trait_mode
        self.voice_rewrites = voice_rewrites
        self.dialogue_validation = dialogue_validation
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
                self.model_results.put(({}, str(error)))

        self.model_worker = threading.Thread(target=fetch, daemon=True, name="Ollama-model-list")
        self.model_worker.start()

    def update(self):
        try:
            models, error = self.model_results.get_nowait()
        except queue.Empty:
            return
        self.model_worker = None
        self.installed_models = models
        self.update_model_choices()
        self.model_message = error
        if not error and not models:
            self.model_message = "No models installed. Use Setup Ollama, then Refresh models."

    def change_model(self, direction):
        index = self.models.index(self.ollama_model)
        self.ollama_model = self.models[(index + direction) % len(self.models)]
        self.error = ""

    def handle_event(self, app, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.key_input.set_value(app.api_key)
            self.key_input.active = False
            self.npc_trait_mode = app.npc_trait_mode
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
        elif any(button.clicked(event) for button in self.trait_mode_buttons.values()):
            self.npc_trait_mode = next(
                mode for mode, button in self.trait_mode_buttons.items() if button.rect.collidepoint(event.pos)
            )
        elif self.voice_rewrite_button.clicked(event):
            self.voice_rewrites = not self.voice_rewrites
        elif self.validation_button.clicked(event):
            self.dialogue_validation = not self.dialogue_validation
        elif self.save_button.clicked(event):
            if self.ollama_model != app.ollama_model:
                try:
                    save_model(self.ollama_model)
                except OSError:
                    self.error = "Could not save your model choice. Check folder permissions and try again."
                    return
            try:
                save_settings(self.npc_trait_mode, self.voice_rewrites, self.dialogue_validation)
            except OSError:
                self.error = "Could not save your settings. Check folder permissions and try again."
                return
            app.ollama_model = self.ollama_model
            app.npc_trait_mode = self.npc_trait_mode
            app.voice_rewrites = self.voice_rewrites
            app.dialogue_validation = self.dialogue_validation
            app.provider = self.provider
            app.api_key = self.key_input.text.strip()
            self.key_input.set_value(app.api_key)
            self.key_input.active = False
            app.screen = "menu"
        elif self.provider == "gemini":
            self.key_input.handle_event(event)

    def draw(self, app):
        app.window.fill(GRASS)
        pygame.draw.rect(app.window, PAPER, (280, 95, 720, 575), border_radius=BORDER_RADIUS)
        draw_centered_text(app.window, "Settings", app.heading_font, 125)
        self.gemini_button.draw(app.window, app.font, enabled=self.provider != "gemini")
        self.ollama_button.draw(app.window, app.font, enabled=self.provider != "ollama")
        if self.provider == "gemini":
            app.window.blit(app.font.render("Gemini API key", True, INK), (340, 213))
            self.key_input.draw(app.window, app.font)
            app.window.blit(app.small_font.render("Session only. Get a key at aistudio.google.com/apikey", True, MUTED), (340, 301))
        else:
            index = self.models.index(self.ollama_model) + 1
            label = f"Local Ollama model ({index}/{len(self.models)})"
            app.window.blit(app.font.render(label, True, INK), (340, 213))
            self.previous_model_button.draw(app.window, app.font, enabled=len(self.models) > 1)
            self.next_model_button.draw(app.window, app.font, enabled=len(self.models) > 1)
            draw_wrapped_text(app.window, self.ollama_model, app.font, INK, (404, 246, 472, 44))
            size = self.installed_models.get(self.ollama_model)
            if size is None and ":" not in self.ollama_model:
                size = self.installed_models.get(f"{self.ollama_model}:latest")
            size_label = f"Size on disk: {format_model_size(size)}"
            app.window.blit(app.small_font.render(size_label, True, MUTED), (404, 302))
            self.refresh_button.draw(app.window, app.small_font, enabled=self.model_worker is None)
        app.window.blit(app.font.render("Random NPC traits", True, INK), (340, 336))
        for mode, button in self.trait_mode_buttons.items():
            button.draw(app.window, app.font)
            if mode == self.npc_trait_mode:
                pygame.draw.rect(app.window, INK, button.rect, 3, border_radius=BORDER_RADIUS)
        explanation = "Typical personalities and speech." if self.npc_trait_mode == "normal" else (
            "Unusually conspicuous personalities and speech." if self.npc_trait_mode == "extreme"
            else "Choose from both typical and conspicuous traits.")
        app.window.blit(app.small_font.render(explanation, True, MUTED), (340, 410))
        self.voice_rewrite_button.label = "Voice rewrites: " + ("On" if self.voice_rewrites else "Off")
        self.voice_rewrite_button.draw(app.window, app.font)
        app.window.blit(app.small_font.render("Add each speaker's style to make dialogue more interesting.", True, MUTED), (340, 480))
        self.validation_button.label = "Base dialogue validation: " + ("On" if self.dialogue_validation else "Off")
        self.validation_button.draw(app.window, app.font)
        app.window.blit(app.small_font.render("Check original dialogue for coherence. Voice rewrites stay as generated.", True, MUTED), (340, 548))
        self.save_button.draw(app.window, app.font)
        if self.provider == "ollama":
            self.setup_button.draw(app.window, app.font)
        if self.provider == "gemini":
            hint = self.key_input.error or "Ctrl+V to paste. Escape discards your edits."
        else:
            hint = self.model_message or "Use < and > to choose a model. Escape discards your edits."
            installed = (self.ollama_model in self.installed_models
                         or f"{self.ollama_model}:latest" in self.installed_models)
            if not self.model_message and not installed:
                hint = "Selected model is not installed. Download it in Ollama, then Refresh models."
        draw_wrapped_text(app.window, self.error or hint, app.small_font, INK, (280, 635, 720, 40))
