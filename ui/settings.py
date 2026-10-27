import queue
import threading

import pygame

from app_settings import save_settings
from character_data import NPC_TRAIT_MODES
from ollama_manager import DEFAULT_OLLAMA_MODEL, list_models, save_model

from .button import Button
from .text import draw_wrapped_text
from .text_input import TextInput
from constants import ACCENT, BORDER_RADIUS, GRASS, INK, MAX_CONFIGURABLE_LLM_REQUESTS, MAX_LLM_REQUESTS, MUTED, PAPER


def format_model_size(size):
    if size is None:
        return "unavailable"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1000


class SettingsScreen:
    def __init__(self):
        # Content controls use coordinates relative to this area; the header
        # and footer use window coordinates.
        self.content_rect = pygame.Rect(260, 116, 760, 478)
        self.gemini_button = Button((0, 28, 366, 74), "Gemini - Online")
        self.ollama_button = Button((378, 28, 366, 74), "Ollama - On this computer")
        self.previous_model_button = Button((0, 138, 38, 36), "<")
        self.next_model_button = Button((392, 138, 38, 36), ">")
        self.refresh_button = Button((454, 109, 138, 26), "Refresh models")
        self.parallel_less = Button((626, 145, 32, 32), "-")
        self.parallel_more = Button((712, 145, 32, 32), "+")
        self.cancel_button = Button((260, 638, 140, 36), "Cancel")
        self.save_button = Button((820, 638, 200, 36), "Save & Back")
        self.setup_button = Button((604, 109, 140, 26), "Setup Ollama")
        self.voice_rewrite_button = Button((626, 382, 118, 34), "")
        self.validation_button = Button((626, 434, 118, 34), "")
        self.key_input = TextInput(
            (0, 138, 744, 36),
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
            mode: Button((index * 252, 279, 240, 68), "Mixed" if mode == "both" else mode.title())
            for index, mode in enumerate(NPC_TRAIT_MODES)
        }
        self.ollama_model = DEFAULT_OLLAMA_MODEL
        self.models = [DEFAULT_OLLAMA_MODEL]
        self.installed_models = {}
        self.model_worker = None
        self.model_results = queue.Queue()
        self.model_message = ""
        self.parallel_limit = MAX_LLM_REQUESTS
        self.error = ""

    def open(self, provider, api_key, ollama_model, npc_trait_mode="normal", *,
             voice_rewrites=True, dialogue_validation=True, max_requests=MAX_LLM_REQUESTS):
        self.provider = provider
        self.ollama_model = ollama_model
        self.npc_trait_mode = npc_trait_mode
        self.voice_rewrites = voice_rewrites
        self.dialogue_validation = dialogue_validation
        self.parallel_limit = max_requests
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
                models, error = list_models(), ""
            except ValueError as exc:
                models, error = {}, str(exc)
            self.model_results.put((models, error))

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

    def _cancel(self, app):
        self.provider = app.provider
        self.ollama_model = app.ollama_model
        self.npc_trait_mode = app.npc_trait_mode
        self.voice_rewrites = app.voice_rewrites
        self.dialogue_validation = app.dialogue_validation
        self.parallel_limit = app.requests.max_requests
        self.key_input.set_value(app.api_key)
        self.key_input.active = False
        self.error = ""
        app.screen = "menu"

    def handle_event(self, app, event):
        if self.cancel_button.clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            self._cancel(app)
            return
        if self.save_button.clicked(event):
            self._save(app)
            return
        if hasattr(event, "pos"):
            if not self.content_rect.collidepoint(event.pos):
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.key_input.active = False
                return
            event = pygame.event.Event(event.type, {
                **event.dict,
                "pos": (event.pos[0] - self.content_rect.x,
                        event.pos[1] - self.content_rect.y),
            })
        if self.provider == "gemini":
            self.key_input.handle_event(event)
        if self.gemini_button.clicked(event):
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
        elif self.provider == "ollama" and self.parallel_less.clicked(event):
            self.parallel_limit = max(1, self.parallel_limit - 1)
            self.error = ""
        elif self.provider == "ollama" and self.parallel_more.clicked(event):
            self.parallel_limit = min(MAX_CONFIGURABLE_LLM_REQUESTS, self.parallel_limit + 1)
            self.error = ""
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

    def _save(self, app):
        if self.ollama_model != app.ollama_model:
            try:
                save_model(self.ollama_model)
            except OSError:
                self.error = "Could not save your model choice. Check folder permissions and try again."
                return
        try:
            save_settings(self.npc_trait_mode, self.voice_rewrites, self.dialogue_validation,
                          max_llm_requests=self.parallel_limit)
        except OSError:
            self.error = "Could not save your settings. Check folder permissions and try again."
            return
        app.ollama_model = self.ollama_model
        app.npc_trait_mode = self.npc_trait_mode
        app.voice_rewrites = self.voice_rewrites
        app.dialogue_validation = self.dialogue_validation
        app.provider = self.provider
        app.api_key = self.key_input.text.strip()
        app.requests.max_requests = self.parallel_limit
        app.game.request_status.set_capacity(self.parallel_limit)
        self.key_input.set_value(app.api_key)
        self.key_input.active = False
        app.screen = "menu"

    def _content_mouse(self):
        x, y = pygame.mouse.get_pos()
        if not self.content_rect.collidepoint(x, y):
            return (-1, -1)
        return (x - self.content_rect.x, y - self.content_rect.y)

    def _draw_secondary(self, surface, font, button, enabled=True, *, content=True):
        mouse = self._content_mouse() if content else pygame.mouse.get_pos()
        color = ACCENT if enabled else (152, 164, 151)
        if enabled and button.rect.collidepoint(mouse):
            pygame.draw.rect(surface, (225, 232, 210), button.rect)
        pygame.draw.rect(surface, color, button.rect, width=1)
        label = font.render(button.label, True, color)
        surface.blit(label, label.get_rect(center=button.rect.center))

    @staticmethod
    def _draw_choice(surface, app, button, selected, description):
        rect = button.rect
        pygame.draw.rect(surface, (225, 235, 216) if selected else (255, 254, 248), rect)
        pygame.draw.rect(surface, ACCENT if selected else (181, 191, 174), rect,
                         width=2 if selected else 1)
        center = (rect.x + 18, rect.y + 18)
        pygame.draw.circle(surface, ACCENT if selected else MUTED, center, 7, 1)
        if selected:
            pygame.draw.circle(surface, ACCENT, center, 4)
        surface.blit(app.font.render(button.label, True, INK), (rect.x + 34, rect.y + 8))
        draw_wrapped_text(surface, description, app.small_font, MUTED,
                          (rect.x + 34, rect.y + 32, rect.width - 44, rect.height - 34))

    def _draw_toggle(self, surface, app, button, value):
        button.label = "On" if value else "Off"
        pygame.draw.rect(surface, (225, 235, 216) if value else (255, 254, 248), button.rect)
        self._draw_secondary(surface, app.font, button)
        center = (button.rect.x + 20, button.rect.centery)
        pygame.draw.circle(surface, ACCENT if value else MUTED, center, 5, 0 if value else 1)

    def draw(self, app):
        surface = app.window
        surface.fill(GRASS)
        pygame.draw.rect(surface, PAPER, (230, 32, 820, 656), border_radius=BORDER_RADIUS)
        surface.blit(app.heading_font.render("Settings", True, INK), (260, 52))
        surface.blit(app.small_font.render("Set up dialogue and choose how your characters feel.", True, MUTED), (260, 87))
        content = pygame.Surface((744, self.content_rect.height))
        content.fill(PAPER)
        content.blit(app.font.render("1. Choose how dialogue is generated", True, INK), (0, 0))
        self._draw_choice(content, app, self.gemini_button, self.provider == "gemini",
                          "Generate dialogue with Google's AI. Requires an API key.")
        self._draw_choice(content, app, self.ollama_button, self.provider == "ollama",
                          "Generate dialogue with a locally installed model.")
        if self.provider == "gemini":
            content.blit(app.font.render("Gemini API key", True, INK), (0, 114))
            self.key_input.draw(content, app.font)
            # The input is drawn in content coordinates, but the OS needs its
            # window position for text composition and the on-screen keyboard.
            if self.key_input.active:
                pygame.key.set_text_input_rect(self.key_input.rect.move(
                    self.content_rect.x, self.content_rect.y))
            draw_wrapped_text(content,
                              "Used for this session only. Get a key at aistudio.google.com/apikey. "
                              "Ctrl+V or Cmd+V to paste.", app.small_font, MUTED, (0, 181, 744, 32))
            draw_wrapped_text(content, self.key_input.error, app.small_font, INK, (0, 216, 744, 32))
        else:
            index = self.models.index(self.ollama_model) + 1
            content.blit(app.font.render(f"Local model ({index}/{len(self.models)})", True, INK), (0, 114))
            self._draw_secondary(content, app.small_font, self.refresh_button, self.model_worker is None)
            self._draw_secondary(content, app.small_font, self.setup_button)
            self._draw_secondary(content, app.font, self.previous_model_button, len(self.models) > 1)
            self._draw_secondary(content, app.font, self.next_model_button, len(self.models) > 1)
            draw_wrapped_text(content, self.ollama_model, app.small_font, INK, (50, 140, 330, 32))
            size = self.installed_models.get(self.ollama_model)
            if size is None and ":" not in self.ollama_model:
                size = self.installed_models.get(f"{self.ollama_model}:latest")
            content.blit(app.small_font.render(f"Size on disk: {format_model_size(size)}", True, MUTED), (0, 182))
            installed = (self.ollama_model in self.installed_models
                         or f"{self.ollama_model}:latest" in self.installed_models)
            hint = self.model_message
            if not hint and not installed:
                hint = "Selected model is not installed. Use Setup Ollama to get started, then Refresh models."
            draw_wrapped_text(content, hint, app.small_font, MUTED, (0, 210, 430, 32))
            content.blit(app.font.render("Ollama threads", True, INK), (454, 152))
            self._draw_secondary(content, app.font, self.parallel_less, self.parallel_limit > 1)
            self._draw_secondary(content, app.font, self.parallel_more,
                                 self.parallel_limit < MAX_CONFIGURABLE_LLM_REQUESTS)
            value = app.font.render(str(self.parallel_limit), True, INK)
            content.blit(value, value.get_rect(center=(685, 161)))
            draw_wrapped_text(content,
                              f"Parallel dialogue requests. Set OLLAMA_NUM_PARALLEL to at least "
                              f"{self.parallel_limit}, then restart Ollama.",
                              app.small_font, MUTED, (454, 185, 290, 64))

        content.blit(app.font.render("2. Personalities for random characters", True, INK), (0, 252))
        descriptions = {
            "normal": "Typical personalities and speech.",
            "extreme": "More exaggerated traits and speech.",
            "both": "A combination of normal and extreme.",
        }
        for mode, button in self.trait_mode_buttons.items():
            self._draw_choice(content, app, button, mode == self.npc_trait_mode, descriptions[mode])

        content.blit(app.font.render("3. Adjust dialogue", True, INK), (0, 362))
        content.blit(app.font.render("Distinctive character voices", True, INK), (0, 384))
        content.blit(app.small_font.render("Give each speaker their own speaking style.", True, MUTED), (0, 407))
        self._draw_toggle(content, app, self.voice_rewrite_button, self.voice_rewrites)
        content.blit(app.font.render("Base dialogue validation", True, INK), (0, 438))
        self._draw_toggle(content, app, self.validation_button, self.dialogue_validation)
        draw_wrapped_text(content, "Off: show validation errors in red; dialogue and outcomes still apply.",
                          app.small_font, MUTED, (0, 462, 744, 16))

        surface.blit(content, self.content_rect.topleft)
        pygame.draw.line(surface, (181, 191, 174), (260, 600), (1020, 600))
        hint = self.error or "Save to apply your changes. Cancel or Escape discards your edits."
        draw_wrapped_text(surface, hint, app.small_font, INK if self.error else MUTED, (260, 603, 760, 32))
        self._draw_secondary(surface, app.font, self.cancel_button, content=False)
        self.save_button.draw(surface, app.font)
