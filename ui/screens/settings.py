import queue
import threading
import pygame
from app_settings import save_settings
from persistence.characters import NPC_TRAIT_MODES
from ai.ollama import DEFAULT_OLLAMA_MODEL, list_models
from ai.online import OnlineConfig, PROVIDERS, load_profiles
from ai.transport import check_online_connection
from dialogue.errors import RequestError
from ui.widgets.button import Button
from ui.text import draw_wrapped_text
from ui.widgets.text_input import TextInput
from constants import (
    ACCENT, BORDER_RADIUS, GRASS, INK, MAX_CONFIGURABLE_LLM_REQUESTS, MAX_LLM_REQUESTS, MUTED, PAPER,
)


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
        self.online_button = Button((0, 28, 366, 74), "Online - API key")
        self.ollama_button = Button((378, 28, 366, 74), "Ollama - On this computer")
        self.provider_button = Button((76, 110, 294, 34), "")
        self.provider_choices = {
            provider: Button((76, 146 + index * 34, 294, 34), label)
            for index, (provider, (label, _, _, _)) in enumerate(PROVIDERS.items())
        }
        self.provider_menu_open = False
        self.test_button = Button((572, 152, 172, 36), "Test connection")
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
            (76, 152, 480, 36),
            max_length=4096,
            secret=True,
            strip_paste=True,
            placeholder="Paste your key here",
        )
        self.model_input = TextInput((454, 110, 290, 34), max_length=256, strip_paste=True,
                                     placeholder="Type a model ID")
        self.base_url_input = TextInput((76, 194, 668, 36), max_length=2048, strip_paste=True,
                                        placeholder="https://your-provider.example/v1")
        self.provider = "online"
        self.online_provider = "gemini"
        self.online_profiles = load_profiles(None)
        self.online_keys = {provider: "" for provider in PROVIDERS}
        self.connection_worker = None
        self.connection_results = queue.Queue()
        self.connection_message = ""
        self.connection_attempt = None
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

    def open(self, app):
        self.provider = app.provider
        self.online_provider = app.online_provider
        self.online_profiles = app.online_profiles.copy()
        self.online_keys = app.online_keys.copy()
        self.ollama_model = app.ollama_model
        self.npc_trait_mode = app.npc_trait_mode
        self.voice_rewrites = app.voice_rewrites
        self.dialogue_validation = app.dialogue_validation
        self.parallel_limit = app.requests.max_requests
        self.error = ""
        self.connection_message = ""
        self.connection_attempt = None
        self.provider_menu_open = False
        self.update_model_choices()
        self._load_online_fields()
        if self.provider == "online":
            self.key_input.focus()
        else:
            self.key_input.active = False
            self.refresh_models()

    def _online_inputs(self):
        fields = [self.model_input, self.key_input]
        if self.online_provider == "custom":
            fields.append(self.base_url_input)
        return fields

    def _blur_inputs(self):
        for field in (self.model_input, self.key_input, self.base_url_input):
            field.active = False

    def _current_online_config(self):
        return OnlineConfig(self.online_provider, self.model_input.text.strip(),
                            self.base_url_input.text.strip().rstrip("/"))

    def _store_online_fields(self):
        self.online_profiles[self.online_provider] = self._current_online_config()
        self.online_keys[self.online_provider] = self.key_input.text.strip()

    def _load_online_fields(self):
        config = self.online_profiles[self.online_provider]
        self._blur_inputs()
        self.model_input.set_value(config.model)
        self.base_url_input.set_value(config.base_url)
        self.key_input.set_value(self.online_keys[self.online_provider])

    def _test_connection(self):
        if self.connection_worker is not None:
            return
        config, key = self._current_online_config(), self.key_input.text.strip()
        try:
            config.validate()
            if not key:
                raise ValueError("Enter an API key to test the connection.")
        except ValueError as exc:
            self.error = str(exc)
            return
        self.error = ""
        self.connection_message = "Testing a small JSON dialogue reply..."
        attempt = object()
        self.connection_attempt = attempt

        def check():
            try:
                message = check_online_connection(key, config)
            except RequestError as exc:
                message = str(exc)
            except Exception:
                message = "Could not test the connection. Check your settings and try again."
            self.connection_results.put((attempt, config, key, message))

        self.connection_worker = threading.Thread(target=check, daemon=True, name="Online-connection-check")
        self.connection_worker.start()

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

    def update(self, dt=0):
        if self.provider == "online":
            for field in self._online_inputs():
                field.update(dt)
        try:
            attempt, config, key, message = self.connection_results.get_nowait()
        except queue.Empty:
            pass
        else:
            self.connection_worker = None
            if (attempt is self.connection_attempt and self.provider == "online"
                    and config == self._current_online_config() and key == self.key_input.text.strip()):
                self.connection_message = message
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
        self.open(app)
        self._blur_inputs()
        app.screen = "menu"

    def handle_event(self, app, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and self.provider_menu_open:
            self.provider_menu_open = False
            return
        if self.cancel_button.clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            self._cancel(app)
            return
        if self.save_button.clicked(event):
            self._save(app)
            return
        if hasattr(event, "pos"):
            if not self.content_rect.collidepoint(event.pos):
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self._blur_inputs()
                    self.provider_menu_open = False
                return
            event = pygame.event.Event(event.type, {
                **event.dict,
                "pos": (event.pos[0] - self.content_rect.x,
                        event.pos[1] - self.content_rect.y),
            })
        if self.provider_menu_open:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                selected = next((provider for provider, button in self.provider_choices.items()
                                 if button.clicked(event)), None)
                self.provider_menu_open = False
                if selected is not None:
                    self._store_online_fields()
                    self.online_provider = selected
                    self._load_online_fields()
                    self.connection_message = ""
                    self.error = ""
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                self.provider_menu_open = False
            else:
                return
        if self.provider == "online":
            fields = self._online_inputs()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                active = next((index for index, field in enumerate(fields) if field.active), -1)
                self._blur_inputs()
                fields[(active + (-1 if event.mod & pygame.KMOD_SHIFT else 1)) % len(fields)].focus()
                return
            before = (self._current_online_config(), self.key_input.text)
            for field in fields:
                field.handle_event(event)
            if before != (self._current_online_config(), self.key_input.text):
                self.connection_message = ""
                self.connection_attempt = None
                self.error = ""
        if self.online_button.clicked(event):
            self.provider = "online"
            self._blur_inputs()
            self.key_input.focus()
        elif self.ollama_button.clicked(event):
            self.provider = "ollama"
            self._blur_inputs()
            self.connection_message = ""
            self.refresh_models()
        elif self.provider == "online" and self.provider_button.clicked(event):
            self.provider_menu_open = True
            self._blur_inputs()
        elif self.provider == "online" and self.test_button.clicked(event):
            self._test_connection()
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
        self._store_online_fields()
        if self.provider == "online":
            try:
                self.online_profiles[self.online_provider].validate()
            except ValueError as exc:
                self.error = str(exc)
                return
        try:
            save_settings(self.npc_trait_mode, self.voice_rewrites, self.dialogue_validation,
                          max_llm_requests=self.parallel_limit, provider=self.provider, ollama_model=self.ollama_model,
                          online_provider=self.online_provider, online_profiles=self.online_profiles)
        except OSError:
            self.error = "Could not save your settings. Check folder permissions and try again."
            return
        app.ollama_model = self.ollama_model
        app.npc_trait_mode = self.npc_trait_mode
        app.voice_rewrites = self.voice_rewrites
        app.dialogue_validation = self.dialogue_validation
        app.provider = self.provider
        app.online_provider = self.online_provider
        app.online_profiles = self.online_profiles.copy()
        app.online_keys = self.online_keys.copy()
        app.requests.max_requests = self.parallel_limit
        app.game.request_status.set_capacity(self.parallel_limit)
        self._blur_inputs()
        self.connection_attempt = None
        self.provider_menu_open = False
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
        self._draw_choice(content, app, self.online_button, self.provider == "online",
                          "Choose an online service, model, and API key.")
        self._draw_choice(content, app, self.ollama_button, self.provider == "ollama",
                          "Generate dialogue with a locally installed model.")
        if self.provider == "online":
            content.blit(app.small_font.render("Provider", True, INK), (0, 119))
            content.blit(app.small_font.render("Model", True, INK), (390, 119))
            self.provider_button.label = PROVIDERS[self.online_provider][0] + "  v"
            self._draw_secondary(content, app.small_font, self.provider_button)
            content.blit(app.small_font.render("API key", True, INK), (0, 162))
            for field in self._online_inputs():
                field.draw(content, app.small_font)
                # Controls use content coordinates; IME placement uses window coordinates.
                if field.active:
                    pygame.key.set_text_input_rect(field.rect.move(self.content_rect.topleft))
            self._draw_secondary(content, app.small_font, self.test_button, self.connection_worker is None)
            if self.online_provider == "custom":
                content.blit(app.small_font.render("Base URL", True, INK), (0, 204))
                hint = "Use your service's OpenAI-compatible API base URL. Keys are used for this session only."
                hint_rect = (0, 234, 744, 16)
            else:
                hint = "Keys are used for this session only. Type the model ID to choose another model."
                if self.online_provider == "gemini":
                    hint += " Get a key at aistudio.google.com/apikey."
                elif self.online_provider == "anthropic":
                    hint += " Get a key at platform.claude.com."
                else:
                    hint += " Get a key from your OpenAI API account."
                hint_rect = (0, 196, 744, 48)
            draw_wrapped_text(content, hint, app.small_font, MUTED, hint_rect)
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

        if self.provider_menu_open and self.provider == "online":
            for provider, button in self.provider_choices.items():
                pygame.draw.rect(content, (225, 235, 216) if provider == self.online_provider else PAPER,
                                 button.rect)
                self._draw_secondary(content, app.small_font, button)

        surface.blit(content, self.content_rect.topleft)
        pygame.draw.line(surface, (181, 191, 174), (260, 600), (1020, 600))
        input_error = next((field.error for field in self._online_inputs() if field.error), "") if self.provider == "online" else ""
        hint = self.error or input_error or self.connection_message or "Save to apply your changes. Cancel or Escape discards your edits."
        draw_wrapped_text(surface, hint, app.small_font, INK if self.error else MUTED, (260, 603, 760, 32))
        self._draw_secondary(surface, app.font, self.cancel_button, content=False)
        self.save_button.draw(surface, app.font)
