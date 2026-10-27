from pathlib import Path

import pygame

from character_data import load_character

from .button import Button
from .file_dialog import character_file_dialog
from .text import draw_wrapped_text
from constants import ACCENT, BORDER_RADIUS, GRASS, INK, MUTED, PAPER, STARTING_CHARACTER_FILE, STARTING_SPECTATOR_NPCS, STARTING_SPECTATOR_MIN_NPCS, MAX_RESIDENTS


class SetupScreen:
    ROWS_PER_PAGE = 4

    def __init__(self):
        # Judy occupies the fixed first slot; other slots are random or JSON paths.
        from constants import STARTING_MODE
        self.mode = STARTING_MODE
        self.mode_buttons = {mode: Button(rect, mode.title()) for mode, rect in zip(("player", "spectator"), ((260, 145, 365, 58), (635, 145, 385, 58)))}
        self.sources = [Path(__file__).resolve().parent.parent / STARTING_CHARACTER_FILE,
                        *[None] * (STARTING_SPECTATOR_NPCS - 1)]
        self.page = 0
        self.message = ""
        self.remove_button = Button((555, 245, 38, 34), "-")
        self.add_button = Button((650, 245, 38, 34), "+")
        self.load_npcs_button = Button((260, 553, 220, 34), "Import multiple NPCs")
        self.previous_button = Button((720, 553, 90, 34), "Previous")
        self.next_button = Button((930, 553, 90, 34), "Next")
        self.back_button = Button((260, 620, 140, 42), "Back")
        self.start_button = Button((820, 620, 200, 42), "Start game")
        self.random_buttons = [Button((715, 349 + row * 46, 130, 32), "Use random") for row in range(self.ROWS_PER_PAGE)]
        self.json_buttons = [Button((855, 349 + row * 46, 150, 32), "Choose file") for row in range(self.ROWS_PER_PAGE)]
        self.pending_mode = None
        self.cancel_mode_button = Button((450, 408, 160, 38), "Cancel")
        self.confirm_mode_button = Button((630, 408, 200, 38), "Switch to player")

    @property
    def max_npcs(self):
        return MAX_RESIDENTS - int(self.mode == "player")

    @property
    def last_page(self):
        return max(0, (len(self.sources) + int(self.mode == "player") - 1) // self.ROWS_PER_PAGE)

    def source_index(self, row):
        """Map a visible row to an NPC source; the player row has no source."""
        position = self.page * self.ROWS_PER_PAGE + row
        if self.mode == "player" and position == 1:
            return None
        return position - int(self.mode == "player" and position > 1)

    def set_count(self, count):
        count = min(self.max_npcs, max(STARTING_SPECTATOR_MIN_NPCS, count))
        if count < len(self.sources):
            del self.sources[count:]
        else:
            self.sources.extend([None] * (count - len(self.sources)))
        self.page = min(self.page, self.last_page)
        self.message = ""

    def load_characters(self, paths, index=None):
        if index == 0:
            raise ValueError("Judy Judge is required and cannot be replaced.")
        paths = [Path(path).resolve() for path in paths]
        if not paths:
            return
        for path in paths:
            try:
                load_character(path)
            except (OSError, ValueError) as error:
                raise ValueError(f"{path.name}: {error}") from error
        slots = ([index] if index is not None else []) + [
            slot for slot, source in enumerate(self.sources) if slot > 0 and source is None and slot != index]
        if len(self.sources) + max(0, len(paths) - len(slots)) > self.max_npcs:
            raise ValueError(f"This mode allows at most {self.max_npcs} NPCs.")
        first = slots[0] if slots else len(self.sources)
        replaced = min(len(slots), len(paths))
        for offset, path in enumerate(paths):
            if offset < len(slots):
                self.sources[slots[offset]] = path
            else:
                self.sources.append(path)
        self.page = (first + int(self.mode == "player" and first > 0)) // self.ROWS_PER_PAGE
        self.message = f"Loaded {len(paths)} NPCs: replaced {replaced} slots, added {len(paths) - replaced} slots."

    def choose_characters(self, index=None):
        try:
            paths = character_file_dialog(save=False, title="Select NPC JSON files", multiple=index is None)
            if index is not None:
                paths = [paths] if paths else []
            self.load_characters(paths, index)
        except (ImportError, OSError, ValueError) as error:
            self.message = f"Could not load NPCs: {error}"

    def handle_event(self, app, event):
        if self.pending_mode:
            if self.cancel_mode_button.clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.pending_mode = None
            elif self.confirm_mode_button.clicked(event):
                self.mode = self.pending_mode
                self.pending_mode = None
                self.set_count(self.max_npcs)
            return
        for mode, button in self.mode_buttons.items():
            if button.clicked(event):
                if mode == "player" and len(self.sources) > MAX_RESIDENTS - 1:
                    self.pending_mode = mode
                    return
                self.mode = mode
                if len(self.sources) > self.max_npcs:
                    self.set_count(self.max_npcs)
                self.page = min(self.page, self.last_page)
                self.message = ""
                return
        if self.back_button.clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            app.screen = "menu"
            return
        if self.start_button.clicked(event):
            app.start_game()
            return
        if self.remove_button.clicked(event, enabled=len(self.sources) > STARTING_SPECTATOR_MIN_NPCS):
            self.set_count(len(self.sources) - 1)
        elif self.add_button.clicked(event, enabled=len(self.sources) < self.max_npcs):
            self.set_count(len(self.sources) + 1)
        elif self.load_npcs_button.clicked(event):
            self.choose_characters()
            return
        elif self.previous_button.clicked(event, enabled=self.page > 0):
            self.page -= 1
        elif self.next_button.clicked(event, enabled=self.page < self.last_page):
            self.page += 1
        elif event.type == pygame.MOUSEWHEEL:
            self.page = max(0, min(self.last_page, self.page - event.y))

        for row, (random_button, json_button) in enumerate(zip(self.random_buttons, self.json_buttons)):
            index = self.source_index(row)
            if index is None:
                continue
            if index >= len(self.sources):
                break
            if index == 0:
                continue
            if random_button.clicked(event, enabled=self.sources[index] is not None):
                self.sources[index] = None
                self.message = ""
            elif json_button.clicked(event):
                self.choose_characters(index)

    def update(self, dt):
        pass

    @staticmethod
    def draw_secondary(surface, font, button, enabled=True):
        color = ACCENT if enabled else (152, 164, 151)
        if enabled and button.rect.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(surface, (225, 232, 210), button.rect)
        pygame.draw.rect(surface, color, button.rect, width=1)
        label = font.render(button.label, True, color)
        surface.blit(label, label.get_rect(center=button.rect.center))

    def draw(self, app):
        surface = app.window
        surface.fill(GRASS)
        pygame.draw.rect(surface, PAPER, (230, 45, 820, 630), border_radius=BORDER_RADIUS)
        surface.blit(app.heading_font.render("Set up your game", True, INK), (260, 67))
        surface.blit(app.font.render("1. How do you want to play?", True, INK), (260, 116))
        for mode, button in self.mode_buttons.items():
            selected = self.mode == mode
            pygame.draw.rect(surface, (225, 235, 216) if selected else (255, 254, 248), button.rect)
            pygame.draw.rect(surface, ACCENT if selected else (181, 191, 174), button.rect, width=2 if selected else 1)
            center = (button.rect.x + 20, button.rect.y + 20)
            pygame.draw.circle(surface, ACCENT if selected else MUTED, center, 7, 1)
            if selected:
                pygame.draw.circle(surface, ACCENT, center, 4)
            surface.blit(app.font.render(mode.title(), True, INK), (button.rect.x + 36, button.rect.y + 8))
            description = "Join the simulation" if mode == "player" else "Watch the simulation"
            surface.blit(app.small_font.render(description, True, MUTED), (button.rect.x + 36, button.rect.y + 32))

        surface.blit(app.font.render("2. Starting characters", True, INK), (260, 218))
        surface.blit(app.small_font.render("Number of characters", True, INK), (260, 253))
        self.draw_secondary(surface, app.font, self.remove_button, len(self.sources) > STARTING_SPECTATOR_MIN_NPCS)
        count = app.font.render(str(len(self.sources) + int(self.mode == "player")), True, INK)
        surface.blit(count, count.get_rect(center=(621, 262)))
        self.draw_secondary(surface, app.font, self.add_button, len(self.sources) < self.max_npcs)
        used = len(self.sources) + int(self.mode == "player")
        surface.blit(app.small_font.render(f"{used} of {MAX_RESIDENTS} spaces used", True, MUTED), (710, 247))
        note = "Includes you and Judy" if self.mode == "player" else "Includes Judy Judge"
        surface.blit(app.small_font.render(note, True, MUTED), (710, 265))

        surface.blit(app.font.render("3. Choose characters", True, INK), (260, 311))
        for row in range(self.ROWS_PER_PAGE):
            index = self.source_index(row)
            if index is not None and index >= len(self.sources):
                break
            y = 343 + row * 46
            pygame.draw.rect(surface, (255, 254, 248) if row % 2 == 0 else (242, 245, 232), (260, y, 760, 46))
            if index is None:
                surface.blit(app.font.render("You", True, INK), (272, y + 12))
                surface.blit(app.small_font.render("Player", True, MUTED), (930, y + 14))
                continue
            path = self.sources[index]
            if index == 0:
                surface.blit(app.font.render("Judy Judge", True, INK), (272, y + 12))
                surface.blit(app.small_font.render("Required", True, MUTED), (930, y + 14))
                continue
            surface.blit(app.small_font.render(f"NPC {index + 1}", True, INK), (272, y + 14))
            label = path.name if path is not None else "Random character"
            clip = surface.get_clip()
            surface.set_clip(pygame.Rect(355, y + 3, 350, 40).clip(clip))
            surface.blit(app.small_font.render(label, True, MUTED), (355, y + 14))
            surface.set_clip(clip)
            if path is not None:
                self.draw_secondary(surface, app.small_font, self.random_buttons[row])
            self.json_buttons[row].label = "Change file" if path is not None else "Choose file"
            self.draw_secondary(surface, app.small_font, self.json_buttons[row])
        self.draw_secondary(surface, app.small_font, self.load_npcs_button)
        if self.last_page > 0:
            self.draw_secondary(surface, app.small_font, self.previous_button, self.page > 0)
            self.draw_secondary(surface, app.small_font, self.next_button, self.page < self.last_page)
            label = app.small_font.render(f"{self.page + 1} / {self.last_page + 1}", True, MUTED)
            surface.blit(label, label.get_rect(center=(870, 570)))
        hint = self.message or "Fills random slots, then adds slots if needed."
        draw_wrapped_text(surface, hint, app.small_font, MUTED, (260, 592, 760, 25))
        self.draw_secondary(surface, app.font, self.back_button)
        self.start_button.draw(surface, app.font)
        if self.pending_mode:
            overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            surface.blit(overlay, (0, 0))
            pygame.draw.rect(surface, PAPER, (410, 260, 460, 205))
            surface.blit(app.font.render("Switch to player mode?", True, INK), (440, 285))
            draw_wrapped_text(surface, "You take one of the 10 spaces in player mode. The last NPC slot will be removed.",
                              app.small_font, INK, (440, 325, 400, 65))
            self.draw_secondary(surface, app.small_font, self.cancel_mode_button)
            self.confirm_mode_button.draw(surface, app.small_font)
