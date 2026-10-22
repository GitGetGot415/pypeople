from pathlib import Path

import pygame

from character_data import load_character

from .button import Button
from .file_dialog import character_file_dialog
from .text import draw_centered_text, draw_wrapped_text
from .text_input import TextInput
from constants import ACCENT, BORDER_RADIUS, GRASS, INK, MUTED, PAPER, STARTING_ADDITIONAL_NPCS, SETUP_LOAD_NPCS_RECT


class SetupScreen:
    ROWS_PER_PAGE = 4

    def __init__(self):
        # None means random; a Path means a character loaded from JSON.
        from constants import SETUP_MODE_RECTS, STARTING_MODE
        self.mode = STARTING_MODE
        self.mode_buttons = {mode: Button(rect, mode.title()) for mode, rect in zip(("player", "spectator"), SETUP_MODE_RECTS)}
        self.sources = [None] * STARTING_ADDITIONAL_NPCS
        self.page = 0
        self.message = ""
        self.count_input = TextInput((590, 138, 100, 40), max_length=None)
        self.count_input.set_value(str(len(self.sources)))
        self.remove_button = Button((540, 138, 40, 40), "-")
        self.add_button = Button((700, 138, 40, 40), "+")
        self.apply_button = Button((755, 138, 145, 40), "Set count")
        self.load_npcs_button = Button(SETUP_LOAD_NPCS_RECT, "Load NPC JSONs")
        self.previous_button = Button((260, 525, 120, 32), "Previous")
        self.next_button = Button((900, 525, 120, 32), "Next")
        self.back_button = Button((260, 605, 160, 44), "Back")
        self.start_button = Button((820, 605, 200, 44), "Start game")
        self.random_buttons = [Button((650, 258 + row * 65, 150, 38), "Random") for row in range(self.ROWS_PER_PAGE)]
        self.json_buttons = [Button((815, 258 + row * 65, 185, 38), "Choose JSON") for row in range(self.ROWS_PER_PAGE)]

    @property
    def last_page(self):
        return max(0, (len(self.sources) - 1) // self.ROWS_PER_PAGE)

    def set_count(self, count):
        if count < len(self.sources):
            del self.sources[count:]
        else:
            self.sources.extend([None] * (count - len(self.sources)))
        self.page = min(self.page, self.last_page)
        self.count_input.set_value(str(count))
        self.message = ""

    def apply_count(self):
        try:
            value = self.count_input.text.strip()
            if not value.isdecimal():
                raise ValueError
            count = int(value)
        except ValueError:
            self.message = "Enter a whole number of additional NPCs (0 or more)."
            return False
        self.set_count(count)
        return True

    def load_characters(self, paths, index=None):
        paths = [Path(path).resolve() for path in paths]
        if not paths:
            return
        for path in paths:
            try:
                load_character(path)
            except (OSError, ValueError) as error:
                raise ValueError(f"{path.name}: {error}") from error
        slots = ([index] if index is not None else []) + [
            slot for slot, source in enumerate(self.sources) if source is None and slot != index]
        first = slots[0] if slots else len(self.sources)
        replaced = min(len(slots), len(paths))
        for offset, path in enumerate(paths):
            if offset < len(slots):
                self.sources[slots[offset]] = path
            else:
                self.sources.append(path)
        self.count_input.set_value(str(len(self.sources)))
        self.page = first // self.ROWS_PER_PAGE
        self.message = f"Loaded {len(paths)} NPCs: replaced {replaced} slots, added {len(paths) - replaced} slots."

    def choose_characters(self, index=None):
        try:
            paths = character_file_dialog(save=False, title="Select NPC JSON files", multiple=True)
            self.load_characters(paths, index)
        except (ImportError, OSError, ValueError) as error:
            self.message = f"Could not load NPCs: {error}"

    def handle_event(self, app, event):
        for mode, button in self.mode_buttons.items():
            if button.clicked(event):
                self.mode = mode
                return
        if self.back_button.clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            self.count_input.active = False
            app.screen = "menu"
            return
        if self.count_input.handle_event(event) or self.apply_button.clicked(event):
            self.apply_count()
            return
        if self.remove_button.clicked(event, enabled=bool(self.sources)):
            self.set_count(len(self.sources) - 1)
        elif self.add_button.clicked(event):
            self.set_count(len(self.sources) + 1)
        elif self.load_npcs_button.clicked(event):
            self.choose_characters()
            return
        elif self.previous_button.clicked(event, enabled=self.page > 0):
            self.page -= 1
        elif self.next_button.clicked(event, enabled=self.page < self.last_page):
            self.page += 1
        elif self.start_button.clicked(event):
            if self.apply_count():
                app.start_game()
            return
        elif event.type == pygame.MOUSEWHEEL:
            self.page = max(0, min(self.last_page, self.page - event.y))

        for row, (random_button, json_button) in enumerate(zip(self.random_buttons, self.json_buttons)):
            index = self.page * self.ROWS_PER_PAGE + row
            if index >= len(self.sources):
                break
            if random_button.clicked(event):
                self.sources[index] = None
                self.message = ""
            elif json_button.clicked(event):
                self.choose_characters(index)

    def update(self, dt):
        self.count_input.update(dt)

    def draw(self, app):
        app.window.fill(GRASS)
        pygame.draw.rect(app.window, PAPER, (230, 65, 820, 600), border_radius=BORDER_RADIUS)
        draw_centered_text(app.window, "Set up your game", app.heading_font, 88)
        app.window.blit(app.font.render("Additional NPCs", True, INK), (260, 148))
        self.count_input.draw(app.window, app.font)
        self.remove_button.draw(app.window, app.font, enabled=bool(self.sources))
        self.add_button.draw(app.window, app.font)
        self.apply_button.draw(app.window, app.font)
        self.load_npcs_button.draw(app.window, app.small_font)
        app.window.blit(app.small_font.render(f"Total: {len(self.sources) + 1} NPCs, including Judy", True, MUTED), (260, 188))
        pygame.draw.rect(app.window, (225, 232, 210), (260, 215, 760, 32))
        app.window.blit(app.font.render("NPC 1: judy.json (fixed)", True, INK), (272, 220))

        for row in range(self.ROWS_PER_PAGE):
            index = self.page * self.ROWS_PER_PAGE + row
            if index >= len(self.sources):
                break
            path = self.sources[index]
            y = 258 + row * 65
            app.window.blit(app.font.render(f"NPC {index + 2}", True, INK), (272, y))
            label = path.name if path is not None else "Randomized when the game starts"
            draw_wrapped_text(app.window, label, app.small_font, MUTED, (272, y + 24, 360, 38))
            self.random_buttons[row].draw(app.window, app.font)
            self.json_buttons[row].draw(app.window, app.font)
            selected = self.random_buttons[row] if path is None else self.json_buttons[row]
            pygame.draw.rect(app.window, ACCENT, selected.rect.inflate(6, 6), width=2)
        if not self.sources:
            draw_centered_text(app.window, "Only Judy will be in this game.", app.font, 340, MUTED)
        self.previous_button.draw(app.window, app.small_font, enabled=self.page > 0)
        self.next_button.draw(app.window, app.small_font, enabled=self.page < self.last_page)
        draw_centered_text(app.window, f"Page {self.page + 1} of {self.last_page + 1}", app.small_font, 532, MUTED)
        draw_wrapped_text(app.window, self.message, app.small_font, MUTED, (260, 564, 760, 38))
        for mode, button in self.mode_buttons.items():
            button.draw(app.window, app.small_font)
            if self.mode == mode:
                pygame.draw.rect(app.window, ACCENT, button.rect.inflate(4, 4), width=2)
        self.back_button.draw(app.window, app.font)
        self.start_button.draw(app.window, app.font)
