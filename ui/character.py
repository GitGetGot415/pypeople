import json
import random
from pathlib import Path

import pygame

from character_data import DEFAULT_COLORS, DIRECTIONS, FIELDS, load_character, parse_color

from .button import Button
from .character_sprite import CharacterRenderer
from .slider import ColorSlider
from .text import draw_centered_text
from .text_input import TextInput
from .theme import BORDER_RADIUS, GRASS, INK, MUTED, PAPER


class CharacterScreen:
    FIELDS = FIELDS
    DEFAULT_COLORS = DEFAULT_COLORS
    COLOR_GRID = ("Eye Color", "Hair Color", "Skin Color", "Shirt Color")
    DIRECTIONS = DIRECTIONS

    def __init__(self):
        self.values = {name: "" for name in self.FIELDS}
        self.values.update(self.DEFAULT_COLORS)
        self.category = "Stats"
        self.inputs = {}
        for index, name in enumerate(self.FIELDS):
            self.inputs[name] = TextInput(
                (550, 210 + index * 58, 300, 42),
                max_length=100,
                strip_paste=True,
                placeholder=f"Enter {name.lower()}",
            )
            self.inputs[name].set_value(self.values[name])
        self.tab_buttons = {
            "Stats": Button((430, 140, 200, 44), "Stats"),
            "Color": Button((650, 140, 200, 44), "Color"),
        }
        self.color_sliders = {}
        self.color_panels = {}
        for index, name in enumerate(self.COLOR_GRID):
            panel = pygame.Rect(430 + index % 2 * 220, 200 + index // 2 * 158, 200, 148)
            self.color_panels[name] = panel
            rgb = self._parse_color(self.DEFAULT_COLORS[name])
            self.color_sliders[name] = [
                ColorSlider(
                    (panel.x + 12, panel.y + 64 + channel * 26, panel.width - 24, 24),
                    label, rgb[channel],
                )
                for channel, label in enumerate(("R", "G", "B"))
            ]
        self.back_button = Button((80, 545, 150, 48), "Back")
        self.random_button = Button((250, 545, 150, 48), "Random")
        self.import_button = Button((570, 545, 130, 48), "Import JSON")
        self.export_button = Button((720, 545, 130, 48), "Export JSON")
        self.turn_left_button = Button((96, 473, 120, 30), "Turn left")
        self.turn_right_button = Button((244, 473, 120, 30), "Turn right")
        self.preview_area = pygame.Rect(96, 206, 268, 260)
        self.direction = 4  # Start facing south, with both hands visible.
        self.message = ""

        asset_dir = Path(__file__).resolve().parent.parent / "assets"
        self.random_config_path = asset_dir / "character_randomization.json"
        self.renderer = CharacterRenderer()
        self.preview = None

    _parse_color = staticmethod(parse_color)

    def _sync_values(self):
        for name, text_input in self.inputs.items():
            self.values[name] = text_input.text
        for name, sliders in self.color_sliders.items():
            self.values[name] = "#" + "".join(f"{slider.value:02X}" for slider in sliders)

    def _clear_focus(self):
        for text_input in self.inputs.values():
            text_input.active = False
            text_input.backspace_held = False
        for sliders in self.color_sliders.values():
            for slider in sliders:
                slider.active = False
                slider.dragging = False

    def _randomize(self):
        self._clear_focus()
        try:
            with self.random_config_path.open(encoding="utf-8") as config_file:
                config = json.load(config_file)
            if not isinstance(config, dict):
                raise ValueError("Random settings must be a JSON object.")
            for key in ("first_names", "last_names"):
                names = config.get(key)
                if not isinstance(names, list) or not names or any(
                    not isinstance(name, str) or not name.strip() for name in names
                ):
                    raise ValueError(f"{key} must be a nonempty list of names.")
            limits = config.get("limits")
            if not isinstance(limits, dict):
                raise ValueError("Random settings must include numeric limits.")
            values = {}
            for name in ("Age", "Weight", "Height"):
                bounds = limits.get(name)
                if not isinstance(bounds, dict):
                    raise ValueError(f"{name} needs min and max limits.")
                low, high = bounds.get("min"), bounds.get("max")
                if type(low) is not int or type(high) is not int or low > high:
                    raise ValueError(f"{name} needs integer limits with min <= max.")
                values[name] = str(random.randint(low, high))
            values["Name"] = f"{random.choice(config['first_names']).strip()} {random.choice(config['last_names']).strip()}"
            values["Sex"] = random.choice(("M", "F"))
            # Validate and generate the stats before replacing the current edit.
            for name, value in values.items():
                self.inputs[name].set_value(value)
            for sliders in self.color_sliders.values():
                for slider in sliders:
                    slider.value = random.randint(0, 255)
            self._sync_values()
            self.message = "Character randomized."
        except (OSError, ValueError) as error:
            self.message = f"Could not randomize character: {error}"

    def _file_dialog(self, save):
        import tkinter as tk
        from tkinter import filedialog

        try:
            root = tk.Tk()
        except tk.TclError as error:
            raise OSError("File dialog is unavailable.") from error
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            try:
                if save:
                    return filedialog.asksaveasfilename(
                        parent=root,
                        title="Export character",
                        defaultextension=".json",
                        filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
                        initialfile="character.json",
                    )
                return filedialog.askopenfilename(
                    parent=root,
                    title="Import character",
                    filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
                )
            except tk.TclError as error:
                raise OSError("File dialog is unavailable.") from error
        finally:
            root.destroy()

    def _export(self):
        self._sync_values()
        try:
            path = self._file_dialog(save=True)
            if not path:
                return
            with open(path, "w", encoding="utf-8") as character_file:
                json.dump(self.values, character_file, indent=2, ensure_ascii=False)
            self.message = "Character exported."
        except (ImportError, OSError) as error:
            self.message = f"Could not export character: {error}"

    def _import(self):
        try:
            path = self._file_dialog(save=False)
            if not path:
                return
            values = load_character(path)
            for name in self.FIELDS:
                self.inputs[name].set_value(values[name])
            for name, sliders in self.color_sliders.items():
                for slider, value in zip(sliders, parse_color(values[name])):
                    slider.value = value
            self._sync_values()
            self.message = "Character imported."
        except (ImportError, OSError, json.JSONDecodeError, ValueError) as error:
            self.message = f"Could not import character: {error}"

    def handle_event(self, app, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._clear_focus()
            app.screen = "menu"
            return

        for name, button in self.tab_buttons.items():
            if button.clicked(event):
                self._clear_focus()
                self.category = name
                return

        if self.turn_left_button.clicked(event) or self.turn_right_button.clicked(event):
            step = -1 if self.turn_left_button.clicked(event) else 1
            self._clear_focus()
            self.direction = (self.direction + step) % len(self.DIRECTIONS)
            return

        if self.category == "Stats":
            for text_input in self.inputs.values():
                text_input.handle_event(event)
        else:
            for sliders in self.color_sliders.values():
                for slider in sliders:
                    slider.handle_event(event)

        self._sync_values()
        if self.back_button.clicked(event):
            self._clear_focus()
            app.screen = "menu"
        elif self.random_button.clicked(event):
            self._randomize()
        elif self.import_button.clicked(event):
            self._clear_focus()
            self._import()
        elif self.export_button.clicked(event):
            self._clear_focus()
            self._export()

    def update(self, dt):
        if self.category == "Stats":
            for text_input in self.inputs.values():
                text_input.update(dt)

    def _draw_preview(self, surface):
        frame = self.renderer.render(self.values, self.direction)
        scale = min(1.75, self.preview_area.width / frame.get_width(), self.preview_area.height / frame.get_height())
        self.preview = self.renderer.render(self.values, self.direction, scale)
        surface.blit(self.preview, self.preview.get_rect(midbottom=self.preview_area.midbottom))

    def draw(self, app):
        self._sync_values()
        surface = app.window
        surface.fill(GRASS)
        pygame.draw.rect(surface, PAPER, (50, 65, 860, 545), border_radius=BORDER_RADIUS)
        draw_centered_text(surface, "Character Creator", app.heading_font, 88)
        pygame.draw.rect(surface, GRASS, (80, 140, 300, 375), border_radius=BORDER_RADIUS)

        label = app.font.render("Preview", True, INK)
        surface.blit(label, label.get_rect(center=(230, 170)))
        self._draw_preview(surface)
        direction_label = app.small_font.render(self.DIRECTIONS[self.direction], True, MUTED)
        surface.blit(direction_label, direction_label.get_rect(center=(230, 194)))
        self.turn_left_button.draw(surface, app.small_font)
        self.turn_right_button.draw(surface, app.small_font)
        for name, button in self.tab_buttons.items():
            button.draw(surface, app.font)
            if self.category == name:
                pygame.draw.line(
                    surface, PAPER,
                    (button.rect.left + 16, button.rect.bottom - 5),
                    (button.rect.right - 16, button.rect.bottom - 5), 2,
                )

        if self.category == "Stats":
            for name, text_input in self.inputs.items():
                label = app.font.render(name, True, INK)
                surface.blit(label, label.get_rect(midleft=(430, text_input.rect.centery)))
                text_input.draw(surface, app.font)
        else:
            for name, sliders in self.color_sliders.items():
                panel = self.color_panels[name]
                pygame.draw.rect(surface, GRASS, panel, 1, border_radius=BORDER_RADIUS)
                color = self._parse_color(self.values[name])
                label = app.font.render(name, True, INK)
                surface.blit(label, (panel.x + 12, panel.y + 12))
                swatch = pygame.Rect(panel.right - 42, panel.y + 12, 30, 24)
                pygame.draw.rect(surface, color, swatch)
                pygame.draw.rect(surface, MUTED, swatch, 1)
                for slider in sliders:
                    slider.draw(surface, app.small_font)
                value = app.small_font.render(self.values[name], True, MUTED)
                surface.blit(value, (panel.x + 12, panel.y + 41))

        self.back_button.draw(surface, app.font)
        self.random_button.draw(surface, app.font)
        self.import_button.draw(surface, app.small_font)
        self.export_button.draw(surface, app.small_font)
        if self.message:
            message = app.small_font.render(self.message, True, MUTED)
            previous_clip = surface.get_clip()
            surface.set_clip(pygame.Rect(430, 510, 420, 28))
            surface.blit(message, (430, 515))
            surface.set_clip(previous_clip)
