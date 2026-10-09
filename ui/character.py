import json
from pathlib import Path

import pygame

from character_data import DEFAULT_COLORS, DIRECTIONS, EYE_STYLES, FIELDS, HAIR_STYLES, load_character, parse_color, random_character

from .button import Button
from .file_dialog import character_file_dialog
from .character_sprite import CharacterRenderer
from .slider import ColorSlider
from .text import draw_centered_text
from .text_input import TextInput
from .theme import ACCENT, BORDER_RADIUS, GRASS, INK, MUTED, PAPER


class CharacterScreen:
    FIELDS = FIELDS
    DEFAULT_COLORS = DEFAULT_COLORS
    COLOR_GRID = ("Eye Color", "Hair Color", "Skin Color", "Shirt Color")
    DIRECTIONS = DIRECTIONS

    def __init__(self):
        self.values = {name: "" for name in self.FIELDS}
        self.values["Hair"] = "one"
        self.values["Eyes"] = "one"
        self.values.update(self.DEFAULT_COLORS)
        self.values["Opinions"] = {}
        self.category = "Stats"
        self.inputs = {}
        for index, name in enumerate(self.FIELDS):
            multiline = name == "Details"
            self.inputs[name] = TextInput(
                (550, 210 + index * 58, 300, 156 if multiline else 42),
                max_length=None if multiline else 100,
                strip_paste=not multiline,
                placeholder=f"Enter {name.lower()}",
                multiline=multiline,
            )
            self.inputs[name].set_value(self.values[name])
        self.stats_viewport = pygame.Rect(430, 200, 450, 308)
        self.stats_scroll = 0
        self.stats_input_tops = {name: field.rect.y for name, field in self.inputs.items()}
        self.stats_scroll_limit = max(field.rect.bottom for field in self.inputs.values()) + 10 - self.stats_viewport.bottom
        self.stats_scrollbar = pygame.Rect(864, self.stats_viewport.y, 8, self.stats_viewport.height)
        self.stats_dragging = False
        self._stats_drag_offset = 0
        self.tab_buttons = {
            name: Button((430 + index * 110, 140, 90, 44), name)
            for index, name in enumerate(("Stats", "Color", "Eyes", "Hair"))
        }
        self.eye_buttons = {
            style: Button((430 + index % 2 * 220, 200 + index // 2 * 158, 200, 148), style.title())
            for index, style in enumerate(EYE_STYLES)
        }
        self.hair_buttons = {
            style: Button((430 + index % 2 * 220, 200 + index // 2 * 158, 200, 148), style.title())
            for index, style in enumerate(HAIR_STYLES)
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
        self.stats_dragging = False
        for text_input in self.inputs.values():
            text_input.active = False
            text_input.backspace_held = False
        for sliders in self.color_sliders.values():
            for slider in sliders:
                slider.active = False
                slider.dragging = False

    def _set_stats_scroll(self, value):
        self.stats_scroll = max(0, min(self.stats_scroll_limit, round(value)))
        for name, field in self.inputs.items():
            field.rect.y = self.stats_input_tops[name] - self.stats_scroll
            if not field.rect.colliderect(self.stats_viewport):
                field.active = False
                field.backspace_held = False
            elif field.active:
                pygame.key.set_text_input_rect(field.rect.clip(self.stats_viewport))

    def _stats_scroll_thumb(self):
        track = self.stats_scrollbar
        height = max(36, round(track.height * self.stats_viewport.height / (self.stats_viewport.height + self.stats_scroll_limit)))
        top = track.y + round((track.height - height) * self.stats_scroll / self.stats_scroll_limit)
        return pygame.Rect(track.x, top, track.width, height)

    def _reveal_stats_input(self, field):
        if field.rect.top < self.stats_viewport.top:
            self._set_stats_scroll(self.stats_scroll + field.rect.top - self.stats_viewport.top - 10)
        elif field.rect.bottom > self.stats_viewport.bottom:
            self._set_stats_scroll(self.stats_scroll + field.rect.bottom - self.stats_viewport.bottom + 10)

    def _handle_stats_event(self, event):
        if event.type == pygame.MOUSEWHEEL:
            pos = getattr(event, "pos", pygame.mouse.get_pos())
            if self.stats_viewport.collidepoint(pos):
                details = self.inputs["Details"]
                if not details.rect.collidepoint(pos) or not details.scroll(event.y):
                    self._set_stats_scroll(self.stats_scroll - event.y * 42)
                return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.stats_dragging = False
        if event.type == pygame.MOUSEMOTION and self.stats_dragging:
            thumb = self._stats_scroll_thumb()
            offset = event.pos[1] - self._stats_drag_offset - self.stats_scrollbar.y
            self._set_stats_scroll(offset * self.stats_scroll_limit / (self.stats_scrollbar.height - thumb.height))
            return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.stats_scrollbar.collidepoint(event.pos):
                self._clear_focus()
                thumb = self._stats_scroll_thumb()
                if not thumb.collidepoint(event.pos):
                    offset = event.pos[1] - self.stats_scrollbar.y - thumb.height / 2
                    self._set_stats_scroll(offset * self.stats_scroll_limit / (self.stats_scrollbar.height - thumb.height))
                    thumb = self._stats_scroll_thumb()
                self.stats_dragging = True
                self._stats_drag_offset = event.pos[1] - thumb.y
                return True
            if not self.stats_viewport.collidepoint(event.pos):
                self._clear_focus()
                return False
        for field in self.inputs.values():
            field.handle_event(event)
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN, pygame.TEXTINPUT):
            for field in self.inputs.values():
                if field.active:
                    self._reveal_stats_input(field)
                    break
        return False

    def _randomize(self):
        self._clear_focus()
        try:
            values = random_character(self.random_config_path)
            for name in self.FIELDS:
                self.inputs[name].set_value(values[name])
            for name, sliders in self.color_sliders.items():
                for slider, value in zip(sliders, parse_color(values[name])):
                    slider.value = value
            self.values["Hair"] = values["Hair"]
            self.values["Eyes"] = values["Eyes"]
            self.values["Opinions"] = values["Opinions"]
            self._sync_values()
            self.message = "Character randomized."
        except (OSError, ValueError) as error:
            self.message = f"Could not randomize character: {error}"

    _file_dialog = staticmethod(character_file_dialog)

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
            self.values["Hair"] = values["Hair"]
            self.values["Eyes"] = values["Eyes"]
            # Preserve opinions on import/export without exposing an editor.
            self.values["Opinions"] = values["Opinions"]
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
            if self._handle_stats_event(event):
                return
        elif self.category in ("Eyes", "Hair"):
            buttons = self.eye_buttons if self.category == "Eyes" else self.hair_buttons
            for style, button in buttons.items():
                if button.clicked(event):
                    self.values[self.category] = style
        elif self.category == "Color":
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
            previous_clip = surface.get_clip()
            surface.set_clip(self.stats_viewport.clip(previous_clip))
            for name, text_input in self.inputs.items():
                if not text_input.rect.colliderect(self.stats_viewport):
                    continue
                label = app.font.render(name, True, INK)
                label_y = text_input.rect.y + 21 if text_input.multiline else text_input.rect.centery
                surface.blit(label, label.get_rect(midleft=(430, label_y)))
                text_input.draw(surface, app.font)
            surface.set_clip(previous_clip)
            pygame.draw.rect(surface, GRASS, self.stats_scrollbar)
            pygame.draw.rect(surface, ACCENT, self._stats_scroll_thumb())
        elif self.category in ("Eyes", "Hair"):
            buttons = self.eye_buttons if self.category == "Eyes" else self.hair_buttons
            for style, button in buttons.items():
                panel = button.rect
                selected = self.values[self.category] == style
                if selected or panel.collidepoint(pygame.mouse.get_pos()):
                    pygame.draw.rect(surface, GRASS, panel, border_radius=BORDER_RADIUS)
                pygame.draw.rect(
                    surface, ACCENT if selected else MUTED, panel,
                    3 if selected else 1, border_radius=BORDER_RADIUS,
                )
                label = app.font.render(button.label, True, INK)
                surface.blit(label, label.get_rect(midtop=(panel.centerx, panel.y + 12)))
                values = {**self.values, self.category: style}
                if self.category == "Eyes":
                    eyes = self.renderer.render_eye_preview(values)
                    preview_area = pygame.Rect(panel.x + 12, panel.y + 40, panel.width - 24, panel.height - 52)
                    scale = min(preview_area.width / eyes.get_width(), preview_area.height / eyes.get_height())
                    size = (max(1, round(eyes.get_width() * scale)), max(1, round(eyes.get_height() * scale)))
                    preview = pygame.transform.scale(eyes, size)
                    surface.blit(preview, preview.get_rect(center=preview_area.center))
                else:
                    frame = self.renderer.render(values, self.direction)
                    scale = min(1.0, (panel.width - 24) / frame.get_width(), (panel.height - 48) / frame.get_height())
                    preview = self.renderer.render(values, self.direction, scale)
                    surface.blit(preview, preview.get_rect(midbottom=(panel.centerx, panel.bottom - 8)))
        elif self.category == "Color":
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
