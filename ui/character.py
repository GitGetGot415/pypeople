import json
from pathlib import Path

import pygame

from .button import Button
from .slider import ColorSlider
from .text import draw_centered_text
from .text_input import TextInput
from .theme import BORDER_RADIUS, GRASS, INK, MUTED, PAPER


class CharacterScreen:
    FIELDS = ("Name", "Age", "Weight", "Height", "Sex")
    DEFAULT_COLORS = {
        "Shirt Color": "#000AFF", "Skin Color": "#FF0000",
        "Eye Color": "#05FF00", "Hair Color": "#FFA800",
    }
    COLOR_GRID = ("Eye Color", "Hair Color", "Skin Color", "Shirt Color")
    DIRECTIONS = ("North", "North east", "East", "South east", "South", "South west", "West", "North west")

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
        self.import_button = Button((570, 545, 130, 48), "Import JSON")
        self.export_button = Button((720, 545, 130, 48), "Export JSON")
        self.turn_left_button = Button((96, 473, 120, 30), "Turn left")
        self.turn_right_button = Button((244, 473, 120, 30), "Turn right")
        self.preview_area = pygame.Rect(96, 206, 268, 260)
        self.direction = 4  # Start facing south, with both hands visible.
        self.message = ""

        asset_dir = Path(__file__).resolve().parent.parent / "assets"
        self.body = pygame.image.load(str(asset_dir / "Body.png")).convert_alpha()
        # Both masks come from the original image, keeping the regions separate
        # even when the chosen skin and shirt colors match the other region.
        self.body_masks = {
            "Skin Color": pygame.mask.from_threshold(self.body, (255, 0, 0, 255), (1, 1, 1, 1)),
            "Shirt Color": pygame.mask.from_threshold(self.body, (0, 10, 255, 255), (1, 1, 1, 1)),
        }
        self.hands = {
            direction: pygame.image.load(str(asset_dir / "hands" / f"Hands_{direction}.png")).convert_alpha()
            for direction in ("NS", "WE", "NSWE")
        }
        self.hand_masks = {
            direction: {"Skin Color": pygame.mask.from_threshold(asset, (255, 0, 0, 255), (1, 1, 1, 1))}
            for direction, asset in self.hands.items()
        }
        self.eyes = {
            direction: pygame.image.load(str(asset_dir / "eyes" / f"Eyes_{direction}.png")).convert_alpha()
            for direction in ("S", "SE", "E")
        }
        self.eye_masks = {
            direction: {"Eye Color": pygame.mask.from_threshold(asset, (5, 255, 0, 255), (1, 1, 1, 1))}
            for direction, asset in self.eyes.items()
        }
        self.eye_anchors = {}
        for direction, asset in self.eyes.items():
            # Registration marks identify the head center even when the eye
            # artwork is cropped asymmetrically for a diagonal or side view.
            marks = [x for x in range(asset.get_width()) if asset.get_at((x, 0)).a]
            self.eye_anchors[direction] = (min(marks) + max(marks) + 1) // 2 if marks else asset.get_width() // 2
        self.hair = {
            direction: pygame.image.load(str(asset_dir / "hair" / f"Hair_{direction}.png")).convert_alpha()
            for direction in ("S", "SE", "E", "NE", "N")
        }
        self.hair_masks = {
            direction: {"Hair Color": pygame.mask.from_threshold(asset, (255, 168, 0, 255), (1, 1, 1, 1))}
            for direction, asset in self.hair.items()
        }
        # Every facing uses the same canvas, so taller hair fits without moving
        # or resizing the standing body as the character turns.
        self.character_size = (
            max(96, *(asset.get_width() for asset in self.hair.values())),
            max(self.body.get_height(), *(asset.get_height() for asset in self.hair.values())),
        )
        self.colored_body = None
        self.colored_hands = {}
        self.colored_eyes = {}
        self.colored_hair = {}
        self.preview = None
        self.preview_colors = None
        self.preview_direction = None

    @staticmethod
    def _parse_color(value):
        if not isinstance(value, str):
            raise ValueError("Colors must be hex text such as #FF0000.")
        hex_color = value.strip().removeprefix("#")
        if len(hex_color) != 6 or any(char not in "0123456789abcdefABCDEF" for char in hex_color):
            raise ValueError("Colors must be hex text such as #FF0000.")
        return tuple(int(hex_color[index:index + 2], 16) for index in (0, 2, 4))

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
            with open(path, "r", encoding="utf-8") as character_file:
                data = json.load(character_file)
            if not isinstance(data, dict):
                raise ValueError("The JSON must contain a character object.")
            values = {}
            for name in self.FIELDS:
                value = data.get(name, "")
                if not isinstance(value, (str, int, float)):
                    raise ValueError(f"{name} must be text.")
                values[name] = str(value)
            colors = {}
            for name, default in self.DEFAULT_COLORS.items():
                # Older exports used one Color field for the preview's head.
                fallback = data.get("Color", default) if name == "Skin Color" else default
                colors[name] = self._parse_color(data.get(name, fallback))
            # Apply only after the entire file has been validated.
            for name, value in values.items():
                self.inputs[name].set_value(value)
            for name, rgb in colors.items():
                for slider, value in zip(self.color_sliders[name], rgb):
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
        colors = tuple(self.values[name] for name in self.DEFAULT_COLORS)
        if colors != self.preview_colors:
            self.colored_body = self._recolor_asset(self.body, self.body_masks)
            self.colored_hands = {
                direction: self._recolor_asset(asset, self.hand_masks[direction])
                for direction, asset in self.hands.items()
            }
            self.colored_eyes = {
                direction: self._recolor_asset(asset, self.eye_masks[direction])
                for direction, asset in self.eyes.items()
            }
            self.colored_hair = {
                direction: self._recolor_asset(asset, self.hair_masks[direction])
                for direction, asset in self.hair.items()
            }
            self.preview_colors = colors
            self.preview_direction = None
        if self.direction != self.preview_direction:
            frame = self._compose_preview()
            scale = min(1.75, self.preview_area.width / frame.get_width(), self.preview_area.height / frame.get_height())
            self.preview = pygame.transform.scale(
                frame, (round(frame.get_width() * scale), round(frame.get_height() * scale)),
            )
            self.preview_direction = self.direction
        surface.blit(self.preview, self.preview.get_rect(midbottom=self.preview_area.midbottom))

    def _compose_preview(self):
        """Turn in the ground plane while keeping the standing body upright."""
        frame = pygame.Surface((96, 144), pygame.SRCALPHA)
        body_rect = self.colored_body.get_rect(midtop=(48, 0))

        def draw_hands(hands):
            # Ignore the tiny alignment marks on the first and last asset rows.
            hand_rect = hands.get_rect(midtop=(48, 0))
            frame.blit(hands, (hand_rect.x, 1), (0, 1, hands.get_width(), hands.get_height() - 2))

        if self.direction in (0, 4):
            # From the back, the body covers the inner edges of the hands.
            if self.direction == 0:
                draw_hands(self.colored_hands["NS"])
            frame.blit(self.colored_body, body_rect)
            if self.direction == 4:
                draw_hands(self.colored_hands["NS"])
        elif self.direction in (2, 6):
            # The two hands overlap in a side view; the WE asset shows one.
            frame.blit(self.colored_body, body_rect)
            draw_hands(self.colored_hands["WE"])
        else:
            # The diagonal asset faces south east. Split its two hands so the
            # far hand can sit behind the body and the near one in front.
            hands = self.colored_hands["NSWE"]
            full_hand = pygame.Surface(hands.get_size(), pygame.SRCALPHA)
            half_hand = pygame.Surface(hands.get_size(), pygame.SRCALPHA)
            split = hands.get_width() // 2
            full_hand.blit(hands, (0, 1), (0, 1, split, hands.get_height() - 2))
            half_hand.blit(hands, (split, 1), (split, 1, hands.get_width() - split, hands.get_height() - 2))
            west = self.direction in (5, 7)
            north = self.direction in (1, 7)
            if north:
                # The asset already cuts off the far hand. When it becomes the
                # near hand, use its full counterpart at the same center.
                half_hand.fill((0, 0, 0, 0))
                half_hand.blit(hands, (46, 92), (0, 100, 24, 24))

            def orient(hand):
                hand = pygame.transform.flip(hand, west, north)
                if north:
                    # Reflect around the hands' ground-plane center (y=108),
                    # rather than the image center, to keep them below the head.
                    shifted = pygame.Surface(hand.get_size(), pygame.SRCALPHA)
                    shifted.blit(hand, (0, 2 * 108 - hand.get_height()))
                    hand = shifted
                return hand

            full_hand, half_hand = orient(full_hand), orient(half_hand)
            behind, front = (full_hand, half_hand) if north else (half_hand, full_hand)
            draw_hands(behind)
            frame.blit(self.colored_body, body_rect)
            draw_hands(front)
        self._draw_eyes(frame)
        return self._add_hair(frame)

    def _add_hair(self, body_frame):
        frame = pygame.Surface(self.character_size, pygame.SRCALPHA)
        frame.blit(body_frame, body_frame.get_rect(midbottom=frame.get_rect().midbottom))
        hair_direction = ("N", "NE", "E", "SE", "S", "SE", "E", "NE")[self.direction]
        hair = self.colored_hair[hair_direction]
        layer = pygame.Surface(frame.get_size(), pygame.SRCALPHA)
        # Align the bottom centers, letting taller hair extend above the head.
        hair_rect = hair.get_rect(midbottom=frame.get_rect().midbottom)
        # The last row contains alignment marks rather than visible hair.
        layer.blit(hair, hair_rect, (0, 0, hair.get_width(), hair.get_height() - 1))
        if self.direction in (5, 6, 7):
            layer = pygame.transform.flip(layer, True, False)
        frame.blit(layer, (0, 0))
        return frame

    def _draw_eyes(self, frame):
        # All three north-facing views show the back of the head.
        eye_direction = {2: "E", 3: "SE", 4: "S", 5: "SE", 6: "E"}.get(self.direction)
        if eye_direction is None:
            return
        eyes = self.colored_eyes[eye_direction]
        layer = pygame.Surface(frame.get_size(), pygame.SRCALPHA)
        layer.blit(
            eyes, (48 - self.eye_anchors[eye_direction], 1),
            (0, 1, eyes.get_width(), eyes.get_height() - 2),
        )
        if self.direction in (5, 6):
            layer = pygame.transform.flip(layer, True, False)
        # Keep the eye artwork inside the head without covering its outline.
        head_clip = pygame.Surface(frame.get_size(), pygame.SRCALPHA)
        head_mask = self.body_masks["Skin Color"].to_surface(
            setcolor=(255, 255, 255, 255), unsetcolor=(0, 0, 0, 0),
        )
        head_clip.blit(head_mask, self.colored_body.get_rect(midtop=(48, 0)))
        layer.blit(head_clip, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        frame.blit(layer, (0, 0))

    def _recolor_asset(self, asset, masks):
        colored = asset.copy()
        for name, mask in masks.items():
            overlay = mask.to_surface(
                setcolor=(*self._parse_color(self.values[name]), 255),
                unsetcolor=(0, 0, 0, 0),
            )
            colored.blit(overlay, (0, 0))
        return colored

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
        self.import_button.draw(surface, app.small_font)
        self.export_button.draw(surface, app.small_font)
        if self.message:
            message = app.small_font.render(self.message, True, MUTED)
            previous_clip = surface.get_clip()
            surface.set_clip(pygame.Rect(430, 510, 420, 28))
            surface.blit(message, (430, 515))
            surface.set_clip(previous_clip)
