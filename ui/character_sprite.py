"""Shared directional sprites for the character creator and the world."""

from pathlib import Path

import pygame

from character_data import DEFAULT_COLORS, EYE_STYLES, HAIR_STYLES, parse_color


class CharacterRenderer:
    """Load assets once and cache each eye style, hairstyle, and facing."""

    _parse_color = staticmethod(parse_color)

    def __init__(self):
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
            style: {
                direction: pygame.image.load(str(asset_dir / "eyes" / style / f"Eyes_{direction}.png")).convert_alpha()
                for direction in ("S", "SE", "E")
            }
            for style in EYE_STYLES
        }
        self.eye_masks = {
            style: {
                direction: {"Eye Color": pygame.mask.from_threshold(asset, (5, 255, 0, 255), (1, 1, 1, 1))}
                for direction, asset in facings.items()
            }
            for style, facings in self.eyes.items()
        }
        self.eye_anchors = {}
        for style, facings in self.eyes.items():
            self.eye_anchors[style] = {}
            for direction, asset in facings.items():
                # Registration marks identify the head center even when the eye
                # artwork is cropped asymmetrically for a diagonal or side view.
                marks = [x for x in range(asset.get_width()) if asset.get_at((x, 0)).a]
                self.eye_anchors[style][direction] = (min(marks) + max(marks) + 1) // 2 if marks else asset.get_width() // 2
        self.hair = {
            style: {
                direction: pygame.image.load(str(asset_dir / "hair" / style / f"Hair_{direction}.png")).convert_alpha()
                for direction in ("S", "SE", "E", "NE", "N")
            }
            for style in HAIR_STYLES
        }
        self.hair_masks = {
            style: {
                direction: {"Hair Color": pygame.mask.from_threshold(asset, (255, 168, 0, 255), (1, 1, 1, 1))}
                for direction, asset in facings.items()
            }
            for style, facings in self.hair.items()
        }
        # Every facing uses the same canvas, so taller hair fits without moving
        # or resizing the standing body as the character turns.
        self.character_size = (
            max(96, *(asset.get_width() for facings in self.hair.values() for asset in facings.values())),
            max(self.body.get_height(), *(asset.get_height() for facings in self.hair.values() for asset in facings.values())),
        )
        self.colored_body = None
        self.colored_hands = {}
        self.colored_eyes = {}
        self.colored_hair = {}
        self.values = {}
        self.direction = 4
        self._color_key = None
        self._frames = {}
        self._scaled_frames = {}

    def render(self, values, direction, scale=1.0):
        """Return a cached sprite; callers should not modify the surface."""
        if type(direction) is not int or not 0 <= direction < 8:
            raise ValueError("Facing must be one of the eight character directions.")
        if scale <= 0:
            raise ValueError("Sprite scale must be positive.")
        style = values["Hair"]
        if style not in HAIR_STYLES:
            raise ValueError("Hair must be one, two, three, or four.")
        eye_style = values["Eyes"]
        if eye_style not in EYE_STYLES:
            raise ValueError("Eyes must be one or two.")
        colors = tuple(parse_color(values.get(name, default)) for name, default in DEFAULT_COLORS.items())
        if colors != self._color_key:
            self.values = {
                name: "#" + "".join(f"{channel:02X}" for channel in rgb)
                for name, rgb in zip(DEFAULT_COLORS, colors)
            }
            self.colored_body = self._recolor_asset(self.body, self.body_masks)
            self.colored_hands = {
                facing: self._recolor_asset(asset, self.hand_masks[facing])
                for facing, asset in self.hands.items()
            }
            self.colored_eyes = {
                style: {
                    facing: self._recolor_asset(asset, self.eye_masks[style][facing])
                    for facing, asset in facings.items()
                }
                for style, facings in self.eyes.items()
            }
            self.colored_hair = {
                style: {
                    facing: self._recolor_asset(asset, self.hair_masks[style][facing])
                    for facing, asset in facings.items()
                }
                for style, facings in self.hair.items()
            }
            self._color_key = colors
            self._frames.clear()
            self._scaled_frames.clear()
        frame_key = (style, eye_style, direction)
        if frame_key not in self._frames:
            self.direction = direction
            self.hair_style = style
            self.eye_style = eye_style
            self._frames[frame_key] = self._compose_preview()
        frame = self._frames[frame_key]
        if scale == 1.0:
            return frame
        key = (style, eye_style, direction, scale)
        if key not in self._scaled_frames:
            size = (max(1, round(frame.get_width() * scale)), max(1, round(frame.get_height() * scale)))
            self._scaled_frames[key] = pygame.transform.scale(frame, size)
        return self._scaled_frames[key]

    def render_portrait(self, values, size):
        """Fit the character's south-facing head inside a portrait box."""
        frame = self.render(values, 4)
        head_bottom = max(rect.bottom for rect in self.body_masks["Skin Color"].get_bounding_rects())
        head_bottom += frame.get_height() - self.body.get_height()
        head = frame.subsurface((0, 0, frame.get_width(), head_bottom))
        head = head.subsurface(head.get_bounding_rect())
        scale = min(size[0] / head.get_width(), size[1] / head.get_height())
        return pygame.transform.scale(
            head, (max(1, round(head.get_width() * scale)), max(1, round(head.get_height() * scale)))
        )

    def render_eye_preview(self, values):
        """Return just the colored, front-facing eyes for a style option."""
        self.render(values, 4)
        eyes = self.colored_eyes[values["Eyes"]]["S"]
        # Remove registration marks before cropping to the visible artwork.
        artwork = eyes.subsurface((0, 1, eyes.get_width(), eyes.get_height() - 2))
        return artwork.subsurface(artwork.get_bounding_rect()).copy()

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
        hair = self.colored_hair[self.hair_style][hair_direction]
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
        eyes = self.colored_eyes[self.eye_style][eye_direction]
        layer = pygame.Surface(frame.get_size(), pygame.SRCALPHA)
        layer.blit(
            eyes, (48 - self.eye_anchors[self.eye_style][eye_direction], 1),
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

