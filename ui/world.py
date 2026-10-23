import pygame
from functools import lru_cache
from pathlib import Path
from constants import (GARDEN_LOCATION, GARDEN_FLOOR_ART, INK, PAPER, SHADOW_COLOR, CHARACTER_SHADOW_HEIGHT,
                      CHARACTER_SHADOW_OFFSET_Y, WORLD_COLORS,
                      APARTMENT_DOOR_RECT, APARTMENT_RECT, DEFAULT_BUBBLE_SPRITE_SIZE,
                      UI_SMALL_FONT_SIZE, NAME_TAG_FONT_SIZE, NAME_TAG_HEAD_GAP, PLAYER_NAME_TAG,
                      ROOM_BED_RECT, SLEEP_FACING, SLEEP_ROTATION, PLAYER_ROOM_NUMBER)
from activity import Behavior
from npc import NPC
from .fonts import UnicodeFont
from .character_sprite import snake_artwork


CHARACTER_SCALE = 0.5
TREE_SCALE = 0.6
TREE_BASE_OFFSET = 64
APPLE_POINTS = ((38, 58), (83, 28), (112, 60), (65, 84), (96, 111), (33, 101), (111, 32), (46, 30))
_tree_artwork = None


@lru_cache(maxsize=64)
def npc_name_font(size):
    return UnicodeFont(size)


@lru_cache(maxsize=None)
def interaction_bubble(name):
    path = Path(__file__).resolve().parent.parent / "assets" / f"{name}.png"
    return pygame.image.load(str(path)).convert_alpha()


def player_bubble_name(npc):
    if npc.generation_pending:
        return "Thinking"
    if npc.player_call and not npc.selected:
        return "Calling Player" if npc.unread_player_reply else "Requesting Player"
    return None


def player_bubble_rect(npc, offset, zoom):
    return _bubble_rect(npc, player_bubble_name(npc), offset, zoom)


def _bubble_rect(npc, name, offset, zoom):
    image = interaction_bubble(name)
    width, height = npc.sprite_size or DEFAULT_BUBBLE_SPRITE_SIZE
    anchor = (round((npc.x + width / 2 - 6) * zoom + offset[0]),
              round((npc.y - height + 8) * zoom + offset[1]))
    return pygame.Rect(0, 0, max(1, round(image.get_width() * zoom)),
                       max(1, round(image.get_height() * zoom))).move(anchor[0], anchor[1] - max(1, round(image.get_height() * zoom)))


def _active_bubble(npc):
    if npc.conversation is not None and npc.conversation.arrived and npc.conversation.initiator is npc:
        return "Talking"
    return "Eating" if npc.eating else "Giving" if npc.giving else "Thanking" if npc.thanking else None


def player_bubble_visible(npc, offset=None, zoom=1.0):
    name = player_bubble_name(npc)
    if not (name and not npc.departure and not npc.busy):
        return False
    if name in ("Thinking", "Calling Player"):
        return True
    if _active_bubble(npc):
        return False
    if offset is not None:
        rect = player_bubble_rect(npc, offset, zoom)
        for other in (n for n in npc.world.npcs if n.location == npc.location):
            if other is npc:
                continue
            active = _active_bubble(other)
            if active and rect.colliderect(_bubble_rect(other, active, offset, zoom)):
                return False
            if ((other.unread_player_reply or other.generation_pending or other.id < npc.id)
                    and player_bubble_name(other) and not other.busy and not active
                    and rect.colliderect(player_bubble_rect(other, offset, zoom))):
                return False
    return True


def tree_artwork():
    global _tree_artwork
    if _tree_artwork is None:
        asset_dir = Path(__file__).resolve().parent.parent / "assets"
        _tree_artwork = tuple(pygame.image.load(str(asset_dir / name)).convert_alpha()
                              for name in ("Tree.png", "Apple.png"))
    return _tree_artwork


def _draw_shadow(surface, rect):
    # A white multiply mask leaves the ground untouched outside the oval.
    # Leave destination alpha unchanged beneath the tree's transparent pixels,
    # avoiding a rectangular alpha region around the shadow on macOS.
    shadow = pygame.Surface(rect.size, depth=32)
    shadow.fill((255, 255, 255))
    shade = (255 - SHADOW_COLOR[3],) * 3
    pygame.draw.ellipse(shadow, shade, shadow.get_rect())
    surface.blit(shadow, rect, special_flags=pygame.BLEND_RGB_MULT)


def draw_world(surface, npcs, tree, font, character_renderer, camera_offset=(0, 0), phase="day", zoom=1.0, viewport=None, snake=None, *, world=None, location="garden"):
    # Use nearest-neighbor scaling throughout to keep the original pixels crisp.
    # Scale the ground separately; artwork is drawn directly at the camera zoom.
    from locations import map_size, targets, player_has_room
    from constants import INDOOR_BACKGROUND, APARTMENT_ART_SCALE, ROOM_EXIT_RECT, HALLWAY_EXIT_RECT
    size = map_size(location)
    scene = pygame.Surface(size)
    artwork = []
    labels = []
    if location == "garden":
        _draw_scene(scene, tree, phase)
        surface.fill(WORLD_COLORS[phase][0])
        artwork.extend((("exterior", APARTMENT_RECT), ("door", APARTMENT_DOOR_RECT)))
    else:
        surface.fill(INDOOR_BACKGROUND)
        scene.fill(INDOOR_BACKGROUND)
        kind = location.split(":")[0]
        art_size = tuple(round(extent * APARTMENT_ART_SCALE) for extent in apartment_art(kind).get_size())
        artwork.append((kind, (0, 0, *art_size)))
        if kind == "room":
            artwork.extend((("bed", ROOM_BED_RECT), ("exit", ROOM_EXIT_RECT)))
        elif location == "hallway:1":
            artwork.append(("exit", HALLWAY_EXIT_RECT))
        if world:
            for target in targets(world, location):
                if target["kind"] in ("room", "elevator"):
                    artwork.append(("elevator" if target["kind"] == "elevator" else "door", target["rect"]))
                    target_rect = pygame.Rect(target["rect"])
                    if target["kind"] == "room":
                        number = target["room"]
                        owner = next((n for n in world.npcs if n.room_number == number), None)
                        label = ("Your room" if number == PLAYER_ROOM_NUMBER and player_has_room(world)
                                 else owner.name if owner else "Vacant")
                        labels.append((label, (target_rect.centerx, target_rect.top - 2)))
    if zoom != 1.0:
        scene = pygame.transform.scale(scene, (round(size[0] * zoom), round(size[1] * zoom)))
    previous_clip = surface.get_clip()
    scene_rect = scene.get_rect(topleft=(round(camera_offset[0]), round(camera_offset[1])))
    viewport = surface.get_rect() if viewport is None else viewport
    # Bubbles can extend beyond the ground's rectangle into the surrounding view.
    surface.set_clip(viewport.clip(previous_clip))
    surface.blit(scene, scene_rect)
    for kind, geometry in artwork:
        x, y, width, height = geometry
        rect = pygame.Rect(round(x * zoom + camera_offset[0]), round(y * zoom + camera_offset[1]),
                           max(1, round(width * zoom)), max(1, round(height * zoom)))
        # Always start from the full-resolution source, including when zooming in.
        surface.blit(apartment_art_at_size(kind, rect.size), rect)
    for label, (x, y) in labels:
        label_image = font.render(label, True, INK)
        if zoom != 1.0:
            label_image = pygame.transform.scale(label_image, (
                max(1, round(label_image.get_width() * zoom)), max(1, round(label_image.get_height() * zoom))))
        surface.blit(label_image, label_image.get_rect(midbottom=(
            round(x * zoom + camera_offset[0]), round(y * zoom + camera_offset[1]))))
    npcs = [npc for npc in npcs if npc.location == location]
    if world and world.mode == "player" and world.player.alive and world.player.location == location:
        npcs.append(world.player)
    _draw_objects(surface, npcs, tree if location == "garden" else None, font, character_renderer,
                  camera_offset, phase, zoom, snake if location == "garden" else None)
    surface.set_clip(previous_clip)


@lru_cache(maxsize=1)
def garden_floor_art():
    return pygame.image.load(str(Path(__file__).resolve().parent.parent / GARDEN_FLOOR_ART)).convert()


@lru_cache(maxsize=2)
def colored_garden_floor(phase):
    floor = garden_floor_art().copy()
    mask = pygame.mask.from_threshold(floor, (255, 0, 0), (1, 1, 1, 255))
    overlay = mask.to_surface(setcolor=(*WORLD_COLORS[phase][1], 255), unsetcolor=(0, 0, 0, 0))
    floor.blit(overlay, (0, 0))
    return floor


def _draw_scene(surface, tree, phase):
    floor = colored_garden_floor(phase)
    if floor.get_size() != surface.get_size():
        floor = pygame.transform.scale(floor, surface.get_size())
    surface.blit(floor, (0, 0))

    tree_x = round(tree.x) + tree.offset
    # Shadows lie on the ground beneath every sprite.
    _draw_shadow(surface, pygame.Rect(tree_x - 43, round(tree.y) + 50, 86, 20))


def _draw_objects(surface, npcs, tree, font, character_renderer, camera_offset, phase, zoom, snake=None):
    def screen_point(point):
        return (round(point[0] * zoom + camera_offset[0]),
                round(point[1] * zoom + camera_offset[1]))

    def resized(image, scale):
        size = (max(1, round(image.get_width() * scale)), max(1, round(image.get_height() * scale)))
        return image if image.get_size() == size else pygame.transform.scale(image, size)

    label_color = PAPER if phase == "night" else INK
    label_font = npc_name_font(max(1, round(UI_SMALL_FONT_SIZE * zoom)))
    name_font = npc_name_font(max(1, round(NAME_TAG_FONT_SIZE * zoom)))
    tags = []
    tree_image, apple_image = tree_artwork()
    # NPC y is at their feet; use the bottom of the tree's trunk for the same comparison.
    objects = ([tree] if tree is not None else []) + list(npcs)
    if snake is not None:
        objects.append(snake)
    objects.sort(key=lambda obj: tree.y + TREE_BASE_OFFSET if obj is tree else obj.y)
    for obj in objects:
        if obj is tree:
            tree_x = tree.x + tree.offset
            base_y = tree.y + TREE_BASE_OFFSET
            image = resized(tree_image, TREE_SCALE * zoom)
            rect = image.get_rect(midbottom=screen_point((tree_x, base_y)))
            surface.blit(image, rect)
            left = tree_x - tree_image.get_width() * TREE_SCALE / 2
            top = base_y - tree_image.get_height() * TREE_SCALE
            apple = resized(apple_image, TREE_SCALE * zoom)
            for x, y in APPLE_POINTS[:tree.fruit]:
                center = screen_point((left + x * TREE_SCALE, top + y * TREE_SCALE))
                surface.blit(apple, apple.get_rect(center=center))
            label = label_font.render(f"Fruit {tree.fruit}", True, label_color)
            surface.blit(label, label.get_rect(midtop=screen_point((tree_x, tree.y + 76))))
            continue
        if obj is snake:
            image = resized(snake_artwork(snake.facing, snake.frame), zoom)
            rect = image.get_rect(midbottom=screen_point((snake.x, snake.y)))
            surface.blit(image, rect)
            tags.append((snake.name, image, rect))
            label = label_font.render(snake.status, True, label_color)
            surface.blit(label, label.get_rect(midtop=screen_point((snake.x, snake.y + 30))))
            if snake.activity == "eating":
                bubble = resized(interaction_bubble("Eating"), zoom)
                surface.blit(bubble, bubble.get_rect(bottomleft=screen_point((snake.x + 26, snake.y - 30))))
            continue
        npc = obj
        position = screen_point((npc.x, npc.y))
        sleeping = getattr(npc, "behavior", None) == Behavior.SLEEP
        # Always resize from the original composed artwork, including when zooming out.
        original = character_renderer.render(npc.appearance if hasattr(npc, "appearance") else npc.character, SLEEP_FACING if sleeping else npc.facing, giving=getattr(npc, "giving", False), eating=getattr(npc, "eating", False))
        if sleeping:
            original = pygame.transform.rotate(original, SLEEP_ROTATION)
        sprite = resized(original, CHARACTER_SCALE * zoom)
        sprite_rect = sprite.get_rect(midbottom=position)
        shadow_height = CHARACTER_SHADOW_HEIGHT
        shadow = pygame.Rect(screen_point((npc.x - 16, npc.y + CHARACTER_SHADOW_OFFSET_Y - shadow_height / 2)),
                             (round(32 * zoom), round(shadow_height * zoom)))
        if not sleeping:
            _draw_shadow(surface, shadow)
        surface.blit(sprite, sprite_rect)
        tag = npc.name if isinstance(npc, NPC) else PLAYER_NAME_TAG
        tags.append((tag, sprite, sprite_rect))
    # Tags remain readable above the scene, anchored to the visible artwork's head.
    for name, sprite, rect in tags:
        lines = [name_font.render(word, True, label_color) for word in name.split()]
        bottom = rect.top + sprite.get_bounding_rect().top - round(NAME_TAG_HEAD_GAP * zoom)
        top = bottom - sum(line.get_height() for line in lines)
        for line in lines:
            surface.blit(line, line.get_rect(midtop=(rect.centerx, top)))
            top += line.get_height()
    # Thinking draws above ordinary speech; ready replies have the final priority.
    for npc in sorted((n for n in npcs if isinstance(n, NPC)), key=lambda person: (bool(person.unread_player_reply), person.generation_pending)):
        bubble_name = (player_bubble_name(npc) if player_bubble_visible(npc, camera_offset, zoom)
                       else _active_bubble(npc))
        if bubble_name is not None:
            width, height = npc.sprite_size or (
                round(character_renderer.character_size[0] * CHARACTER_SCALE),
                round(character_renderer.character_size[1] * CHARACTER_SCALE),
            )
            bubble = resized(interaction_bubble(bubble_name), zoom)
            rect = bubble.get_rect(bottomleft=screen_point((npc.x + width / 2 - 6, npc.y - height + 8)))
            surface.blit(bubble, rect)


@lru_cache(maxsize=None)
def apartment_art(kind):
    from constants import APARTMENT_ART
    return pygame.image.load(str(Path(__file__).resolve().parent.parent / APARTMENT_ART[kind])).convert_alpha()


@lru_cache(maxsize=16)
def apartment_art_at_size(kind, size):
    """Fit map artwork to its logical pixel size with nearest-neighbor scaling."""
    image = apartment_art(kind)
    return image if image.get_size() == size else pygame.transform.scale(image, size)
