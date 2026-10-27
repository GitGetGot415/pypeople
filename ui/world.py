import re
import pygame
from functools import lru_cache
from pathlib import Path
from constants import (GARDEN_LOCATION, GARDEN_FLOOR_ART, GARDEN_FLOOR_OFFSET, INK, PAPER, SHADOW_COLOR, CHARACTER_SHADOW_HEIGHT,
                      CHARACTER_SHADOW_OFFSET_Y, WORLD_COLORS,
                      APARTMENT_DOOR_RECT, APARTMENT_RECT, APARTMENT_FLOORS, APARTMENT_FLOOR_SIZE,
                      APARTMENT_BASE_Y, APARTMENT_ROOF_RECT, GARDEN_BRIDGE_RECT,
                      GARDEN_UNDERSIDE_RECT, APARTMENT_UNDERSIDE_RECT,
                      UI_SMALL_FONT_SIZE, NAME_TAG_FONT_SIZE, NAME_TAG_HEAD_GAP, PLAYER_NAME_TAG,
                      ROOM_BED_RECT, PLAYER_ROOM_NUMBER, TREE_BASE_OFFSET)
from locations import active_player
from npc import NPC
from .fonts import UnicodeFont
from .character_sprite import snake_artwork


CHARACTER_SCALE = 0.5
TREE_SCALE = 0.6
APPLE_POINTS = ((38, 58), (83, 28), (112, 60), (65, 84), (96, 111), (33, 101), (111, 32), (46, 30))
_tree_artwork = None


@lru_cache(maxsize=64)
def npc_name_font(size):
    return UnicodeFont(size)


@lru_cache(maxsize=None)
def interaction_bubble(name):
    path = Path(__file__).resolve().parent.parent / "assets" / f"{name}.png"
    return pygame.image.load(str(path)).convert_alpha()


_TURN_STAGE = re.compile(r"^(?:Generating|Correcting|Recovery: rewriting) turn \d+ for (.+)$")


def generation_bubbles(conversations, workers):
    """Bubbles for both participants of each conversation being generated, keyed by actor ID.

    The person whose turn is being written talks; any other stage (waiting for the model, assessments,
    validation, voice rewrites) shows the initiator thinking. The other participant shows nothing.
    """
    stages = {worker.request_id: worker.stage for worker in workers}
    bubbles = {}
    for request_id, conversation in conversations.items():
        match = _TURN_STAGE.match(stages.get(request_id, ""))
        speaker = next((person for person in (conversation.initiator, conversation.recipient)
                        if match and person.name == match[1]), None)
        bubbles[conversation.initiator.id] = None
        bubbles[conversation.recipient.id] = None
        bubbles[(speaker or conversation.initiator).id] = "Talking" if speaker else "Thinking"
    return bubbles


def _active_bubble(npc):
    if npc.conversation is not None and npc.conversation.arrived and npc.conversation.initiator is npc:
        return "Talking"
    return "Eating" if npc.eating else "Giving" if npc.giving else "Thanking" if npc.thanking else None


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


def draw_world(surface, npcs, tree, font, character_renderer, camera_offset=(0, 0), zoom=1.0, viewport=None, snake=None, *, world=None, location="garden", speaking=None):
    # Use nearest-neighbor scaling throughout to keep the original pixels crisp.
    # Scale the ground separately; artwork is drawn directly at the camera zoom.
    from locations import map_size, targets
    from constants import INDOOR_BACKGROUND, HALLWAY_SIZE, ROOM_SIZE, ROOM_EXIT_RECT, HALLWAY_EXIT_RECT
    size = map_size(location)
    # macOS window formats include alpha even without SRCALPHA. Blitting the
    # floor into an inherited format can clear that alpha, making later sprites
    # and text replace the ground with their transparent background colors.
    scene = pygame.Surface(size, depth=32)
    artwork = []
    labels = []
    if location == "garden":
        _draw_scene(scene, tree)
        surface.fill(WORLD_COLORS[0])
        artwork.extend((("garden_underside", GARDEN_UNDERSIDE_RECT),
                        ("apartment_underside", APARTMENT_UNDERSIDE_RECT)))
        artwork.append(("bridge", GARDEN_BRIDGE_RECT))
        width, height = APARTMENT_FLOOR_SIZE
        for floor in range(APARTMENT_FLOORS):
            kind = "exterior_ground" if floor == 0 else "exterior_upper"
            artwork.append((kind, (APARTMENT_RECT[0], APARTMENT_BASE_Y - (floor + 1) * height, width, height)))
        artwork.extend((("roof", APARTMENT_ROOF_RECT), ("door", APARTMENT_DOOR_RECT)))
    else:
        surface.fill(INDOOR_BACKGROUND)
        scene.fill(INDOOR_BACKGROUND)
        kind = location.split(":")[0]
        # Map dimensions stay fixed even when the artwork's resolution changes.
        art_size = ROOM_SIZE if kind == "room" else HALLWAY_SIZE
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
                        label = ("Your room" if number == PLAYER_ROOM_NUMBER and active_player(world)
                                 else owner.name if owner else "Vacant")
                        labels.append((label, (target_rect.centerx, target_rect.top - 2)))
    if zoom != 1.0:
        scene = pygame.transform.scale(scene, (round(size[0] * zoom), round(size[1] * zoom)))
    if location == GARDEN_LOCATION:
        # Nearest-neighbor shrinking can skip the artwork's last column or row.
        # Restore its outline after both map resizing and camera zooming.
        pygame.draw.rect(scene, garden_floor_art().get_at((0, 0)), scene.get_rect(),
                         width=max(1, round(zoom)))
    previous_clip = surface.get_clip()
    floor_offset = GARDEN_FLOOR_OFFSET if location == GARDEN_LOCATION else (0, 0)
    scene_rect = scene.get_rect(topleft=(round(camera_offset[0] + floor_offset[0] * zoom),
                                        round(camera_offset[1] + floor_offset[1] * zoom)))
    viewport = surface.get_rect() if viewport is None else viewport
    # Bubbles can extend beyond the ground's rectangle into the surrounding view.
    surface.set_clip(viewport.clip(previous_clip))
    surface.blit(scene, scene_rect)
    for kind, geometry in artwork:
        x, y, width, height = geometry
        left, top = round(x * zoom + camera_offset[0]), round(y * zoom + camera_offset[1])
        # Round shared edges identically so stacked floors never leave a gap.
        right = round((x + width) * zoom + camera_offset[0])
        bottom = round((y + height) * zoom + camera_offset[1])
        rect = pygame.Rect(left, top, max(1, right - left), max(1, bottom - top))
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
    if world and active_player(world) and world.player.location == location:
        npcs.append(world.player)
    _draw_objects(surface, npcs, tree if location == "garden" else None, font, character_renderer,
                  camera_offset, zoom, snake if location == "garden" else None,
                  bodies=[body for body in world.bodies if body["location"] == location] if world else (),
                  speaking=speaking or {})
    surface.set_clip(previous_clip)


@lru_cache(maxsize=1)
def garden_floor_art():
    return pygame.image.load(str(Path(__file__).resolve().parent.parent / GARDEN_FLOOR_ART)).convert()


@lru_cache(maxsize=1)
def colored_garden_floor():
    floor = garden_floor_art().copy()
    mask = pygame.mask.from_threshold(floor, (255, 0, 0), (1, 1, 1, 255))
    overlay = mask.to_surface(setcolor=(*WORLD_COLORS[1], 255), unsetcolor=(0, 0, 0, 0))
    floor.blit(overlay, (0, 0))
    return floor


def _draw_scene(surface, tree):
    floor = colored_garden_floor()
    if floor.get_size() != surface.get_size():
        floor = pygame.transform.scale(floor, surface.get_size())
    surface.blit(floor, (0, 0))

    tree_x = round(tree.x) - GARDEN_FLOOR_OFFSET[0]
    # Shadows lie on the ground beneath every sprite.
    _draw_shadow(surface, pygame.Rect(tree_x - 43, round(tree.y) + 50 - GARDEN_FLOOR_OFFSET[1], 86, 20))


def _draw_objects(surface, npcs, tree, font, character_renderer, camera_offset, zoom, snake=None, *, bodies=(), speaking=None):
    speaking = speaking or {}
    def screen_point(point):
        return (round(point[0] * zoom + camera_offset[0]),
                round(point[1] * zoom + camera_offset[1]))

    def resized(image, scale):
        size = (max(1, round(image.get_width() * scale)), max(1, round(image.get_height() * scale)))
        return image if image.get_size() == size else pygame.transform.scale(image, size)

    label_color = INK
    label_font = npc_name_font(max(1, round(UI_SMALL_FONT_SIZE * zoom)))
    name_font = npc_name_font(max(1, round(NAME_TAG_FONT_SIZE * zoom)))
    tags = []
    tree_image, apple_image = tree_artwork()
    # NPC y is at their feet; use the bottom of the tree's trunk for the same comparison.
    objects = ([tree] if tree is not None else []) + list(npcs)
    if snake is not None:
        objects.append(snake)
    objects.extend(bodies)
    objects.sort(key=lambda obj: obj["y"] if isinstance(obj, dict) else tree.y + TREE_BASE_OFFSET if obj is tree else obj.y)
    for obj in objects:
        if isinstance(obj, dict):
            from constants import BODY_SECONDS, BODY_FALL_SECONDS, BODY_FADE_SECONDS, BODY_ROTATION
            original = character_renderer.render(obj["appearance"], obj["facing"])
            progress = min(1.0, max(0.0, obj["elapsed"] / BODY_FALL_SECONDS))
            angle = BODY_ROTATION * progress ** 2
            scale = CHARACTER_SCALE * zoom
            image = resized(pygame.transform.rotate(original, angle), scale).copy()
            # Rotate the canvas center around its original bottom-center pivot.
            # Pygame's image rotation is counterclockwise in screen coordinates.
            pivot = pygame.Vector2(screen_point((obj["x"], obj["y"])))
            center = pivot - pygame.Vector2(0, original.get_height() / 2).rotate(-angle) * scale
            rect = image.get_rect(topleft=(round(center.x - image.get_width() / 2),
                                            round(center.y - image.get_height() / 2)))
            remaining = BODY_SECONDS - obj["elapsed"]
            image.set_alpha(round(255 * min(1.0, remaining / BODY_FADE_SECONDS)))
            surface.blit(image, rect)
            continue
        if obj is tree:
            import math
            from constants import TREE_SHAKE_OFFSET, TREE_SHAKE_FREQUENCY
            tree_x = tree.x + (math.sin(tree.shake_remaining * TREE_SHAKE_FREQUENCY) * TREE_SHAKE_OFFSET if tree.shake_remaining else 0)
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
            snake_rect = rect
            tags.append((snake.name, image, rect))
            label = label_font.render(snake.status, True, label_color)
            surface.blit(label, label.get_rect(midtop=screen_point((snake.x, snake.y + 30))))
            continue
        npc = obj
        position = screen_point((npc.x, npc.y))
        # Always resize from the original composed artwork, including when zooming out.
        original = character_renderer.render(npc.appearance if hasattr(npc, "appearance") else npc.character, npc.facing, giving=getattr(npc, "giving", False), eating=getattr(npc, "eating", False))
        sprite = resized(original, CHARACTER_SCALE * zoom)
        sprite_rect = sprite.get_rect(midbottom=position)
        shadow_height = CHARACTER_SHADOW_HEIGHT
        shadow = pygame.Rect(screen_point((npc.x - 16, npc.y + CHARACTER_SHADOW_OFFSET_Y - shadow_height / 2)),
                             (round(32 * zoom), round(shadow_height * zoom)))
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
    # Thinking draws above ordinary speech.
    if snake is not None:
        # The python's bubble follows the same rules, anchored to its own artwork.
        bubble_name = speaking.get(snake.id, "Talking" if snake.conversation is not None and snake.conversation.arrived
                                   and snake.conversation.initiator is snake else None)
        if bubble_name is not None:
            bubble = resized(interaction_bubble(bubble_name), zoom)
            surface.blit(bubble, bubble.get_rect(bottomleft=(snake_rect.right - round(6 * zoom),
                                                             snake_rect.top + round(8 * zoom))))
    for npc in sorted((n for n in npcs if isinstance(n, NPC)), key=lambda person: person.generation_pending):
        interaction = npc.world.player_interaction if npc.world else None
        approaching = (interaction and interaction.npc_id == npc.id and interaction.incoming
                       and interaction.stage == "approach")
        bubble_name = ("Thinking" if npc.generation_pending and npc.alive and not npc.busy and not approaching
                       else speaking[npc.id] if npc.id in speaking else _active_bubble(npc))
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
