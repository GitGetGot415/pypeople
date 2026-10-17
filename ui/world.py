from functools import lru_cache
from pathlib import Path

import pygame

from constants import INK, PAPER, SHADOW_COLOR, WORLD_COLORS, WORLD_HEIGHT, WORLD_WIDTH



CHARACTER_SCALE = 0.5
TREE_SCALE = 0.6
TREE_BASE_OFFSET = 64
APPLE_POINTS = ((38, 58), (83, 28), (112, 60), (65, 84), (96, 111), (33, 101), (111, 32), (46, 30))
_tree_artwork = None


@lru_cache(maxsize=64)
def npc_name_font(size):
    return pygame.font.Font(None, size)


@lru_cache(maxsize=4)
def interaction_bubble(name):
    path = Path(__file__).resolve().parent.parent / "assets" / f"{name}.png"
    return pygame.image.load(str(path)).convert_alpha()


def tree_artwork():
    global _tree_artwork
    if _tree_artwork is None:
        asset_dir = Path(__file__).resolve().parent.parent / "assets"
        _tree_artwork = tuple(pygame.image.load(str(asset_dir / name)).convert_alpha()
                              for name in ("Tree.png", "Apple.png"))
    return _tree_artwork


@lru_cache(maxsize=16)
def snake_artwork(facing, frame):
    direction = ("N", "NE", "E", "SE", "S", "SE", "E", "NE")[facing]
    path = Path(__file__).resolve().parent.parent / "assets" / "snake" / f"{direction}{frame}.png"
    artwork = pygame.image.load(str(path)).convert_alpha()
    return pygame.transform.flip(artwork, True, False) if facing > 4 else artwork


def _draw_shadow(surface, rect):
    shadow = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(shadow, SHADOW_COLOR, shadow.get_rect())
    surface.blit(shadow, rect)


def draw_world(surface, npcs, tree, font, character_renderer, camera_offset=(0, 0), phase="day", zoom=1.0, viewport=None, snake=None):
    # Use nearest-neighbor scaling throughout to keep the original pixels crisp.
    # Scale the ground separately; sprites are drawn from their original artwork.
    scene = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))
    _draw_scene(scene, npcs, tree, font, phase)
    surface.fill(WORLD_COLORS[phase][0])
    if zoom != 1.0:
        scene = pygame.transform.scale(scene, (round(WORLD_WIDTH * zoom), round(WORLD_HEIGHT * zoom)))
    previous_clip = surface.get_clip()
    scene_rect = scene.get_rect(topleft=(round(camera_offset[0]), round(camera_offset[1])))
    viewport = surface.get_rect() if viewport is None else viewport
    # Bubbles can extend beyond the ground's rectangle into the surrounding view.
    surface.set_clip(viewport.clip(previous_clip))
    surface.blit(scene, scene_rect)
    _draw_objects(surface, npcs, tree, font, character_renderer, camera_offset, phase, zoom, snake)
    surface.set_clip(previous_clip)


def _draw_scene(surface, npcs, tree, font, phase):
    grass, walkable = WORLD_COLORS[phase]
    # Darken only the band between the walking area and the map's outer edge.
    surface.fill(tuple(channel - 10 for channel in grass))
    if npcs:
        left, right, top, bottom = npcs[0].movement_bounds
        walkable_rect = pygame.Rect(
            round(left),
            round(top),
            round(right - left),
            round(bottom - top),
        )
        pygame.draw.rect(surface, walkable, walkable_rect)

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
    name_font = npc_name_font(max(1, round(22 * zoom)))
    map_rect = pygame.Rect(screen_point((0, 0)), (round(WORLD_WIDTH * zoom), round(WORLD_HEIGHT * zoom)))
    tree_image, apple_image = tree_artwork()
    # NPC y is at their feet; use the bottom of the tree's trunk for the same comparison.
    objects = [tree, *npcs]
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
            label = name_font.render(f"Fruit {tree.fruit}/{tree.capacity}", True, label_color)
            surface.blit(label, label.get_rect(midtop=screen_point((tree_x, tree.y + 76))))
            continue
        if obj is snake:
            image = resized(snake_artwork(snake.facing, snake.frame), zoom)
            surface.blit(image, image.get_rect(midbottom=screen_point((snake.x, snake.y))))
            label = name_font.render(f"{snake.name}: {snake.status}", True, label_color)
            surface.blit(label, label.get_rect(midtop=screen_point((snake.x, snake.y + 30))))
            if snake.activity == "eating":
                bubble = resized(interaction_bubble("Eating"), zoom)
                surface.blit(bubble, bubble.get_rect(bottomleft=screen_point((snake.x + 26, snake.y - 30))))
            continue
        npc = obj
        position = screen_point((npc.x, npc.y))
        # Always resize from the original composed artwork, including when zooming out.
        original = character_renderer.render(npc.character, npc.facing, giving=npc.giving, eating=npc.eating)
        sprite = resized(original, CHARACTER_SCALE * zoom)
        sprite_rect = sprite.get_rect(midbottom=position)
        shadow = pygame.Rect(screen_point((npc.x - 16, npc.y - 4)), (round(32 * zoom), round(8 * zoom)))
        _draw_shadow(surface, shadow)
        surface.blit(sprite, sprite_rect)
        name = name_font.render(npc.name, True, label_color)
        name_rect = name.get_rect(midtop=screen_point((npc.x, npc.y + 6)))
        name_rect.clamp_ip(map_rect)
        surface.blit(name, name_rect)
    # Draw interaction bubbles above the characters, anchored to their top right.
    for npc in npcs:
        bubble_name = "Eating" if npc.eating else "Giving" if npc.giving else "Thanking" if npc.thanking else None
        if npc.conversation is not None and npc.conversation.arrived and npc.conversation.initiator is npc:
            bubble_name = "Talking"
        if bubble_name is not None:
            width, height = npc.sprite_size or (
                round(character_renderer.character_size[0] * CHARACTER_SCALE),
                round(character_renderer.character_size[1] * CHARACTER_SCALE),
            )
            bubble = resized(interaction_bubble(bubble_name), zoom)
            rect = bubble.get_rect(bottomleft=screen_point((npc.x + width / 2 - 6, npc.y - height + 8)))
            surface.blit(bubble, rect)
