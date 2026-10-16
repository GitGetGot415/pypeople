from functools import lru_cache
from pathlib import Path

import pygame

from npc import WORLD_WIDTH, WORLD_HEIGHT

from .theme import GRASS, INK, MUTED, PAPER, WALKABLE_GRASS


CHARACTER_SCALE = 0.5
TREE_SCALE = 0.6
TREE_BASE_OFFSET = 64
APPLE_POINTS = ((38, 58), (83, 28), (112, 60), (65, 84), (96, 111), (33, 101), (111, 32), (46, 30))
_talking_bubble = None
_tree_artwork = None


@lru_cache(maxsize=64)
def npc_name_font(size):
    return pygame.font.Font(None, size)


def talking_bubble():
    global _talking_bubble
    if _talking_bubble is None:
        path = Path(__file__).resolve().parent.parent / "assets" / "Talking.png"
        _talking_bubble = pygame.image.load(str(path)).convert_alpha()
    return _talking_bubble


def tree_artwork():
    global _tree_artwork
    if _tree_artwork is None:
        asset_dir = Path(__file__).resolve().parent.parent / "assets"
        _tree_artwork = tuple(pygame.image.load(str(asset_dir / name)).convert_alpha()
                              for name in ("Tree.png", "Apple.png"))
    return _tree_artwork


def draw_world(surface, npcs, tree, font, character_renderer, camera_offset=(0, 0), phase="day", zoom=1.0, viewport=None):
    # Scale the ground separately; sprites are drawn from their original artwork.
    scene = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))
    _draw_scene(scene, npcs, tree, font, phase)
    surface.fill(scene.get_at((0, 0)))
    if zoom != 1.0:
        scene = pygame.transform.smoothscale(scene, (round(WORLD_WIDTH * zoom), round(WORLD_HEIGHT * zoom)))
    previous_clip = surface.get_clip()
    scene_rect = scene.get_rect(topleft=(round(camera_offset[0]), round(camera_offset[1])))
    viewport = surface.get_rect() if viewport is None else viewport
    surface.set_clip(viewport.clip(previous_clip).clip(scene_rect))
    surface.blit(scene, scene_rect)
    _draw_objects(surface, npcs, tree, font, character_renderer, camera_offset, phase, zoom)
    surface.set_clip(previous_clip)


def _draw_scene(surface, npcs, tree, font, phase):
    grass, walkable = {
        "day": (GRASS, WALKABLE_GRASS),
        "evening": ((190, 156, 120), (204, 176, 134)),
        "night": ((64, 77, 98), (77, 92, 111)),
    }[phase]
    surface.fill(grass)
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
    pygame.draw.ellipse(surface, (145, 171, 111), (tree_x - 43, round(tree.y) + 50, 86, 20))


def _draw_objects(surface, npcs, tree, font, character_renderer, camera_offset, phase, zoom):
    def screen_point(point):
        return (round(point[0] * zoom + camera_offset[0]),
                round(point[1] * zoom + camera_offset[1]))

    def resized(image, scale):
        size = (max(1, round(image.get_width() * scale)), max(1, round(image.get_height() * scale)))
        return image if image.get_size() == size else pygame.transform.smoothscale(image, size)

    label_color = PAPER if phase == "night" else INK
    name_font = npc_name_font(max(1, round(22 * zoom)))
    map_rect = pygame.Rect(screen_point((0, 0)), (round(WORLD_WIDTH * zoom), round(WORLD_HEIGHT * zoom)))
    tree_image, apple_image = tree_artwork()
    # NPC y is at their feet; use the bottom of the tree's trunk for the same comparison.
    objects = sorted([tree, *npcs], key=lambda obj: tree.y + TREE_BASE_OFFSET if obj is tree else obj.y)
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
        npc = obj
        position = screen_point((npc.x, npc.y))
        # Always resize from the original composed artwork, including when zooming out.
        original = character_renderer.render(npc.character, npc.facing)
        sprite = resized(original, CHARACTER_SCALE * zoom)
        sprite_rect = sprite.get_rect(midbottom=position)
        shadow = pygame.Rect(screen_point((npc.x - 16, npc.y - 4)), (round(32 * zoom), round(8 * zoom)))
        pygame.draw.ellipse(surface, MUTED, shadow)
        surface.blit(sprite, sprite_rect)
        name = name_font.render(npc.name, True, label_color)
        name_rect = name.get_rect(midtop=screen_point((npc.x, npc.y + 6)))
        name_rect.clamp_ip(map_rect)
        surface.blit(name, name_rect)
    # Draw above every character, anchored to the initiator's top right.
    for npc in npcs:
        if npc.conversation is not None and npc.conversation.arrived and npc.conversation.initiator is npc:
            width, height = npc.sprite_size or (
                round(character_renderer.character_size[0] * CHARACTER_SCALE),
                round(character_renderer.character_size[1] * CHARACTER_SCALE),
            )
            bubble = resized(talking_bubble(), zoom)
            rect = bubble.get_rect(bottomleft=screen_point((npc.x + width / 2 - 6, npc.y - height + 8)))
            surface.blit(bubble, rect)
