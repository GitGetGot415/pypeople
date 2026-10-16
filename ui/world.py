from pathlib import Path

import pygame

from npc import WORLD_WIDTH, WORLD_HEIGHT

from .theme import GRASS, INK, MUTED, PAPER, WALKABLE_GRASS


CHARACTER_SCALE = 0.5
_talking_bubble = None


def talking_bubble():
    global _talking_bubble
    if _talking_bubble is None:
        path = Path(__file__).resolve().parent.parent / "assets" / "Talking.png"
        _talking_bubble = pygame.image.load(str(path)).convert_alpha()
    return _talking_bubble


def draw_world(surface, npcs, tree, font, character_renderer, camera_offset=(0, 0), phase="day", zoom=1.0):
    # Scale the scenery separately so characters never pass through a reduced map image.
    scene = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))
    _draw_scene(scene, npcs, tree, font, phase)
    surface.fill(scene.get_at((0, 0)))
    if zoom != 1.0:
        scene = pygame.transform.smoothscale(scene, (round(WORLD_WIDTH * zoom), round(WORLD_HEIGHT * zoom)))
    previous_clip = surface.get_clip()
    scene_rect = scene.get_rect(topleft=(round(camera_offset[0]), round(camera_offset[1])))
    surface.set_clip(pygame.Rect(0, 0, WORLD_WIDTH, WORLD_HEIGHT).clip(previous_clip).clip(scene_rect))
    surface.blit(scene, scene_rect)
    _draw_people(surface, npcs, font, character_renderer, camera_offset, phase, zoom)
    surface.set_clip(previous_clip)


def _draw_scene(surface, npcs, tree, font, phase):
    grass, walkable = {
        "day": (GRASS, WALKABLE_GRASS),
        "evening": ((190, 156, 120), (204, 176, 134)),
        "night": ((64, 77, 98), (77, 92, 111)),
    }[phase]
    label_color = PAPER if phase == "night" else INK
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
    tree_y = round(tree.y)
    pygame.draw.ellipse(surface, (145, 171, 111), (tree_x - 43, tree_y + 50, 86, 20))
    pygame.draw.rect(surface, (134, 89, 55), (tree_x - 13, tree_y - 12, 26, 76))
    pygame.draw.circle(surface, (68, 124, 66), (tree_x, tree_y - 38), 53)
    pygame.draw.circle(surface, (83, 141, 72), (tree_x - 11, tree_y - 49), 31)
    fruit_points = [(-30, -43), (5, -65), (28, -40), (-10, -27), (10, -12), (-32, -13), (28, -66), (-20, -72)]
    for dx, dy in fruit_points[:tree.fruit]:
        pygame.draw.circle(surface, (205, 65, 48), (tree_x + dx, tree_y + dy), 6)
    label = font.render(f"Fruit {tree.fruit}/{tree.capacity}", True, label_color)
    surface.blit(label, label.get_rect(midtop=(tree_x, tree_y + 76)))


def _draw_people(surface, npcs, font, character_renderer, camera_offset, phase, zoom):
    def screen_point(point):
        return (round(point[0] * zoom + camera_offset[0]),
                round(point[1] * zoom + camera_offset[1]))

    def resized(image, scale):
        size = (max(1, round(image.get_width() * scale)), max(1, round(image.get_height() * scale)))
        return image if image.get_size() == size else pygame.transform.smoothscale(image, size)

    label_color = PAPER if phase == "night" else INK
    for npc in sorted(npcs, key=lambda npc: npc.y):
        position = screen_point((npc.x, npc.y))
        # Always resize from the original composed artwork, including when zooming out.
        original = character_renderer.render(npc.character, npc.facing)
        sprite = resized(original, CHARACTER_SCALE * zoom)
        sprite_rect = sprite.get_rect(midbottom=position)
        shadow = pygame.Rect(screen_point((npc.x - 16, npc.y - 4)), (round(32 * zoom), round(8 * zoom)))
        pygame.draw.ellipse(surface, MUTED, shadow)
        surface.blit(sprite, sprite_rect)
        name = font.render(npc.name, True, label_color)
        name_rect = name.get_rect(midtop=(round(npc.x), round(npc.y) + 6))
        name_rect.clamp_ip(pygame.Rect(0, 0, WORLD_WIDTH, WORLD_HEIGHT))
        surface.blit(resized(name, zoom), screen_point(name_rect.topleft))
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
