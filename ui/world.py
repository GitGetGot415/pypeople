from pathlib import Path

import pygame

from npc import WORLD_WIDTH, WORLD_HEIGHT

from .theme import GRASS, INK, MUTED, WALKABLE_GRASS


CHARACTER_SCALE = 0.5
_talking_bubble = None


def talking_bubble():
    global _talking_bubble
    if _talking_bubble is None:
        path = Path(__file__).resolve().parent.parent / "assets" / "Talking.png"
        _talking_bubble = pygame.image.load(str(path)).convert_alpha()
    return _talking_bubble


def draw_world(surface, npcs, tree, font, character_renderer, camera_offset=(0, 0)):
    surface.fill(GRASS)
    offset_x, offset_y = camera_offset
    if npcs:
        left, right, top, bottom = npcs[0].movement_bounds
        walkable_rect = pygame.Rect(
            round(left + offset_x),
            round(top + offset_y),
            round(right - left),
            round(bottom - top),
        )
        pygame.draw.rect(surface, WALKABLE_GRASS, walkable_rect)

    tree_x = round(tree.x + offset_x) + tree.offset
    tree_y = round(tree.y + offset_y)
    pygame.draw.ellipse(surface, (145, 171, 111), (tree_x - 43, tree_y + 50, 86, 20))
    pygame.draw.rect(surface, (134, 89, 55), (tree_x - 13, tree_y - 12, 26, 76))
    pygame.draw.circle(surface, (68, 124, 66), (tree_x, tree_y - 38), 53)
    pygame.draw.circle(surface, (83, 141, 72), (tree_x - 11, tree_y - 49), 31)
    for npc in sorted(npcs, key=lambda npc: npc.y):
        position = (round(npc.x + offset_x), round(npc.y + offset_y))
        sprite = character_renderer.render(npc.character, npc.facing, CHARACTER_SCALE)
        sprite_rect = sprite.get_rect(midbottom=position)
        pygame.draw.ellipse(surface, MUTED, (position[0] - 16, position[1] - 4, 32, 8))
        surface.blit(sprite, sprite_rect)
        name = font.render(npc.name, True, INK)
        name_rect = name.get_rect(midtop=(position[0], position[1] + 6))
        name_rect.clamp_ip(pygame.Rect(0, 0, WORLD_WIDTH, WORLD_HEIGHT))
        surface.blit(name, name_rect)
    # Draw above every character, anchored to the initiator's top right.
    for npc in npcs:
        if npc.conversation is not None and npc.conversation.arrived and npc.conversation.initiator is npc:
            width, height = npc.sprite_size or character_renderer.render(npc.character, npc.facing, CHARACTER_SCALE).get_size()
            bubble = talking_bubble()
            rect = bubble.get_rect(bottomleft=(round(npc.x + offset_x + width / 2 - 6), round(npc.y + offset_y - height + 8)))
            surface.blit(bubble, rect)
