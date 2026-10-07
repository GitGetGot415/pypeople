import pygame

from npc import WORLD_WIDTH, WORLD_HEIGHT

from .theme import GRASS, INK, MUTED


CHARACTER_SCALE = 0.5


def draw_world(surface, npcs, tree, font, character_renderer):
    surface.fill(GRASS)
    tree_x = round(tree.x) + tree.offset
    tree_y = round(tree.y)
    pygame.draw.ellipse(surface, (145, 171, 111), (tree_x - 43, tree_y + 50, 86, 20))
    pygame.draw.rect(surface, (134, 89, 55), (tree_x - 13, tree_y - 12, 26, 76))
    pygame.draw.circle(surface, (68, 124, 66), (tree_x, tree_y - 38), 53)
    pygame.draw.circle(surface, (83, 141, 72), (tree_x - 11, tree_y - 49), 31)
    for npc in sorted(npcs, key=lambda npc: npc.y):
        position = (round(npc.x), round(npc.y))
        sprite = character_renderer.render(npc.character, npc.facing, CHARACTER_SCALE)
        sprite_rect = sprite.get_rect(midbottom=position)
        pygame.draw.ellipse(surface, MUTED, (position[0] - 16, position[1] - 4, 32, 8))
        surface.blit(sprite, sprite_rect)
        name = font.render(npc.name, True, INK)
        name_rect = name.get_rect(midtop=(position[0], position[1] + 6))
        name_rect.clamp_ip(pygame.Rect(0, 0, WORLD_WIDTH, WORLD_HEIGHT))
        surface.blit(name, name_rect)
