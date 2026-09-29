import pygame

from npc import NPC_RADIUS, WORLD_WIDTH, WORLD_HEIGHT

from .theme import GRASS, INK


def draw_world(surface, npc, tree, font):
    surface.fill(GRASS)
    tree_x = round(tree.x) + tree.offset
    tree_y = round(tree.y)
    pygame.draw.ellipse(surface, (145, 171, 111), (tree_x - 43, tree_y + 50, 86, 20))
    pygame.draw.rect(surface, (134, 89, 55), (tree_x - 13, tree_y - 12, 26, 76))
    pygame.draw.circle(surface, (68, 124, 66), (tree_x, tree_y - 38), 53)
    pygame.draw.circle(surface, (83, 141, 72), (tree_x - 11, tree_y - 49), 31)
    judy = (round(npc.x), round(npc.y))
    pygame.draw.circle(surface, (88, 156, 214), judy, NPC_RADIUS)
    pygame.draw.circle(surface, (41, 89, 133), judy, NPC_RADIUS, width=2)
    name = font.render("Judy", True, INK)
    name_rect = name.get_rect(midtop=(judy[0], judy[1] + NPC_RADIUS + 6))
    name_rect.clamp_ip(pygame.Rect(0, 0, WORLD_WIDTH, WORLD_HEIGHT))
    surface.blit(name, name_rect)
