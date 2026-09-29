import pygame

from .theme import ACCENT, BORDER_RADIUS, PAPER


class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label

    def clicked(self, event, enabled=True):
        return (
            enabled
            and event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(event.pos)
        )

    def draw(self, surface, font, enabled=True):
        hovered = self.rect.collidepoint(pygame.mouse.get_pos())
        color = (66, 130, 91) if hovered else ACCENT
        if not enabled:
            color = (152, 164, 151)
        pygame.draw.rect(surface, color, self.rect, border_radius=BORDER_RADIUS)
        label = font.render(self.label, True, PAPER)
        surface.blit(label, label.get_rect(center=self.rect.center))
