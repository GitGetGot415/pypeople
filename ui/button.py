import pygame

from constants import (ACCENT, BORDER_RADIUS, PAPER, BUTTON_ICON_SIZE,
                       BUTTON_ICON_LABEL_PADDING, BUTTON_ICON_LABEL_GAP)


class Button:
    def __init__(self, rect, label, *, icon=None, show_label=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.icon = icon
        self.show_label = show_label

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
        if self.icon is not None:
            if self.show_label:
                icon_left = self.rect.left + BUTTON_ICON_LABEL_PADDING
                self._draw_icon(surface, (icon_left + BUTTON_ICON_SIZE // 2, self.rect.centery))
                label = font.render(self.label, True, PAPER)
                surface.blit(label, label.get_rect(
                    midleft=(icon_left + BUTTON_ICON_SIZE + BUTTON_ICON_LABEL_GAP, self.rect.centery)))
            else:
                self._draw_icon(surface)
            return
        label = font.render(self.label, True, PAPER)
        surface.blit(label, label.get_rect(center=self.rect.center))

    def _draw_icon(self, surface, center=None):
        """Draw toolbar symbols with the same crisp strokes as the clock."""
        x, y = self.rect.center if center is None else center
        if self.icon == "clock":
            pygame.draw.circle(surface, PAPER, (x, y), 12, width=2)
            pygame.draw.line(surface, PAPER, (x, y), (x, y - 8), 2)
            pygame.draw.line(surface, PAPER, (x, y), (x + 6, y + 3), 2)
        elif self.icon == "inventory":
            pygame.draw.rect(surface, PAPER, (x - 4, y - 13, 8, 5), width=2, border_radius=2)
            pygame.draw.rect(surface, PAPER, (x - 10, y - 9, 20, 22), width=2, border_radius=5)
            pygame.draw.rect(surface, PAPER, (x - 6, y + 2, 12, 7), width=2, border_radius=2)
            pygame.draw.line(surface, PAPER, (x - 6, y - 3), (x + 6, y - 3), 2)
        elif self.icon == "pause":
            pygame.draw.rect(surface, PAPER, (x - 8, y - 10, 5, 20))
            pygame.draw.rect(surface, PAPER, (x + 3, y - 10, 5, 20))
        elif self.icon == "play":
            pygame.draw.polygon(surface, PAPER, ((x - 7, y - 11), (x + 10, y), (x - 7, y + 11)))
        elif self.icon == "save":
            pygame.draw.polygon(surface, PAPER, (
                (x - 11, y - 12), (x + 6, y - 12), (x + 11, y - 7),
                (x + 11, y + 12), (x - 11, y + 12)), width=2)
            pygame.draw.rect(surface, PAPER, (x - 6, y - 11, 12, 8), width=2)
            pygame.draw.rect(surface, PAPER, (x - 7, y + 2, 14, 10), width=2)
        elif self.icon == "stats":
            pygame.draw.line(surface, PAPER, (x - 12, y + 11), (x + 12, y + 11), 2)
            for left, top, height in ((x - 10, y + 3, 7), (x - 2, y - 3, 13), (x + 6, y - 10, 20)):
                pygame.draw.rect(surface, PAPER, (left, top, 5, height))
        elif self.icon == "menu":
            pygame.draw.lines(surface, PAPER, False,
                              ((x - 12, y - 1), (x, y - 12), (x + 12, y - 1)), 2)
            pygame.draw.lines(surface, PAPER, False, (
                (x - 9, y - 3), (x - 9, y + 11), (x - 3, y + 11), (x - 3, y + 3),
                (x + 3, y + 3), (x + 3, y + 11), (x + 9, y + 11), (x + 9, y - 3)), 2)
