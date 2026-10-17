import pygame

from constants import ACCENT, GRASS, INK, MUTED, PAPER


class ColorSlider:
    """A 0-255 color channel controlled by dragging or the arrow keys."""

    def __init__(self, rect, label, value=0):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.value = value
        self.active = False
        self.dragging = False

    @property
    def track(self):
        return pygame.Rect(self.rect.x + 28, self.rect.centery - 3, self.rect.width - 76, 6)

    def _set_from_position(self, x):
        track = self.track
        fraction = max(0.0, min(1.0, (x - track.left) / (track.width - 1)))
        self.value = round(fraction * 255)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.active = self.rect.collidepoint(event.pos)
            self.dragging = self.active
            if self.dragging:
                self._set_from_position(event.pos[0])
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_position(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self._set_from_position(event.pos[0])
            self.dragging = False
        elif event.type == pygame.KEYDOWN and self.active:
            step = 10 if event.mod & pygame.KMOD_SHIFT else 1
            if event.key in (pygame.K_LEFT, pygame.K_DOWN):
                self.value = max(0, self.value - step)
            elif event.key in (pygame.K_RIGHT, pygame.K_UP):
                self.value = min(255, self.value + step)
            elif event.key == pygame.K_HOME:
                self.value = 0
            elif event.key == pygame.K_END:
                self.value = 255

    def draw(self, surface, font):
        track = self.track
        thumb_x = track.left + round(self.value / 255 * (track.width - 1))
        pygame.draw.rect(surface, GRASS, track)
        pygame.draw.rect(surface, ACCENT, (track.left, track.top, thumb_x - track.left + 1, track.height))
        if self.active:
            pygame.draw.circle(surface, ACCENT, (thumb_x, track.centery), 10, 2)
        pygame.draw.circle(surface, PAPER, (thumb_x, track.centery), 6)
        pygame.draw.circle(surface, INK, (thumb_x, track.centery), 6, 2)
        label = font.render(self.label, True, INK)
        value = font.render(str(self.value), True, MUTED)
        surface.blit(label, label.get_rect(midleft=(self.rect.left, self.rect.centery)))
        surface.blit(value, value.get_rect(midright=(self.rect.right, self.rect.centery)))
