"""Live model-worker capacity, conversation details, and elapsed time."""

import math
from time import monotonic

import pygame

from .theme import ACCENT, INK, MUTED, PAPER


class RequestStatus:
    HEADER_HEIGHT = 30
    SLOT_HEIGHT = 84

    def __init__(self, max_requests):
        self.rect = pygame.Rect(24, 16, 432, self.HEADER_HEIGHT + max_requests * self.SLOT_HEIGHT + 8)

    def draw(self, app):
        manager = app.requests
        workers = {worker.slot: worker for worker in manager.statuses()}
        surface, font = app.window, app.small_font
        now = monotonic()
        pygame.draw.rect(surface, PAPER, self.rect)
        pygame.draw.rect(surface, INK, self.rect, width=1)
        pygame.draw.rect(surface, ACCENT, (self.rect.x, self.rect.y, self.rect.width, self.HEADER_HEIGHT))
        title = f"LLM threads: {manager.available_slots}/{manager.max_requests} available"
        surface.blit(font.render(title, True, PAPER), (self.rect.x + 10, self.rect.y + 6))

        def line(text, x, y, width, color=INK):
            text = " ".join(text.split())
            if font.size(text)[0] > width:
                while text and font.size(text + "...")[0] > width:
                    text = text[:-1]
                text += "..."
            surface.blit(font.render(text, True, color), (x, y))

        for slot in range(1, manager.max_requests + 1):
            y = self.rect.y + self.HEADER_HEIGHT + 8 + (slot - 1) * self.SLOT_HEIGHT
            x, width = self.rect.x + 12, self.rect.width - 24
            worker = workers.get(slot)
            center = (x + 5, y + 7)
            if worker is None:
                pygame.draw.circle(surface, MUTED, center, 4, width=1)
                line(f"Thread {slot}: Available", x + 18, y, width - 18, MUTED)
                line("Ready for a conversation.", x, y + 20, width, MUTED)
                continue

            running = worker.finished_at is None
            if running:
                angle = now * 4
                pygame.draw.arc(surface, ACCENT, (x, y + 2, 10, 10), angle, angle + math.pi * 1.5, 2)
            else:
                pygame.draw.circle(surface, ACCENT, center, 4)
            state = "Running" if running else "Waiting to apply"
            if worker.session_id != app.session_id:
                state = "Previous session" if running else "Waiting to discard"
            elapsed = max(0, int((now if running else worker.finished_at) - worker.started_at))
            minutes, seconds = divmod(elapsed, 60)
            timer = font.render(f"{minutes}:{seconds:02d}", True, INK)
            timer_rect = timer.get_rect(topright=(self.rect.right - 12, y))
            surface.blit(timer, timer_rect)
            line(f"Thread {slot}: {state}", x + 18, y, timer_rect.left - x - 28)
            line(worker.participants, x, y + 20, width)
            line(worker.topic, x, y + 38, width, MUTED)
            line(worker.stage, x, y + 56, width, ACCENT)
