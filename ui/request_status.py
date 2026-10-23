"""Live model-worker capacity, conversation details, and elapsed time."""

import math
from time import monotonic

import pygame

from constants import ACCENT, HUD_HEADER_HEIGHT, INK, LLM_PANEL_WIDTH, LLM_SLOT_HEIGHT, MUTED, PAPER
from .button import Button
from .text import scrollbar_rects, ScrollbarDrag


class RequestStatus:
    def __init__(self, max_requests):
        self.rect = pygame.Rect(24, 16, LLM_PANEL_WIDTH, 0)
        self.header = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, HUD_HEADER_HEIGHT)
        self.collapsed = False
        self.slot_scroll = 0
        self.scrollbar = ScrollbarDrag()
        self.set_capacity(max_requests)

    def set_capacity(self, max_requests):
        self.max_requests = max_requests
        self.visible_slots = min(max_requests, 5)
        self.slot_scroll = min(self.slot_scroll, max_requests - self.visible_slots)
        self.expanded_height = HUD_HEADER_HEIGHT + self.visible_slots * LLM_SLOT_HEIGHT + 8
        if max_requests > self.visible_slots:
            self.expanded_height += 20
        self.rect.height = HUD_HEADER_HEIGHT if self.collapsed else self.expanded_height

    def output_button(self, slot):
        y = self.rect.y + HUD_HEADER_HEIGHT + 8 + (slot - 1 - self.slot_scroll) * LLM_SLOT_HEIGHT
        return Button((self.rect.right - 64, y + 23, 54, 26), "View")

    def cancel_button(self, slot):
        y = self.rect.y + HUD_HEADER_HEIGHT + 8 + (slot - 1 - self.slot_scroll) * LLM_SLOT_HEIGHT
        return Button((self.rect.right - 34, y - 2, 24, 22), "X")

    @property
    def body(self):
        return pygame.Rect(self.rect.x, self.header.bottom, self.rect.width,
                           self.visible_slots * LLM_SLOT_HEIGHT + 8)

    def handle_event(self, app, event):
        if event.type not in (pygame.MOUSEWHEEL, pygame.MOUSEBUTTONDOWN,
                              pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.WINDOWFOCUSLOST):
            return False
        pos = getattr(event, "pos", pygame.mouse.get_pos())
        if not self.rect.collidepoint(pos) and not self.scrollbar.active:
            return False
        app.game.pan_start = app.game.pan_last = None
        app.game.panning = False
        limit = (self.max_requests - self.visible_slots) * LLM_SLOT_HEIGHT if not self.collapsed else 0
        value = self.scrollbar.handle_event(event, self.body, self.slot_scroll * LLM_SLOT_HEIGHT, limit)
        if value is not None:
            self.slot_scroll = round(value / LLM_SLOT_HEIGHT)
            return True
        if event.type == pygame.MOUSEWHEEL and not self.collapsed:
            self.slot_scroll = max(0, min(self.max_requests - self.visible_slots, self.slot_scroll - event.y))
            return True
        if not self.collapsed:
            for worker in app.requests.statuses():
                if not self.slot_scroll < worker.slot <= self.slot_scroll + self.visible_slots:
                    continue
                if self.cancel_button(worker.slot).clicked(event):
                    app.cancel_model_request(worker.session_id, worker.request_id)
                    return True
                if worker.log is not None and self.output_button(worker.slot).clicked(event):
                    app.open_stats()
                    app.stats.open_log(worker.log)
                    return True
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.header.collidepoint(pos):
            self.collapsed = not self.collapsed
            self.rect.height = HUD_HEADER_HEIGHT if self.collapsed else self.expanded_height
        return True

    def draw(self, app):
        manager = app.requests
        workers = {worker.slot: worker for worker in manager.statuses()}
        self.set_capacity(max(manager.max_requests, max(workers, default=0)))
        surface, font = app.window, app.small_font
        now = monotonic()
        pygame.draw.rect(surface, PAPER, self.rect)
        pygame.draw.rect(surface, INK, self.rect, width=1)
        pygame.draw.rect(surface, ACCENT, self.header)
        title = f"{'+' if self.collapsed else '-'} LLM calls: {manager.admission.active}/{manager.max_requests}; exchanges: {len(workers)}"
        surface.blit(font.render(title, True, PAPER), (self.rect.x + 10, self.rect.y + 6))
        if self.collapsed:
            return

        def line(text, x, y, width, color=INK):
            text = " ".join(text.split())
            if font.size(text)[0] > width:
                while text and font.size(text + "...")[0] > width:
                    text = text[:-1]
                text += "..."
            surface.blit(font.render(text, True, color), (x, y))

        if self.max_requests > self.visible_slots:
            label = f"Scroll for threads {self.slot_scroll + 1}-{self.slot_scroll + self.visible_slots} of {self.max_requests}"
            surface.blit(font.render(label, True, MUTED), (self.rect.x + 12, self.rect.bottom - 22))
        for slot in range(self.slot_scroll + 1, self.slot_scroll + self.visible_slots + 1):
            y = self.rect.y + HUD_HEADER_HEIGHT + 8 + (slot - 1 - self.slot_scroll) * LLM_SLOT_HEIGHT
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
            state = ("Generating" if worker.control and worker.control.provider_active else "Suspended / preparing") if running else "Waiting to apply"
            if worker.session_id != app.session_id:
                state = "Previous session" if running else "Waiting to discard"
            elapsed = max(0, int((now if running else worker.finished_at) - worker.started_at))
            minutes, seconds = divmod(elapsed, 60)
            timer = font.render(f"{minutes}:{seconds:02d}", True, INK)
            timer_rect = timer.get_rect(topright=(self.rect.right - 42, y))
            surface.blit(timer, timer_rect)
            self.cancel_button(slot).draw(surface, font)
            line(f"Thread {slot}: {state}", x + 18, y, timer_rect.left - x - 28)
            self.output_button(slot).draw(surface, font, enabled=worker.log is not None)
            line(worker.participants, x, y + 20, width - 62)
            line(worker.topic, x, y + 38, width - 62, MUTED)
            line(worker.stage, x, y + 56, width, ACCENT)
        if self.max_requests > self.visible_slots:
            track, thumb = scrollbar_rects(self.body, self.slot_scroll * LLM_SLOT_HEIGHT,
                                           (self.max_requests - self.visible_slots) * LLM_SLOT_HEIGHT)
            pygame.draw.rect(surface, MUTED, track)
            pygame.draw.rect(surface, ACCENT, thumb)
