"""A clipped, scrollable view of committed simulation events."""

import pygame

from .text import wrap_text
from constants import ACCENT, INK, MUTED, PAPER


class EventFeed:
    def __init__(self):
        self.header = pygame.Rect(24, 72, 432, 30)
        self.body = pygame.Rect(24, 102, 432, 142)
        self.collapsed = False
        self.scroll = 0
        self.following = True
        self.unread = 0
        self.rows = []
        self.known_ids = set()
        self.limit = 0
        self.drag_start = None
        self.drag_offset = None
        self.dragging = False

    @property
    def rect(self):
        return self.header if self.collapsed else self.header.union(self.body)

    def keep_visible(self, app):
        # Leave the worker display, top controls, and open dialogue accessible.
        viewport = app.game.world_viewport
        top = app.game.request_status.rect.bottom + 12
        bounds = pygame.Rect(viewport.left, top, viewport.width, viewport.bottom - top)
        self.header.topleft = self.rect.clamp(bounds).topleft
        self.body.topleft = self.header.bottomleft

    def layout(self, app):
        events = app.simulation.events
        ids = {event.id for event in events}
        new_count = len(ids - self.known_ids)
        if self.collapsed or not self.following:
            self.unread += new_count
        # Removing old rows must not move the event the reader is looking at.
        removed_height = sum(height for event, _, height, _ in self.rows if event.id not in ids)
        if not self.following:
            self.scroll = max(0, self.scroll - removed_height)
        self.known_ids = ids
        self.rows = []
        top = 6
        for event in events:
            text = f"Day {event.day} / {event.phase.title()} - {event.text}"
            if event.history is not None:
                text += " [view]"
            lines = wrap_text(text, app.small_font, self.body.width - 28)
            height = len(lines) * app.small_font.get_linesize() + 10
            self.rows.append((event, top, height, lines))
            top += height
        self.limit = max(0, top - self.body.height)
        self.scroll = self.limit if self.following else min(self.scroll, self.limit)
        if self.following and not self.collapsed:
            self.unread = 0

    def handle_event(self, app, event):
        if event.type == pygame.WINDOWFOCUSLOST:
            self.drag_start = self.drag_offset = None
            self.dragging = False
            return False
        if event.type not in (pygame.MOUSEWHEEL, pygame.MOUSEBUTTONDOWN,
                              pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            return False
        self.keep_visible(app)
        pos = getattr(event, "pos", pygame.mouse.get_pos())
        if self.drag_start is None and not self.rect.collidepoint(pos):
            return False
        # A drag that crosses the panel cannot select the world on release.
        app.game.pan_start = app.game.pan_last = None
        app.game.panning = False
        if self.drag_start is not None:
            if event.type == pygame.MOUSEMOTION:
                dx, dy = pos[0] - self.drag_start[0], pos[1] - self.drag_start[1]
                if self.dragging or dx * dx + dy * dy >= 16:
                    self.dragging = True
                    self.header.topleft = (pos[0] - self.drag_offset[0], pos[1] - self.drag_offset[1])
                    self.body.topleft = self.header.bottomleft
                    self.keep_visible(app)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if not self.dragging and self.header.collidepoint(pos):
                    self.collapsed = not self.collapsed
                    self.keep_visible(app)
                self.drag_start = self.drag_offset = None
                self.dragging = False
            return True
        self.layout(app)
        if event.type == pygame.MOUSEWHEEL and not self.collapsed:
            self.scroll = max(0, min(self.limit, self.scroll - event.y * 42))
            self.following = self.scroll >= self.limit
            if self.following:
                self.unread = 0
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.header.collidepoint(pos):
                self.drag_start = pos
                self.drag_offset = (pos[0] - self.header.x, pos[1] - self.header.y)
                self.dragging = False
            elif not self.collapsed:
                for entry, top, height, _ in self.rows:
                    if top <= pos[1] - self.body.y + self.scroll < top + height and entry.history is not None:
                        app.open_stats()
                        app.stats.open_record(app.conversation_history[entry.history])
                        break
        return True

    def draw(self, app):
        self.keep_visible(app)
        self.layout(app)
        surface = app.window
        pygame.draw.rect(surface, PAPER, self.rect)
        pygame.draw.rect(surface, ACCENT, self.header)
        title = f"{'+' if self.collapsed else '-'} Events"
        if self.unread:
            title += f" ({self.unread} unread)"
        surface.blit(app.small_font.render(title, True, PAPER), (self.header.x + 10, self.header.y + 5))
        hint = app.small_font.render("Drag to move", True, PAPER)
        surface.blit(hint, hint.get_rect(topright=(self.header.right - 10, self.header.y + 5)))
        if self.collapsed:
            return
        previous_clip = surface.get_clip()
        surface.set_clip(self.body.clip(previous_clip))
        if not self.rows:
            surface.blit(app.small_font.render("Pass time to see what happens.", True, MUTED), (self.body.x + 10, self.body.y + 12))
        for entry, top, height, lines in self.rows:
            y = self.body.y + top - self.scroll
            if y + height <= self.body.top or y >= self.body.bottom:
                continue
            color = ACCENT if entry.history is not None else INK
            for line in lines:
                surface.blit(app.small_font.render(line, True, color), (self.body.x + 10, y))
                y += app.small_font.get_linesize()
        surface.set_clip(previous_clip)
        if self.limit:
            height = max(12, round(self.body.height ** 2 / (self.limit + self.body.height)))
            top = self.body.y + round((self.body.height - height) * self.scroll / self.limit)
            pygame.draw.rect(surface, MUTED, (self.body.right - 6, top, 4, height))
