"""A clipped, scrollable view of committed simulation events."""

import pygame

from .text import wrap_text, scrollbar_rects, ScrollbarDrag
from constants import (ACCENT, EVENT_FEED_BODY_HEIGHT, EVENT_FEED_WIDTH, EVENT_FEED_TOOLBAR_GAP,
                       HUD_HEADER_HEIGHT, INK, MUTED, PAPER)


class EventFeed:
    def __init__(self):
        self.header = pygame.Rect(0, 0, 0, HUD_HEADER_HEIGHT)
        self.body = pygame.Rect(0, 0, 0, EVENT_FEED_BODY_HEIGHT)
        self.collapsed = True
        self.scroll = 0
        self.following = True
        self.unread = 0
        self.rows = []
        self.known_ids = set()
        self.limit = 0
        self.scrollbar = ScrollbarDrag()

    @property
    def rect(self):
        return self.header if self.collapsed else self.header.union(self.body)

    def position(self, app):
        controls = app.game.save_button.rect.unionall([app.game.stats_button.rect, app.game.menu_button.rect])
        # Keep event text readable when toolbar buttons only contain symbols.
        self.header.width = max(EVENT_FEED_WIDTH, controls.width)
        self.header.topright = (controls.right, app.game.toolbar_rect(app).bottom + EVENT_FEED_TOOLBAR_GAP)
        self.body.topleft = self.header.bottomleft
        self.body.width = self.header.width

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
        if event.type not in (pygame.MOUSEWHEEL, pygame.MOUSEBUTTONDOWN,
                              pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.WINDOWFOCUSLOST):
            return False
        self.position(app)
        pos = getattr(event, "pos", pygame.mouse.get_pos())
        if not self.rect.collidepoint(pos) and not self.scrollbar.active:
            return False
        # A drag that crosses the panel cannot select the world on release.
        app.game.pan_start = app.game.pan_last = None
        app.game.panning = False
        self.layout(app)
        value = self.scrollbar.handle_event(event, self.body, self.scroll, self.limit if not self.collapsed else 0)
        if value is not None:
            self.scroll = value
            self.following = self.scroll >= self.limit
            if self.following:
                self.unread = 0
            return True
        if event.type == pygame.MOUSEWHEEL and not self.collapsed:
            self.scroll = max(0, min(self.limit, self.scroll - event.y * 42))
            self.following = self.scroll >= self.limit
            if self.following:
                self.unread = 0
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.header.collidepoint(pos):
                self.collapsed = not self.collapsed
            elif not self.collapsed:
                for entry, top, height, _ in self.rows:
                    if top <= pos[1] - self.body.y + self.scroll < top + height and entry.history is not None:
                        app.open_stats()
                        app.stats.open_record(app.simulation.record_by_id(entry.history))
                        break
        return True

    def draw(self, app):
        self.position(app)
        self.layout(app)
        surface = app.window
        pygame.draw.rect(surface, PAPER, self.rect)
        pygame.draw.rect(surface, ACCENT, self.header)
        title = f"{'+' if self.collapsed else '-'} Events"
        if self.unread:
            title += f" ({self.unread} unread)"
        surface.blit(app.small_font.render(title, True, PAPER), (self.header.x + 10, self.header.y + 5))
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
            track, thumb = scrollbar_rects(self.body, self.scroll, self.limit)
            pygame.draw.rect(surface, MUTED, track)
            pygame.draw.rect(surface, ACCENT, thumb)
