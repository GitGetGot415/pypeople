import pygame

from character_data import DEFAULT_COLORS, FIELDS

from .button import Button
from .text import wrap_text
from .theme import ACCENT, GRASS, INK, MUTED, PAPER


class StatsScreen:
    """Browse live NPC profiles without editing them or leaving the simulation."""

    def __init__(self):
        self.back_button = Button((24, 16, 170, 38), "Back to game")
        self.tabs = {
            name: Button((352 + index * 176, 106, 160, 36), name)
            for index, name in enumerate(("Statistics", "Opinions"))
        }
        self.list_rect = pygame.Rect(40, 144, 264, 440)
        self.detail_rect = pygame.Rect(352, 160, 568, 424)
        self.open()

    def open(self):
        self.selected_npc = None
        self.tab = "Statistics"
        self.list_scroll = 0
        self.detail_scroll = 0
        self.scroll_focus = "list"

    def _npc_rows(self, app):
        rows = []
        top = 0
        for npc in app.npcs:
            lines = wrap_text(npc.name, app.font, self.list_rect.width - 30)
            height = max(48, len(lines) * app.font.get_linesize() + 20)
            rows.append((npc, top, height, lines))
            top += height + 8
        return rows, max(0, top - 8)

    def _detail_lines(self, app):
        lines = []

        def add(text, font=app.font, color=INK):
            for paragraph in str(text).splitlines() or [""]:
                for line in wrap_text(paragraph, font, self.detail_rect.width - 20) or [""]:
                    lines.append((line, font, color))

        def gap():
            lines.append(("", app.small_font, MUTED))

        npc = self.selected_npc
        if npc is None:
            add("Select an NPC to view their statistics and opinions.", color=MUTED)
            return lines
        add(npc.name, app.heading_font)
        gap()
        if self.tab == "Statistics":
            fields = tuple(field for field in FIELDS if field != "Details")
            for field in (*fields, "Hair", "Eyes", *DEFAULT_COLORS, "Details"):
                value = npc.name if field == "Name" else npc.character.get(field, "")
                if field == "Details":
                    gap()
                    add("Details", color=MUTED)
                    add(value or "Not specified")
                    gap()
                else:
                    add(f"{field}: {value or 'Not specified'}")
        else:
            add("Opinion scores range from -100 to 100.", app.small_font, MUTED)
            gap()
            opinions = npc.opinions
            if not opinions:
                add("No opinions recorded yet.", color=MUTED)
            for person, entry in sorted(opinions.items(), key=lambda item: item[0].casefold()):
                add(person, app.heading_font)
                add(f"Opinion: {entry['opinion']:+d}")
                add(f"Reason: {entry['reason']}")
                gap()
        return lines

    def _scroll_limits(self, app):
        _, list_height = self._npc_rows(app)
        detail_height = sum(font.get_linesize() for _, font, _ in self._detail_lines(app))
        return {
            "list": max(0, list_height - self.list_rect.height),
            "detail": max(0, detail_height - self.detail_rect.height),
        }

    def _scroll(self, pane, value, limits):
        setattr(self, f"{pane}_scroll", max(0, min(round(value), limits[pane])))

    @staticmethod
    def _scrollbar(rect, scroll, limit):
        track = pygame.Rect(rect.right - 8, rect.top, 8, rect.height)
        height = max(24, round(rect.height * rect.height / (rect.height + limit)))
        top = track.top + round((track.height - height) * scroll / limit) if limit else track.top
        return track, pygame.Rect(track.x, top, track.width, height)

    def handle_event(self, app, event):
        if self.back_button.clicked(event) or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            app.screen = "game"
            return
        for name, button in self.tabs.items():
            if button.clicked(event):
                self.tab = name
                self.detail_scroll = 0
                self.scroll_focus = "detail"
                return

        limits = self._scroll_limits(app)
        for pane in limits:
            self._scroll(pane, getattr(self, f"{pane}_scroll"), limits)
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEWHEEL):
            pos = getattr(event, "pos", pygame.mouse.get_pos())
            for pane, rect in (("list", self.list_rect), ("detail", self.detail_rect)):
                if not rect.collidepoint(pos):
                    continue
                self.scroll_focus = pane
                scroll = getattr(self, f"{pane}_scroll")
                if event.type == pygame.MOUSEWHEEL:
                    self._scroll(pane, scroll - event.y * 48, limits)
                    return
                if event.button != 1:
                    return
                track, thumb = self._scrollbar(rect, scroll, limits[pane])
                if limits[pane] and track.collidepoint(pos):
                    offset = pos[1] - track.top - thumb.height / 2
                    self._scroll(pane, offset * limits[pane] / (track.height - thumb.height), limits)
                    return
                if pane == "list":
                    rows, _ = self._npc_rows(app)
                    for npc, top, height, _ in rows:
                        if top <= pos[1] - rect.top + scroll < top + height:
                            self.selected_npc = npc
                            self.detail_scroll = 0
                            self.scroll_focus = "detail"
                            return
        elif event.type == pygame.KEYDOWN:
            pane = self.scroll_focus
            scroll = getattr(self, f"{pane}_scroll")
            rect = self.list_rect if pane == "list" else self.detail_rect
            steps = {
                pygame.K_UP: -32, pygame.K_DOWN: 32,
                pygame.K_PAGEUP: -rect.height, pygame.K_PAGEDOWN: rect.height,
                pygame.K_HOME: -scroll, pygame.K_END: limits[pane] - scroll,
            }
            if event.key in steps:
                self._scroll(pane, scroll + steps[event.key], limits)

    def draw(self, app):
        surface = app.window
        surface.fill(GRASS)
        self.back_button.draw(surface, app.small_font)
        surface.blit(app.heading_font.render("NPC Stats", True, INK), (220, 21))
        surface.blit(app.small_font.render("Read only. Select anyone to browse their profile.", True, MUTED), (24, 68))
        pygame.draw.rect(surface, PAPER, (24, 94, 296, 506))
        pygame.draw.rect(surface, PAPER, (336, 94, 600, 506))
        surface.blit(app.font.render(f"NPCs ({len(app.npcs)})", True, INK), (40, 110))
        for name, button in self.tabs.items():
            button.draw(surface, app.small_font)
            if self.tab == name:
                pygame.draw.line(surface, PAPER, (button.rect.left + 16, 136), (button.rect.right - 16, 136), 2)

        limits = self._scroll_limits(app)
        for pane in limits:
            self._scroll(pane, getattr(self, f"{pane}_scroll"), limits)
        previous_clip = surface.get_clip()
        surface.set_clip(self.list_rect.clip(previous_clip))
        rows, _ = self._npc_rows(app)
        for npc, top, height, lines in rows:
            row = pygame.Rect(self.list_rect.x, self.list_rect.y + top - self.list_scroll, self.list_rect.width - 14, height)
            if not row.colliderect(self.list_rect):
                continue
            selected = npc is self.selected_npc
            pygame.draw.rect(surface, ACCENT if selected else GRASS, row)
            for index, line in enumerate(lines):
                surface.blit(app.font.render(line, True, PAPER if selected else INK), (row.x + 8, row.y + 10 + index * app.font.get_linesize()))
        surface.set_clip(self.detail_rect.clip(previous_clip))
        y = self.detail_rect.top - self.detail_scroll
        for line, font, color in self._detail_lines(app):
            if self.detail_rect.top - font.get_linesize() < y < self.detail_rect.bottom:
                surface.blit(font.render(line, True, color), (self.detail_rect.x, y))
            y += font.get_linesize()
        surface.set_clip(previous_clip)
        for pane, rect in (("list", self.list_rect), ("detail", self.detail_rect)):
            if limits[pane]:
                track, thumb = self._scrollbar(rect, getattr(self, f"{pane}_scroll"), limits[pane])
                pygame.draw.rect(surface, GRASS, track)
                pygame.draw.rect(surface, MUTED, thumb)
        surface.blit(app.small_font.render("Scroll or use Up/Down and Page Up/Down. Escape returns to the game.", True, MUTED), (24, 615))
