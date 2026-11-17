from ui.stats_details import StatsDetails
import pygame
from simulation.interactions import player_request_pending
from dialogue.history import has_failed_voice_rewrite
from simulation.npc import NPC
from ui.creations import CreationsGallery
from ui.widgets.button import Button
from ui.text import wrap_text, scrollbar_rects, ScrollbarDrag
from ui.rendering.character_sprite import draw_actor_portrait
from constants import (
    STATS_PORTRAIT_PADDING, CONVERSATION_FAILED, ACCENT, FAILED_CONVERSATION_COLOR,
    FAILED_CONVERSATION_SELECTED_COLOR, GENERATING_CONVERSATION_COLOR,
    GENERATING_CONVERSATION_SELECTED_COLOR, GIFT_CONVERSATION_COLOR, GIFT_CONVERSATION_SELECTED_COLOR,
    QUEUED_CONVERSATION_COLOR, QUEUED_CONVERSATION_SELECTED_COLOR, REWRITE_FAILED_CONVERSATION_COLOR,
    REWRITE_FAILED_CONVERSATION_SELECTED_COLOR, GRASS, INK, MODEL_LOG_COLORS, MUTED,
    ORIGINAL_DIALOGUE_PAPER, PAPER, STATS_TAB_RECTS, STATS_FOLLOW_RECT, STATS_INVENTORY_RECT,
    STATS_INVENTORY_DETAIL_TOP, STATS_SORT_RECT, OPINION_BAR_LEFT, OPINION_BAR_WIDTH,
    OPINION_POSITIVE_COLOR, OPINION_NEGATIVE_COLOR, PLAYER_LABEL, STATS_VIEW_RECT, STATS_SUBTAB_RECTS,
    GRID_LABEL_WIDTH, STATS_DEAD_COLOR, STATS_DEAD_SELECTED_COLOR,
)


# Row colors by conversation state: (unselected, selected).
CONVERSATION_COLORS = {
    "generating": (GENERATING_CONVERSATION_COLOR, GENERATING_CONVERSATION_SELECTED_COLOR),
    "queued": (QUEUED_CONVERSATION_COLOR, QUEUED_CONVERSATION_SELECTED_COLOR),
    "failed": (FAILED_CONVERSATION_COLOR, FAILED_CONVERSATION_SELECTED_COLOR),
    "rewrite_failed": (REWRITE_FAILED_CONVERSATION_COLOR, REWRITE_FAILED_CONVERSATION_SELECTED_COLOR),
    "gift": (GIFT_CONVERSATION_COLOR, GIFT_CONVERSATION_SELECTED_COLOR),
    "normal": (GRASS, ACCENT),
}


class StatsScreen:
    """Browse NPCs, conversation history, and live or saved model diagnostics."""

    def __init__(self):
        self.back_button = Button((184, 56, 170, 38), "Back to game")
        self.tabs = {name: Button(rect, name) for name, rect in zip(
            ("Statistics", "Opinions", "Conversations", "Creations", "Model logs"), STATS_TAB_RECTS)}
        self.gallery = CreationsGallery()
        self.inventory_button = Button(STATS_INVENTORY_RECT, "Inventory")
        self.original_toggle = Button((888, 102, 192, 30), "Original dialogue: Off")
        self.condense_toggle = Button((888, 66, 192, 30), "Condense: Off")
        self.follow_toggle = Button(STATS_FOLLOW_RECT, "Follow")
        self.sort_toggle = Button(STATS_SORT_RECT, "Sort: Name")
        self.view_toggle = Button(STATS_VIEW_RECT, "View: Table")
        self.subtabs = {name: Button(rect, name) for name, rect in zip(
            ("Overview", "Profile", "Social", "Observer"), STATS_SUBTAB_RECTS)}
        self.list_rect = pygame.Rect(200, 184, 264, 440)
        self.detail_rect = pygame.Rect(512, 200, 568, 424)
        self.details = StatsDetails(self)
        self.open()

    def open(self):
        self.gallery.close()
        self.gallery.selected_id = None
        self.gallery.filter = "All"
        self.gallery.status = ""
        self._creation_order = set()
        self.selected_npc = None
        self.selected_conversation = None
        self.selected_log = None
        self.follow_log = True
        self._log_cache = None
        self.tab = "Statistics"
        self.list_scroll = 0
        self.detail_scroll = 0
        self.scroll_focus = "list"
        self.scrollbars = {pane: ScrollbarDrag() for pane in ("list", "detail")}
        self.show_private = False
        self.private_toggle_top = None
        self.opinion_bars = []
        self.opinion_sort = "Name"
        self.opinion_view = "Table"
        self.selected_opinion = None
        self.subtab = "Overview"
        self.expanded = set()
        self.show_departed = False
        self.hits = []
        self.dividers = []
        self.grid = None
        self.dialogue_portraits = []
        self.dialogue_backgrounds = []
        self.dead_backgrounds = []

    def open_record(self, record):
        self.tab = "Conversations"
        self.selected_conversation = record
        self.detail_scroll = 0
        self.scroll_focus = "detail"

    def open_log(self, log):
        self.tab = "Model logs"
        self.selected_log = log
        self.detail_scroll = 0
        self.follow_log = True
        self.scroll_focus = "detail"

    def _list_rows(self, app):
        rows = []
        top = 0
        if self.tab == "Creations":
            entries = [(work, f"{work.kind.title()}: {work.title}\n{app.simulation.actor_by_id(work.creator_id).name}\n{work.day}")
                       for work in self.gallery.entries(app.simulation)]
        elif self.tab == "Model logs":
            entries = [(log, f"{log.metadata['participants']}\n{log.status_label} / "
                       f"{('Thread ' + str(log.metadata['slot'])) if log.metadata['slot'] is not None else 'Waiting for a thread'}\n"
                       f"{log.metadata['started'][:19]} UTC")
                       for log in app.requests.debug_logs.history()]
        elif self.tab == "Conversations":
            entries = [
                (record, f"{index}. {' & '.join(record.participants)}")
                for index, record in reversed(list(enumerate(app.conversation_history, start=1)))
            ]
        else:
            sim = app.simulation
            entries = [(npc, npc.name) for npc in (*sim.npcs, sim.snake)]
            departed = [*sim.archived, *([] if sim.player.alive else [sim.player])]
            if departed:
                opened = self.show_departed or self.selected_npc in departed
                entries.append(("departed", f"{'-' if opened else '+'} Dead ({len(departed)})"))
                if opened:
                    entries += [(npc, PLAYER_LABEL if npc is sim.player else npc.name) for npc in departed]
        for entry, label in entries:
            lines = ([line for paragraph in label.splitlines()
                      for line in wrap_text(paragraph, app.font, self.list_rect.width - 30)]
                     if self.tab == "Creations" else wrap_text(label, app.font, self.list_rect.width - 30))
            height = max(48, len(lines) * app.font.get_linesize() + 20)
            rows.append((entry, top, height, lines))
            top += height + 8
        if self.tab == "Creations":
            current_ids = {entry.id for entry, *_ in rows}
            if self._creation_order and self.list_scroll > 0:
                self.list_scroll += sum(height + 8 for entry, _, height, _ in rows if entry.id not in self._creation_order)
            self._creation_order = current_ids
        return rows, max(0, top - 8)

    def _link(self, top, bottom, action):
        """Make a span of the detail pane clickable."""
        self.hits.append((top, bottom, action))

    def _open_conversation(self, app, record_id):
        record = next((entry for entry in app.conversation_history if entry.id == record_id), None)
        if record is not None:
            self.open_record(record)

    def _toggle(self, app, add, height, key, label):
        top = height()
        add(label, app.small_font, ACCENT)
        self._link(top, height(), lambda: self.expanded.symmetric_difference_update({key}))

    def _scroll_limits(self, app):
        _, list_height = self._list_rows(app)
        detail_height = (self.gallery.layout(app)[1] if self.tab == "Creations" else
                         sum(font.get_linesize() for _, font, _, _ in self.details.lines(app)))
        limits = {
            "list": max(0, list_height - self.list_rect.height),
            "detail": max(0, detail_height - self.detail_rect.height),
        }
        if self.tab == "Model logs" and self.follow_log:
            self.detail_scroll = limits["detail"]
        return limits

    def _scroll(self, pane, value, limits):
        setattr(self, f"{pane}_scroll", max(0, min(round(value), limits[pane])))

    @staticmethod
    def _scrollbar(rect, scroll, limit):
        return scrollbar_rects(rect, scroll, limit)

    def handle_event(self, app, event):
        self._position_detail()
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION,
                          pygame.WINDOWFOCUSLOST):
            limits = self._scroll_limits(app)
            for pane, rect in (("list", self.list_rect), ("detail", self.detail_rect)):
                value = self.scrollbars[pane].handle_event(event, rect, getattr(self, f"{pane}_scroll"), limits[pane])
                if value is not None:
                    self.scroll_focus = pane
                    if pane == "detail" and self.tab == "Model logs":
                        self.follow_log = False
                    self._scroll(pane, value, limits)
                    return
        if self.tab == "Statistics" and isinstance(self.selected_npc, NPC) and self.inventory_button.clicked(event):
            app.game.inventory.open(self.selected_npc)
            return
        if self.back_button.clicked(event) or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            self.gallery.stop()
            app.screen = "game"
            return
        for name, button in self.tabs.items():
            if button.clicked(event):
                if self.tab != name and (self.tab not in ("Statistics", "Opinions") or name not in ("Statistics", "Opinions")):
                    self.list_scroll = 0
                if self.tab != name:
                    self.gallery.stop()
                self.tab = name
                if name == "Conversations" and self.selected_conversation is None and app.conversation_history:
                    self.selected_conversation = app.conversation_history[-1]
                if name == "Model logs" and self.selected_log is None:
                    logs = app.requests.debug_logs.history()
                    if logs:
                        self.open_log(logs[0])
                self.detail_scroll = 0
                self.scroll_focus = "detail"
                return

        if self.tab == "Creations":
            action = self.gallery.handle_event(app, event, self.detail_scroll)
            if action:
                if action == "filter":
                    self.list_scroll = self.detail_scroll = 0
                    self._creation_order = set()
                    selected = self.gallery.selected(app.simulation)
                    if selected and selected not in self.gallery.entries(app.simulation):
                        self.gallery.selected_id = None
                return

        if self.tab == "Model logs" and self.follow_toggle.clicked(event):
            self.follow_log = not self.follow_log
            return

        if self.tab == "Statistics" and isinstance(self.selected_npc, NPC):
            for name, button in self.subtabs.items():
                if button.clicked(event):
                    self.subtab = name
                    self.detail_scroll = 0
                    return

        if self.tab == "Opinions" and self.selected_npc is not None and self.view_toggle.clicked(event):
            self.opinion_view = "Grid" if self.opinion_view == "Table" else "Table"
            self.detail_scroll = 0
            return

        if (self.tab == "Opinions" and self.opinion_view == "Table" and self.selected_npc is not None
                and self.sort_toggle.clicked(event)):
            self.opinion_sort = "Score" if self.opinion_sort == "Name" else "Name"
            return

        if self.tab == "Conversations" and self.condense_toggle.clicked(event):
            app.condensed_dialogue = not getattr(app, "condensed_dialogue", False)
            self._scroll("detail", self.detail_scroll, self._scroll_limits(app))
            return

        if self.tab == "Conversations" and self.original_toggle.clicked(event):
            app.original_dialogue = not app.original_dialogue
            self._scroll("detail", self.detail_scroll, self._scroll_limits(app))
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
                    if pane == "detail" and self.tab == "Model logs":
                        self.follow_log = False
                    self._scroll(pane, scroll - event.y * 48, limits)
                    return
                if event.button != 1:
                    return
                if pane == "detail":
                    content_y = pos[1] - rect.top + scroll
                    if (self.private_toggle_top is not None
                            and self.private_toggle_top <= content_y < self.private_toggle_top + app.small_font.get_linesize()):
                        self.show_private = not self.show_private
                        return
                    if self.grid and self.tab == "Opinions":
                        grid, x = self.grid, pos[0] - rect.x - GRID_LABEL_WIDTH
                        row, column = (content_y - grid["top"] - grid["header"]) // grid["cell"], x // grid["cell"]
                        if 0 <= row < len(grid["people"]) and (0 <= column < len(grid["people"]) or x < 0):
                            self.selected_npc = grid["people"][row]
                            self.selected_opinion = self.details.person_label(app, grid["people"][column]) if 0 <= column < len(grid["people"]) and column != row else None
                            self.opinion_view = "Table"
                            self.detail_scroll = 0
                            return
                    for top, bottom, action in self.hits:
                        if top <= content_y < bottom:
                            action()
                            return
                if pane == "list":
                    rows, _ = self._list_rows(app)
                    for entry, top, height, _ in rows:
                        if top <= pos[1] - rect.top + scroll < top + height:
                            if self.tab == "Creations":
                                self.gallery.select(entry)
                            elif self.tab == "Conversations":
                                self.selected_conversation = entry
                            elif self.tab == "Model logs":
                                self.open_log(entry)
                            elif isinstance(entry, str):
                                self.show_departed = not self.show_departed
                                return
                            else:
                                self.selected_npc = entry
                                self.show_private = False
                                self.selected_opinion = None
                                self.expanded.clear()
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
                if pane == "detail" and self.tab == "Model logs":
                    self.follow_log = event.key == pygame.K_END
                self._scroll(pane, scroll + steps[event.key], limits)

    def draw(self, app):
        self._position_detail()
        surface = app.window
        surface.fill(GRASS)
        self.back_button.draw(surface, app.small_font)
        surface.blit(app.heading_font.render("World Stats", True, INK), (380, 61))
        surface.blit(app.small_font.render("People, items, and conversations.", True, MUTED), (184, 108))
        pygame.draw.rect(surface, PAPER, (184, 134, 296, 506))
        pygame.draw.rect(surface, PAPER, (496, 134, 600, 506))
        list_title = ("Model requests" if self.tab == "Model logs" else
                      f"Conversations ({len(app.conversation_history)})" if self.tab == "Conversations" else
                      f"Living {app.simulation.population} / Dead {app.simulation.departed_count}")
        if self.tab != "Creations":
            surface.blit(app.font.render(list_title, True, INK), (200, 150))
        for name, button in self.tabs.items():
            button.draw(surface, app.small_font)
            if self.tab == name:
                pygame.draw.line(surface, PAPER, (button.rect.left + 16, 176), (button.rect.right - 16, 176), 2)
        if self.tab == "Conversations":
            self.condense_toggle.label = "Condense: " + ("On" if getattr(app, "condensed_dialogue", False) else "Off")
            self.condense_toggle.draw(surface, app.small_font)
            self.original_toggle.label = "Original dialogue: " + ("On" if app.original_dialogue else "Off")
            self.original_toggle.draw(surface, app.small_font)
        if self.tab == "Model logs":
            self.follow_toggle.label = "Follow" if self.follow_log else "Paused"
            self.follow_toggle.draw(surface, app.small_font)

        limits = self._scroll_limits(app)
        for pane in limits:
            self._scroll(pane, getattr(self, f"{pane}_scroll"), limits)
        previous_clip = surface.get_clip()
        surface.set_clip(self.list_rect.clip(previous_clip))
        rows, _ = self._list_rows(app)
        gift_ids = set()
        generating_ids = set()
        queued_ids = set()
        if self.tab == "Conversations":
            gift_ids = {fact.conversation_id for fact in app.simulation.facts if fact.kind == "apple_given"}
            queued_ids = {
                request_id for (session_id, request_id), log in app.requests.logs.items()
                if session_id == app.session_id and log.status in ("queued", "waiting_for_ollama")
            }
            for request in getattr(app, "player_requests", {}).values():
                if player_request_pending(request) and request.id in queued_ids:
                    queued_ids.add(request.record.id)
            active_requests = {
                worker.request_id for worker in app.requests.statuses()
                if worker.session_id == app.session_id and worker.log is not None and worker.log.status == "running"
            }
            generating_ids = {
                conversation.record.id for request_id, conversation in app.conversation_requests.items()
                if request_id in active_requests
            }
            generating_ids.update(request.record.id for request in getattr(app, "player_requests", {}).values()
                                  if player_request_pending(request) and request.id in active_requests)
        if not rows and self.tab == "Conversations":
            surface.blit(app.small_font.render("No conversations yet.", True, MUTED), (self.list_rect.x, self.list_rect.y + 12))
        if not rows and self.tab == "Model logs":
            surface.blit(app.small_font.render("No model requests yet.", True, MUTED), (self.list_rect.x, self.list_rect.y + 12))
        for entry, top, height, lines in rows:
            row = pygame.Rect(self.list_rect.x, self.list_rect.y + top - self.list_scroll, self.list_rect.width - 14, height)
            if not row.colliderect(self.list_rect):
                continue
            selected_entry = (self.gallery.selected(app.simulation) if self.tab == "Creations" else
                              self.selected_log if self.tab == "Model logs" else
                              self.selected_conversation if self.tab == "Conversations" else self.selected_npc)
            selected = entry is selected_entry
            if self.tab == "Model logs":
                color = MODEL_LOG_COLORS[entry.status]
            elif self.tab == "Conversations":
                if entry.id in generating_ids:
                    state = "generating"
                elif entry.id in queued_ids:
                    state = "queued"
                elif entry.status == CONVERSATION_FAILED or entry.dialogue_errors:
                    state = "failed"
                elif has_failed_voice_rewrite(entry):
                    state = "rewrite_failed"
                elif entry.id in gift_ids and not entry.turns and not entry.failed_turns:
                    state = "gift"
                else:
                    state = "normal"
                color = CONVERSATION_COLORS[state][selected]
            elif self.tab in ("Statistics", "Opinions") and (entry == "departed" or not entry.alive):
                color = STATS_DEAD_SELECTED_COLOR if selected else STATS_DEAD_COLOR
            else:
                color = ACCENT if selected else GRASS
            pygame.draw.rect(surface, color, row)
            if selected and self.tab == "Model logs":
                pygame.draw.rect(surface, INK, row, 2)
            for index, line in enumerate(lines):
                surface.blit(app.font.render(line, True, PAPER if selected and self.tab != "Model logs" else INK),
                             (row.x + 8, row.y + 10 + index * app.font.get_linesize()))
        surface.set_clip(self.detail_rect.clip(previous_clip))
        lines = self.details.lines(app)
        for top, bottom in self.dialogue_backgrounds:
            pygame.draw.rect(surface, ORIGINAL_DIALOGUE_PAPER,
                (self.detail_rect.x - 4, self.detail_rect.top + top - self.detail_scroll,
                 self.detail_rect.width - 8, bottom - top))
        for top, bottom, selected in self.dead_backgrounds:
            row = pygame.Rect(self.detail_rect.x - 4, self.detail_rect.top + top - self.detail_scroll,
                              self.detail_rect.width - 8, bottom - top)
            pygame.draw.rect(surface, STATS_DEAD_COLOR, row)
            if selected:
                pygame.draw.rect(surface, ACCENT, row, 2)
        y = self.detail_rect.top - self.detail_scroll
        for line, font, color, indent in lines:
            if self.detail_rect.top - font.get_linesize() < y < self.detail_rect.bottom:
                surface.blit(font.render(line, True, color), (self.detail_rect.x + indent, y))
            y += font.get_linesize()
        for top, actor, emotion, labels in self.dialogue_portraits:
            rect = pygame.Rect(self.detail_rect.x, self.detail_rect.top + top - self.detail_scroll, 88, 88)
            if rect.bottom + 6 + len(labels) * app.small_font.get_linesize() < self.detail_rect.top or rect.top >= self.detail_rect.bottom:
                continue
            draw_actor_portrait(surface, actor, rect, app.character.renderer, snake=app.simulation.snake,
                padding=STATS_PORTRAIT_PADDING, background=GRASS, emotion=emotion)
            for index, label in enumerate(labels):
                surface.blit(app.small_font.render(label, True, MUTED),
                             (rect.x, rect.bottom + 6 + index * app.small_font.get_linesize()))
        for rule_top in self.dividers:
            y = self.detail_rect.top + rule_top - self.detail_scroll
            if self.detail_rect.top <= y < self.detail_rect.bottom:
                pygame.draw.line(surface, GRASS, (self.detail_rect.x, y), (self.detail_rect.right - 24, y), 2)
        if self.grid:
            self.details.draw_grid(app, surface)
        for top, label, value in self.opinion_bars:
            y = self.detail_rect.top + top - self.detail_scroll
            line = app.small_font.get_linesize()
            if y + line < self.detail_rect.top or y > self.detail_rect.bottom:
                continue
            track = pygame.Rect(self.detail_rect.x + OPINION_BAR_LEFT, y + 3, OPINION_BAR_WIDTH, line - 6)
            surface.blit(app.small_font.render(label, True, MUTED), (self.detail_rect.x + 8, y))
            pygame.draw.rect(surface, GRASS, track)
            if value is not None:
                length = round(abs(value) / 100 * track.width / 2)
                fill = pygame.Rect(track.centerx if value >= 0 else track.centerx - length, track.y, length, track.height)
                pygame.draw.rect(surface, OPINION_POSITIVE_COLOR if value >= 0 else OPINION_NEGATIVE_COLOR, fill)
            pygame.draw.line(surface, INK, (track.centerx, track.y - 1), (track.centerx, track.bottom))
            surface.blit(app.small_font.render("n/a" if value is None else f"{value:+d}", True, INK), (track.right + 10, y))
        surface.set_clip(previous_clip)
        if self.tab == "Opinions" and self.selected_npc is not None:
            self.view_toggle.label = f"View: {self.opinion_view}"
            self.view_toggle.draw(surface, app.small_font)
            if self.opinion_view == "Table":
                self.sort_toggle.label = f"Sort: {self.opinion_sort}"
                self.sort_toggle.draw(surface, app.small_font)
        if self.tab == "Statistics" and isinstance(self.selected_npc, NPC) and self.selected_npc is not app.simulation.snake:
            for name, button in self.subtabs.items():
                button.draw(surface, app.small_font)
                if self.subtab == name:
                    pygame.draw.line(surface, PAPER, (button.rect.left + 10, button.rect.bottom - 5),
                                     (button.rect.right - 10, button.rect.bottom - 5), 2)
        if self.tab == "Statistics" and isinstance(self.selected_npc, NPC):
            self.inventory_button.draw(surface, app.small_font)
        if self.tab == "Creations":
            self.gallery.draw(app, self.detail_scroll)
        for pane, rect in (("list", self.list_rect), ("detail", self.detail_rect)):
            if limits[pane]:
                track, thumb = self._scrollbar(rect, getattr(self, f"{pane}_scroll"), limits[pane])
                pygame.draw.rect(surface, GRASS, track)
                pygame.draw.rect(surface, MUTED, thumb)
        surface.blit(app.small_font.render("Drag the scrollbar, scroll, or use Up/Down and Page Up/Down. Escape returns to the game.", True, MUTED), (184, 655))

    def _position_detail(self):
        top = STATS_INVENTORY_DETAIL_TOP if self.tab == "Statistics" and isinstance(self.selected_npc, NPC) else 200
        self.detail_rect.top = top
        self.detail_rect.height = 624 - top
