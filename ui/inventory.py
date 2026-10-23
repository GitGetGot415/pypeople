"""Player possessions and read-only resident inventories."""

import pygame
from constants import (INVENTORY_RECT, INVENTORY_ROW_HEIGHT, INVENTORY_PREVIEW_SCALE, MODAL_OVERLAY_COLOR,
                      PAPER, INK, MUTED, ACCENT, GRASS, INVENTORY_LIST_RECT, INVENTORY_RECIPIENT_RECT,
                      INVENTORY_CLOSE_RECT, INVENTORY_FILTER_RECTS, INVENTORY_MINUS_RECT,
                      INVENTORY_PLUS_RECT, INVENTORY_GIVE_RECT, INVENTORY_CONFIRM_RECT, INVENTORY_PLAY_RECT,
                      INVENTORY_STOP_RECT, INVENTORY_TITLE_POSITION, INVENTORY_DETAIL_POSITION,
                      INVENTORY_DETAIL_WIDTH, INVENTORY_STATUS_RECT, INVENTORY_TITLE_WIDTH,
                      INVENTORY_APPLES_POSITION)
from .button import Button
from .creations import CreationsGallery
from .text import draw_wrapped_text, scrollbar_rects, ScrollbarDrag


class InventoryPanel:
    def __init__(self):
        self.visible = False
        self.media = CreationsGallery()
        self.rect = pygame.Rect(INVENTORY_RECT)
        self.close_button = Button(INVENTORY_CLOSE_RECT, "Close")
        self.list_rect = pygame.Rect(INVENTORY_LIST_RECT)
        self.recipient_rect = pygame.Rect(INVENTORY_RECIPIENT_RECT)
        self.scrollbars = {pane: ScrollbarDrag() for pane in ("scroll", "recipient_scroll")}
        self.filters = {name: Button(rect, name) for name, rect in zip(("All", "Food", "Art", "Music"), INVENTORY_FILTER_RECTS)}
        self.minus = Button(INVENTORY_MINUS_RECT, "-")
        self.plus = Button(INVENTORY_PLUS_RECT, "+")
        self.give = Button(INVENTORY_GIVE_RECT, "Give")
        self.confirm = Button(INVENTORY_CONFIRM_RECT, "Confirm gift offer")
        self.play = Button(INVENTORY_PLAY_RECT, "Play")
        self.stop = Button(INVENTORY_STOP_RECT, "Stop")

    def open(self, owner):
        self.media.close()
        self.owner = owner
        self.visible = True
        self.filter, self.selected_id = "All", "fruit"
        self.scroll = self.recipient_scroll = 0
        self.quantity, self.recipient_id = 1, None
        self.choosing = False
        self.status = ""

    def close(self):
        self.visible = False
        for scrollbar in self.scrollbars.values():
            scrollbar.cancel()
        self.media.close()

    def entries(self, world):
        works = [(w.id, f'[{w.id[:8]}] {w.kind.title()}: {w.title}') for w in reversed(world.owned_works(self.owner.id))
                 if self.filter == "All" or w.kind == self.filter.lower()]
        return ([("fruit", self.apple_label())]
                if self.filter in ("All", "Food") else []) + works

    def apple_label(self):
        return f"Apples: {self.owner.inventory['fruit']}"

    def recipients(self, app):
        from locations import available as available_outdoors
        from player import active_player
        if not active_player(app.simulation):
            return []
        return [n for n in app.npcs if n.location == app.simulation.player.location and available_outdoors(n)
                and not n.busy and not n.interaction_blocked and not n.selected and not n.unread_player_reply]

    def handle_event(self, app, event):
        rows = self.entries(app.simulation)
        panes = [("scroll", self.list_rect, len(rows))]
        if self.choosing:
            panes.append(("recipient_scroll", self.recipient_rect, len(self.recipients(app))))
        for pane, rect, count in panes:
            limit = max(0, count * INVENTORY_ROW_HEIGHT - rect.height)
            value = self.scrollbars[pane].handle_event(event, rect, getattr(self, pane), limit)
            if value is not None:
                setattr(self, pane, value)
                return
        if self.close_button.clicked(event):
            self.close()
            return
        pos = getattr(event, "pos", pygame.mouse.get_pos())
        if event.type == pygame.MOUSEWHEEL:
            if self.list_rect.collidepoint(pos):
                self.scroll = max(0, min(max(0, len(rows) * INVENTORY_ROW_HEIGHT - self.list_rect.height),
                                         self.scroll - event.y * INVENTORY_ROW_HEIGHT))
            elif self.choosing and self.recipient_rect.collidepoint(pos):
                self.recipient_scroll = max(0, min(max(0, len(self.recipients(app)) * INVENTORY_ROW_HEIGHT - self.recipient_rect.height),
                                                  self.recipient_scroll - event.y * INVENTORY_ROW_HEIGHT))
            return
        for name, button in self.filters.items():
            if button.clicked(event):
                self.filter, self.scroll = name, 0
                return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.list_rect.collidepoint(pos):
                index = (pos[1] - self.list_rect.y + self.scroll) // INVENTORY_ROW_HEIGHT
                if index < len(rows):
                    self.selected_id = rows[index][0]
                    self.media.stop()
                    self.media.selected_id = self.selected_id
                    self.choosing = False
                return
            if self.choosing and self.recipient_rect.collidepoint(pos):
                index = (pos[1] - self.recipient_rect.y + self.recipient_scroll) // INVENTORY_ROW_HEIGHT
                people = self.recipients(app)
                if index < len(people):
                    self.recipient_id = people[index].id
                return
        work = next((w for w in app.simulation.creations if w.id == self.selected_id and w.owner_id == self.owner.id), None)
        if work and work.kind == "music":
            if self.play.clicked(event):
                self.media.select(work)
                self.media.play(work)
            elif self.stop.clicked(event):
                self.media.stop()
        if self.owner is not app.simulation.player or app.simulation.paused or app.simulation.mode != "player" or not app.simulation.player.alive:
            return
        if self.minus.clicked(event):
            self.quantity = max(1, self.quantity - 1)
        elif self.plus.clicked(event):
            self.quantity = min(max(1, self.owner.inventory["fruit"]), self.quantity + 1)
        elif self.give.clicked(event):
            self.choosing, self.recipient_id = True, None
        elif self.choosing and self.confirm.clicked(event):
            recipient = next((n for n in self.recipients(app) if n.id == self.recipient_id), None)
            if recipient is None:
                self.status = "Select an available resident; the earlier recipient may now be occupied."
                return
            try:
                app._sync_conversation_view()
                app.conversations.offer_player_item(recipient, self.selected_id, self.quantity if self.selected_id == "fruit" else 1)
                app._present_conversation_events()
                self.close()
            except ValueError as exc:
                self.status = str(exc)

    def draw(self, app):
        self.media.update()
        surface = app.window
        shade = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        shade.fill(MODAL_OVERLAY_COLOR)
        surface.blit(shade, (0, 0))
        pygame.draw.rect(surface, PAPER, self.rect)
        title = "Player inventory" if self.owner is app.simulation.player else f"{self.owner.name}'s inventory (read only)"
        if getattr(self.owner, "departure", None):
            title += " / Departed"
        if app.heading_font.size(title)[0] > INVENTORY_TITLE_WIDTH:
            name = self.owner.name
            suffix = "... inventory (read only)" + (" / Departed" if getattr(self.owner, "departure", None) else "")
            while name and app.heading_font.size(name + suffix)[0] > INVENTORY_TITLE_WIDTH:
                name = name[:-1]
            title = name + suffix
        surface.blit(app.heading_font.render(title, True, INK), INVENTORY_TITLE_POSITION)
        surface.blit(app.font.render(self.apple_label(), True, INK),
                     INVENTORY_APPLES_POSITION)
        self.close_button.draw(surface, app.small_font)
        for name, button in self.filters.items():
            button.draw(surface, app.small_font)
            if name == self.filter:
                pygame.draw.line(surface, INK, button.rect.bottomleft, button.rect.bottomright, 2)
        rows = self.entries(app.simulation)
        self.scroll = min(self.scroll, max(0, len(rows) * INVENTORY_ROW_HEIGHT - self.list_rect.height))
        clip = surface.get_clip()
        surface.set_clip(self.list_rect)
        for index, (item, title) in enumerate(rows):
            rect = pygame.Rect(self.list_rect.x, self.list_rect.y + index * INVENTORY_ROW_HEIGHT - self.scroll,
                               self.list_rect.width, INVENTORY_ROW_HEIGHT - 2)
            pygame.draw.rect(surface, ACCENT if item == self.selected_id else PAPER, rect)
            surface.blit(app.small_font.render(title, True, PAPER if item == self.selected_id else INK), (rect.x + 6, rect.y + 6))
        if not rows:
            surface.blit(app.font.render("No possessions in this filter.", True, MUTED), self.list_rect.topleft)
        surface.set_clip(clip)
        self.draw_scrollbar(surface, self.list_rect, self.scroll, len(rows))
        x, y = INVENTORY_DETAIL_POSITION
        work = next((w for w in app.simulation.creations if w.id == self.selected_id and w.owner_id == self.owner.id), None)
        if self.selected_id == "fruit":
            text = self.apple_label() + ". Creative works use no apple space."
            text += " Apple storage is unlimited."
            if self.owner.inventory["fruit"] == 0:
                text += " No apples to give."
            draw_wrapped_text(surface, text, app.font, INK, (x, y, INVENTORY_DETAIL_WIDTH, 110))
        elif work:
            creator = app.simulation.participant_by_id(work.creator_id)
            credit = "Composer" if work.kind == "music" else "Creator"
            draw_wrapped_text(surface, f'{work.title} [{work.id[:8]}]\n{credit}: {creator.name} / Owner: {self.owner.name}',
                              app.font, INK, (x, y, INVENTORY_DETAIL_WIDTH, 64))
            transfers = [t for r in app.conversation_history for t in r.transfers if t.item == work.id]
            if transfers:
                last = max(transfers, key=lambda t: t.sequence)
                giver = app.simulation.participant_by_id(last.giver_id)
                draw_wrapped_text(surface, f'Latest giver: {giver.name}, day {last.day} / {last.phase}',
                                  app.small_font, MUTED, (x + 220, y + 78, 350, 90))
            if work.kind == "art":
                surface.blit(self.media.painting_preview(work, INVENTORY_PREVIEW_SCALE), (x, y + 70))
            else:
                self.play.draw(surface, app.small_font)
                self.stop.draw(surface, app.small_font)
                draw_wrapped_text(surface, self.media.status or "Press Play to hear this composition.", app.small_font, MUTED,
                                  (x, y + 90, INVENTORY_DETAIL_WIDTH, 70))
        else:
            draw_wrapped_text(surface, "This work is no longer owned here. Select a current possession.", app.font, MUTED,
                              (x, y, INVENTORY_DETAIL_WIDTH, 110))
        if self.owner is app.simulation.player:
            enabled = not app.simulation.paused and not app.game.dialogue_open
            can_give = enabled and (self.owner.inventory["fruit"] > 0 if self.selected_id == "fruit" else work is not None)
            self.minus.draw(surface, app.small_font, enabled=enabled and self.selected_id == "fruit")
            self.plus.draw(surface, app.small_font, enabled=enabled and self.selected_id == "fruit")
            surface.blit(app.font.render(str(self.quantity if self.selected_id == "fruit" else 1), True, INK), (self.minus.rect.right + 24, self.minus.rect.y + 6))
            self.give.draw(surface, app.small_font, enabled=can_give)
            if self.choosing:
                people = self.recipients(app)
                self.recipient_scroll = min(self.recipient_scroll, max(0, len(people) * INVENTORY_ROW_HEIGHT - self.recipient_rect.height))
                surface.set_clip(self.recipient_rect)
                for i, npc in enumerate(people):
                    rect = pygame.Rect(self.recipient_rect.x, self.recipient_rect.y + i * INVENTORY_ROW_HEIGHT - self.recipient_scroll,
                                       self.recipient_rect.width, INVENTORY_ROW_HEIGHT)
                    pygame.draw.rect(surface, ACCENT if npc.id == self.recipient_id else PAPER, rect)
                    surface.blit(app.small_font.render(f'{npc.name} / unlimited apple storage', True,
                                 PAPER if npc.id == self.recipient_id else INK), (rect.x + 4, rect.y + 6))
                if not people:
                    surface.blit(app.small_font.render("No available recipients here.", True, MUTED), self.recipient_rect.topleft)
                surface.set_clip(clip)
                self.draw_scrollbar(surface, self.recipient_rect, self.recipient_scroll, len(people))
                self.confirm.draw(surface, app.small_font, enabled=can_give and bool(self.recipient_id))
            note = ("Resume time to give." if app.simulation.paused else "Finish the open conversation to give." if app.game.dialogue_open else
                    f'Offer: {self.quantity if self.selected_id == "fruit" else 1} × {"apples" if self.selected_id == "fruit" else work.title + " [" + work.id + "]" if work else "unavailable work"}. The resident may accept or decline.')
            draw_wrapped_text(surface, self.status or note, app.small_font, MUTED, INVENTORY_STATUS_RECT)
        elif work:
            draw_wrapped_text(surface, f"Work ID: {work.id}", app.small_font, MUTED, INVENTORY_STATUS_RECT)

    @staticmethod
    def draw_scrollbar(surface, rect, scroll, count):
        limit = max(0, count * INVENTORY_ROW_HEIGHT - rect.height)
        if limit:
            track, thumb = scrollbar_rects(rect, scroll, limit)
            pygame.draw.rect(surface, GRASS, track)
            pygame.draw.rect(surface, MUTED, thumb)
