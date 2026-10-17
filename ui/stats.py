import pygame

from character_data import DEFAULT_COLORS, FIELDS
from constants import ACCENT, EMOTION_THRESHOLD, GRASS, INK, MUTED, PAPER
from social import assertion_text, claim_by_id, fact_by_id, phase_label, selected_beliefs

from .button import Button
from .text import wrap_text
from .world import snake_artwork


class StatsScreen:
    """Browse live NPC profiles and the current game's conversation history."""

    def __init__(self):
        self.back_button = Button((184, 56, 170, 38), "Back to game")
        self.tabs = {
            name: Button((512 + index * 176, 146, 160, 36), name)
            for index, name in enumerate(("Statistics", "Opinions", "Conversations"))
        }
        self.list_rect = pygame.Rect(200, 184, 264, 440)
        self.detail_rect = pygame.Rect(512, 200, 568, 424)
        self.open()

    def open(self):
        self.selected_npc = None
        self.selected_conversation = None
        self.tab = "Statistics"
        self.list_scroll = 0
        self.detail_scroll = 0
        self.scroll_focus = "list"
        self.show_private = False
        self.private_toggle_top = None
        self.promise_links = []
        self.dialogue_portraits = []

    def open_record(self, record):
        self.tab = "Conversations"
        self.selected_conversation = record
        self.detail_scroll = 0
        self.scroll_focus = "detail"

    def _list_rows(self, app):
        rows = []
        top = 0
        if self.tab == "Conversations":
            entries = [
                (record, f"{index}. {' & '.join(record.participants)}")
                for index, record in reversed(list(enumerate(app.conversation_history, start=1)))
            ]
        else:
            entries = [(npc, npc.name + (" (departed)" if getattr(npc, "departure", None) else "")) for npc in app.simulation.actors]
        for entry, label in entries:
            lines = wrap_text(label, app.font, self.list_rect.width - 30)
            height = max(48, len(lines) * app.font.get_linesize() + 20)
            rows.append((entry, top, height, lines))
            top += height + 8
        return rows, max(0, top - 8)

    def _detail_lines(self, app):
        lines = []
        self.private_toggle_top = None
        self.promise_links = []
        self.dialogue_portraits = []

        def add(text, font=app.font, color=INK, indent=0):
            for paragraph in str(text).splitlines() or [""]:
                for line in wrap_text(paragraph, font, self.detail_rect.width - 20 - indent) or [""]:
                    lines.append((line, font, color, indent))

        def gap():
            lines.append(("", app.small_font, MUTED, 0))

        def height():
            return sum(font.get_linesize() for _, font, _, _ in lines)

        def message(speaker, text, emotion=None):
            npc = next((npc for npc in app.simulation.actors if npc.name == speaker), None)
            top = height()
            indent = 112 if npc is not None else 0
            add(speaker, color=ACCENT, indent=indent)
            add(text, indent=indent)
            if npc is not None:
                visible = emotion is not None and emotion["intensity"] >= EMOTION_THRESHOLD
                expression = emotion["name"] if visible else None
                label = (expression or "neutral").title() if emotion is not None else "Not recorded"
                labels = wrap_text(label, app.small_font, 100)
                self.dialogue_portraits.append((top, npc, expression, labels))
                portrait_height = 94 + len(labels) * app.small_font.get_linesize()
                while height() - top < portrait_height:
                    gap()
            gap()

        if self.tab == "Conversations":
            record = self.selected_conversation
            if record is None:
                add(
                    "Select a conversation to view the participants, outcome, and dialogue."
                    if app.conversation_history else "No conversations have happened in this game yet.",
                    color=MUTED,
                )
                return lines
            add(" & ".join(record.participants), app.heading_font)
            add(f"Status: {record.status}", app.small_font, MUTED)
            add(f"{record.origin.title()} / {record.purpose.replace('_', ' ')}", app.small_font, MUTED)
            gap()
            add(f"Topic / request: {record.topic}")
            gap()
            add("Outcome", app.heading_font)
            if record.error:
                add(record.error, color=MUTED)
            for outcome in record.outcomes:
                add(outcome)
                gap()
            if not record.outcomes:
                add("No outcome recorded yet.", color=MUTED)
                gap()
            add("Dialogue", app.heading_font)
            gap()
            if not record.turns:
                add("No dialogue generated yet.", color=MUTED)
            for index, (speaker, text) in enumerate(record.turns):
                parts = record.dialogue_parts.get(index)
                if parts:
                    for part in parts:
                        message(speaker, part["text"], part["emotion"])
                else:
                    message(speaker, text)
            return lines

        npc = self.selected_npc
        if npc is None:
            add("Select an NPC to view their statistics and opinions.", color=MUTED)
            return lines
        add(npc.name, app.heading_font)
        gap()
        if npc is app.simulation.snake:
            add(f"Activity: {npc.status}")
            add(f"Eaten today: {npc.eaten_today}; recovery: {npc.recovery_remaining:.1f}s")
            add(f"Tree stock: {app.tree.fruit}; unreserved: {app.simulation.available_fruit}")
            add("The python claims divine authority. Its claims about destinations and other zones are unverified.", app.small_font, MUTED)
            gap()
            add("Resident standing (the snake's standards)", app.heading_font)
            for actor_id, score in npc.standing.items():
                add(f"{app.simulation.actor_by_id(actor_id).name}: {score:+d}")
            gap()
            add("Upfront apples: offerings still owed", app.heading_font)
            for actor_id, advance in npc.advances.items():
                offering = "private information" if advance["offering"] == "secret" else "a public endorsement"
                add(f"{app.simulation.actor_by_id(actor_id).name} owes {offering}; follow-up in {advance['remaining']:.0f}s.")
            if not npc.advances:
                add("No outstanding offerings.", app.small_font, MUTED)
            gap()
            add("Known accounts (observer inspection)", app.heading_font)
            for account in npc.snapshot()["known_accounts"]:
                add(account["assertion"], app.small_font)
                add(f"Source: {account['immediate_source_name']}; original: {account['original_speaker_name']}", app.small_font, MUTED)
            return lines
        if self.tab == "Statistics":
            add("Description", app.heading_font)
            add(npc.character["Details"] or "Not specified")
            gap()
            add(f"Arrival: day {npc.arrival_day}")
            add(f"Activity: {npc.status}")
            if npc.departure:
                source = next(f for f in app.simulation.facts if f.source_id == npc.departure["event_id"] and f.kind == "eating")
                top = height()
                add("View departure encounter", app.small_font, ACCENT)
                self.promise_links.append((top, height(), source.conversation_id))
            add(f"Why: {npc.decision_reason}")
            add(f"Control: {npc.owner.replace('_', ' ')}", app.small_font, MUTED)
            if npc.request_pending:
                add("Player request pending; new decisions are deferred.", app.small_font, MUTED)
            gap()
            add(f"Satisfaction: {npc.social:.0f}/100")
            inventory_label = "Inventory at departure" if npc.departure else "Carrying"
            add(f"{inventory_label}: {npc.inventory.get('fruit', 0)} fruit / {npc.inventory_capacity} items")
            advance = app.simulation.snake.advances.get(npc.id)
            if advance:
                offering = "private information" if advance["offering"] == "secret" else "a public endorsement"
                add(f"Received an apple upfront; owes {offering} to the snake.", app.small_font, MUTED)
            add(f"Sociability: {npc.preferences['sociability']:.0%}; generosity: {npc.preferences['generosity']:.0%}", app.small_font, MUTED)
            gap()
            add("Personal objective", app.heading_font)
            goal = npc.goal
            if goal:
                add(goal.objective)
                add(f"{goal.status.title()} / due {phase_label(goal.deadline)}", app.small_font, MUTED)
                if goal.outcome:
                    add(goal.outcome, app.small_font)
            else:
                add("No feasible personal objective right now.", color=MUTED)
            self.private_toggle_top = height()
            add(f"{'Hide' if self.show_private else 'Show'} private details (observer only)", app.small_font, ACCENT)
            if self.show_private:
                add("Attitudes toward the snake", app.heading_font)
                for field in ("belief", "trust", "authority"):
                    add(f"{field.title()}: {npc.attitudes[field]}")
                    add(npc.attitudes[field + "_reason"], app.small_font, MUTED)
                gap()
                if goal:
                    add(f"Motive: {goal.motive}")
                    add(f"Approach: {goal.approach}")
                    add(f"Boundary: {goal.boundary}")
                add("Known accounts and sources", app.heading_font)
                if not npc.beliefs:
                    add("No firsthand accounts or rumors yet.", app.small_font, MUTED)
                for belief in selected_beliefs(npc):
                    claim = claim_by_id(app.simulation, belief.claim_id)
                    fact = fact_by_id(app.simulation, claim.fact_id)
                    source = app.simulation.actor_by_id(belief.source_id)
                    originator = app.simulation.actor_by_id(claim.originator_id)
                    add(assertion_text(app.simulation, fact, claim.affirmed), app.small_font)
                    add(f"{belief.stance.title()} / {'firsthand' if belief.direct else 'personal assertion' if belief.source_id == npc.id else 'hearsay'}", app.small_font, MUTED)
                    add(f"Immediate source: {source.name}; original account: {originator.name}", app.small_font, MUTED)
                    add(f"Observer: {'supported by the actual event or intention' if claim.affirmed else 'contradicts the actual event or intention'}.", app.small_font, MUTED)
                    if claim.deliberate:
                        add(f"Observer: the original assertion was deliberately fabricated by {originator.name}.", app.small_font, MUTED)
                    if belief.disclosure_id:
                        disclosure = next(d for d in app.simulation.disclosures if d.id == belief.disclosure_id)
                        top = height()
                        add("View source conversation", app.small_font, ACCENT)
                        bottom = height()
                        self.promise_links.append((top, bottom, disclosure.conversation_id))
                    gap()
            gap()
            add("Apple promises", app.heading_font)
            promises = [p for p in app.simulation.promises if npc.id in (p.promisor_id, p.recipient_id)]
            promises.sort(key=lambda p: (p.status == "pending", p.created_phase), reverse=True)
            if not promises:
                add("No promises yet.", color=MUTED)
            for promise in promises[:8]:
                giver = app.simulation.actor_by_id(promise.promisor_id)
                recipient = app.simulation.actor_by_id(promise.recipient_id)
                top = height()
                add(f"{giver.name} -> {recipient.name}: one apple / {promise.status} [view]", app.small_font, ACCENT)
                bottom = height()
                self.promise_links.append((top, bottom, promise.conversation_id))
                add(f"Due {phase_label(promise.deadline)}", app.small_font, MUTED)
                if promise.outcome:
                    add(promise.outcome, app.small_font)
            if npc.pending_social_opinions:
                add("An opinion could not be saved; the simulation will retry.", app.small_font, MUTED)
            gap()
            add("Recent memories", app.heading_font)
            if not npc.memories:
                add("No personal experiences remembered yet.", color=MUTED)
            for memory in reversed(npc.memories):
                add(f"Day {memory.day} / {memory.phase.title()}: {memory.summary}")
                add(f"Influence: {memory.weight:+g}", app.small_font, MUTED)
            gap()
            fields = tuple(field for field in FIELDS if field != "Details")
            for field in (*fields, "Hair", "Eyes", *DEFAULT_COLORS):
                value = npc.name if field == "Name" else npc.character.get(field, "")
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
        _, list_height = self._list_rows(app)
        detail_height = sum(font.get_linesize() for _, font, _, _ in self._detail_lines(app))
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
        if app.game.handle_clock_event(app, event):
            return
        if self.back_button.clicked(event) or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            app.screen = "game"
            return
        for name, button in self.tabs.items():
            if button.clicked(event):
                if (self.tab == "Conversations") != (name == "Conversations"):
                    self.list_scroll = 0
                self.tab = name
                if name == "Conversations" and self.selected_conversation is None and app.conversation_history:
                    self.selected_conversation = app.conversation_history[-1]
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
                if pane == "detail":
                    content_y = pos[1] - rect.top + scroll
                    if (self.private_toggle_top is not None
                            and self.private_toggle_top <= content_y < self.private_toggle_top + app.small_font.get_linesize()):
                        self.show_private = not self.show_private
                        return
                    for top, bottom, record_id in self.promise_links:
                        if top <= content_y < bottom:
                            record = next(entry for entry in app.conversation_history if entry.id == record_id)
                            self.open_record(record)
                            return
                track, thumb = self._scrollbar(rect, scroll, limits[pane])
                if limits[pane] and track.collidepoint(pos):
                    offset = pos[1] - track.top - thumb.height / 2
                    self._scroll(pane, offset * limits[pane] / (track.height - thumb.height), limits)
                    return
                if pane == "list":
                    rows, _ = self._list_rows(app)
                    for entry, top, height, _ in rows:
                        if top <= pos[1] - rect.top + scroll < top + height:
                            if self.tab == "Conversations":
                                self.selected_conversation = entry
                            else:
                                self.selected_npc = entry
                                self.show_private = False
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
        surface.blit(app.heading_font.render("World Stats", True, INK), (380, 61))
        surface.blit(app.small_font.render("People, items, and conversations.", True, MUTED), (184, 108))
        app.game.draw_clock(app)
        pygame.draw.rect(surface, PAPER, (184, 134, 296, 506))
        pygame.draw.rect(surface, PAPER, (496, 134, 600, 506))
        list_title = f"Conversations ({len(app.conversation_history)})" if self.tab == "Conversations" else f"Living {len(app.npcs)} / Departed {len(app.simulation.archived)}"
        surface.blit(app.font.render(list_title, True, INK), (200, 150))
        for name, button in self.tabs.items():
            button.draw(surface, app.small_font)
            if self.tab == name:
                pygame.draw.line(surface, PAPER, (button.rect.left + 16, 176), (button.rect.right - 16, 176), 2)

        limits = self._scroll_limits(app)
        for pane in limits:
            self._scroll(pane, getattr(self, f"{pane}_scroll"), limits)
        previous_clip = surface.get_clip()
        surface.set_clip(self.list_rect.clip(previous_clip))
        rows, _ = self._list_rows(app)
        if not rows and self.tab == "Conversations":
            surface.blit(app.small_font.render("No conversations yet.", True, MUTED), (self.list_rect.x, self.list_rect.y + 12))
        for entry, top, height, lines in rows:
            row = pygame.Rect(self.list_rect.x, self.list_rect.y + top - self.list_scroll, self.list_rect.width - 14, height)
            if not row.colliderect(self.list_rect):
                continue
            selected_entry = self.selected_conversation if self.tab == "Conversations" else self.selected_npc
            selected = entry is selected_entry
            pygame.draw.rect(surface, ACCENT if selected else GRASS, row)
            for index, line in enumerate(lines):
                surface.blit(app.font.render(line, True, PAPER if selected else INK), (row.x + 8, row.y + 10 + index * app.font.get_linesize()))
        surface.set_clip(self.detail_rect.clip(previous_clip))
        y = self.detail_rect.top - self.detail_scroll
        for line, font, color, indent in self._detail_lines(app):
            if self.detail_rect.top - font.get_linesize() < y < self.detail_rect.bottom:
                surface.blit(font.render(line, True, color), (self.detail_rect.x + indent, y))
            y += font.get_linesize()
        for top, actor, emotion, labels in self.dialogue_portraits:
            rect = pygame.Rect(self.detail_rect.x, self.detail_rect.top + top - self.detail_scroll, 88, 88)
            if rect.bottom + 6 + len(labels) * app.small_font.get_linesize() < self.detail_rect.top or rect.top >= self.detail_rect.bottom:
                continue
            pygame.draw.rect(surface, GRASS, rect)
            if actor is app.simulation.snake:
                art = snake_artwork(4, 1)
                ratio = min(80 / art.get_width(), 80 / art.get_height())
                portrait = pygame.transform.scale(art, (round(art.get_width() * ratio), round(art.get_height() * ratio)))
            else:
                portrait = app.character.renderer.render_portrait(actor.character, rect.inflate(-8, -8).size, emotion=emotion)
            surface.blit(portrait, portrait.get_rect(center=rect.center))
            for index, label in enumerate(labels):
                surface.blit(app.small_font.render(label, True, MUTED),
                             (rect.x, rect.bottom + 6 + index * app.small_font.get_linesize()))
        surface.set_clip(previous_clip)
        for pane, rect in (("list", self.list_rect), ("detail", self.detail_rect)):
            if limits[pane]:
                track, thumb = self._scrollbar(rect, getattr(self, f"{pane}_scroll"), limits[pane])
                pygame.draw.rect(surface, GRASS, track)
                pygame.draw.rect(surface, MUTED, thumb)
        surface.blit(app.small_font.render("Scroll or use Up/Down and Page Up/Down. Escape returns to the game.", True, MUTED), (184, 655))
