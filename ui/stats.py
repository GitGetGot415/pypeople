import pygame
from interactions import player_request_pending
from time import monotonic
from character_data import DEFAULT_COLORS, IDENTITY_FIELDS, PROFILE_FIELDS
from conversation_history import (has_failed_voice_rewrite, usable_rewrite, uses_original_dialogue,
                                 voice_rewrite_errors)
from emotions import portrait_expression
from npc import NPC
from social import assertion_text, claim_by_id, concern_description, fact_by_id, phase_label, selected_beliefs
from .creations import CreationsGallery
from .button import Button
from .text import paginate_turn, wrap_text, scrollbar_rects
from .character_sprite import draw_actor_portrait
from constants import (STATS_PORTRAIT_PADDING, CONVERSATION_FAILED, ACCENT, FAILED_CONVERSATION_COLOR,
                       FAILED_CONVERSATION_SELECTED_COLOR, GENERATING_CONVERSATION_COLOR,
                       GENERATING_CONVERSATION_SELECTED_COLOR, GIFT_CONVERSATION_COLOR,
                       GIFT_CONVERSATION_SELECTED_COLOR, QUEUED_CONVERSATION_COLOR,
                       QUEUED_CONVERSATION_SELECTED_COLOR, REWRITE_FAILED_CONVERSATION_COLOR,
                       REWRITE_FAILED_CONVERSATION_SELECTED_COLOR, GRASS, INK, MODEL_LOG_COLORS,
                       MODEL_LOG_REFRESH_SECONDS, MUTED, ORIGINAL_DIALOGUE_PAPER, PAPER, SNAKE_NAME,
                       STATS_TAB_RECTS, STATS_FOLLOW_RECT, STATS_INVENTORY_RECT, STATS_INVENTORY_DETAIL_TOP)


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
        self.list_rect = pygame.Rect(200, 184, 264, 440)
        self.detail_rect = pygame.Rect(512, 200, 568, 424)
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
        self.show_private = False
        self.private_toggle_top = None
        self.promise_links = []
        self.dialogue_portraits = []
        self.dialogue_backgrounds = []

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
            entries = [(work, f"{work.kind.title()}: {work.title}\n{app.simulation.actor_by_id(work.creator_id).name}\nDay {work.day} / {work.phase.title()}")
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
            entries = [(npc, npc.name + (" (departed)" if getattr(npc, "departure", None) else "")) for npc in app.simulation.actors]
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

    def _detail_lines(self, app):
        lines = []
        self.private_toggle_top = None
        self.promise_links = []
        self.dialogue_portraits = []
        self.dialogue_backgrounds = []

        if self.tab == "Creations":
            return lines

        def add(text, font=app.font, color=INK, indent=0):
            for paragraph in str(text).splitlines() or [""]:
                for line in wrap_text(paragraph, font, self.detail_rect.width - 20 - indent) or [""]:
                    lines.append((line, font, color, indent))

        def gap():
            lines.append(("", app.small_font, MUTED, 0))

        def height():
            return sum(font.get_linesize() for _, font, _, _ in lines)

        def message(turn, *, original=True):
            speaker = turn.speaker
            npc = app.simulation.participant_by_id(turn.speaker_id)
            if npc is app.simulation.player:
                npc = None
            indent = 112 if npc is not None else 0
            for emotion, page in paginate_turn(turn, app.font, self.detail_rect.width - 20 - indent,
                    original=app.original_dialogue, condensed=getattr(app, "condensed_dialogue", False)):
                top = height()
                add(speaker, color=ACCENT, indent=indent)
                add(page, indent=indent)
                if npc is not None:
                    expression = portrait_expression(emotion["name"], emotion["intensity"]) if emotion is not None else None
                    label = (expression or "neutral").replace("_", " ").title() if emotion is not None else "Not recorded"
                    labels = wrap_text(label, app.small_font, 100)
                    self.dialogue_portraits.append((top, npc, expression, labels))
                    portrait_height = 94 + len(labels) * app.small_font.get_linesize()
                    while height() - top < portrait_height:
                        gap()
                if original:
                    self.dialogue_backgrounds.append((top, height()))
                gap()

        if self.tab == "Model logs":
            log = self.selected_log
            if log is None:
                add("Select a model request to inspect its output, reasoning, stages, and errors.", color=MUTED)
                add("Yellow: running. Orange: queued. Green: completed. Red: unusable.", app.small_font)
                add("Amber: waiting for Ollama output.", app.small_font)
                add(f"Saved in: {app.requests.debug_logs.directory}", app.small_font, MUTED)
                if app.requests.debug_logs.read_error:
                    add(app.requests.debug_logs.read_error, app.small_font, MUTED)
                return lines
            revision, events, save_error = log.snapshot()
            cache_key = (log, revision, save_error, log.status)
            now = monotonic()
            if self._log_cache is not None:
                previous_key, previous_lines, refreshed_at = self._log_cache
                if previous_key == cache_key or (previous_key[0] is log and previous_key[2] == save_error
                                                and now - refreshed_at < MODEL_LOG_REFRESH_SECONDS):
                    return previous_lines
            meta = log.metadata
            add(meta["participants"], app.heading_font)
            add(f"Status: {log.status_label}", app.small_font)
            if log.status == "waiting_for_ollama":
                add("Waiting for the first text or reasoning token. Ollama may be queued, loading, or processing the prompt.",
                    app.small_font, MUTED)
            slot = f"Thread {meta['slot']}" if meta['slot'] is not None else "Waiting for a thread"
            add(f"{meta['provider']} / {meta['model']} / {slot}", app.small_font, MUTED)
            add(f"Started: {meta['started']}", app.small_font, MUTED)
            add(f"Request: {meta['request_id']} / Session: {meta['session_id']}", app.small_font, MUTED)
            add(meta["topic"], app.small_font)
            add(f"Saved in: {log.path}", app.small_font, MUTED)
            if save_error:
                add(save_error, color=ACCENT)
            gap()
            add("Raw diagnostics; generated output has not necessarily been accepted.", app.small_font, MUTED)
            add("Reasoning is shown only when supplied by the provider. Gemini supplies summaries.", app.small_font, MUTED)
            gap()
            labels = {"output": "Model output", "reasoning": "Reasoning / summary"}
            for event in events:
                add(f"{event['elapsed']:.1f}s / {labels.get(event['kind'], event['kind'].title())}", app.small_font, ACCENT)
                add(event["text"].replace("\x00", "[NUL]"), app.small_font)
                gap()
            if not any(event["kind"] in ("output", "reasoning", "finished", "error") for event in events):
                add("Waiting for provider output; model loading and prompt evaluation may take time.", app.small_font, MUTED)
            self._log_cache = (cache_key, lines, now)
            return lines

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
            add("What they wanted to discuss", app.heading_font)
            if record.intended_subjects:
                for name in record.participants:
                    if name in record.intended_subjects:
                        add(name, color=ACCENT)
                        add(record.intended_subjects[name])
                        gap()
                add("Interests at the start; the conversation may follow another subject.", app.small_font, MUTED)
            else:
                add("Individual subjects were not recorded for this exchange.", app.small_font, MUTED)
            gap()
            if record.dialogue_errors:
                add("Base dialogue validation", app.heading_font)
                for reason in record.dialogue_errors:
                    add(reason, app.small_font, FAILED_CONVERSATION_SELECTED_COLOR)
                add("Validation was off; dialogue accepted and outcomes applied.", app.small_font, MUTED)
                gap()
            rewrite_errors = {}
            for index, speaker, reason in voice_rewrite_errors(record):
                rewrite_errors.setdefault(reason, []).append(f"{speaker} (turn {index + 1})")
            if rewrite_errors:
                add("Voice rewrite validation", app.heading_font)
                for reason, affected in rewrite_errors.items():
                    add(", ".join(affected), app.small_font, MUTED)
                    add(reason, app.small_font, REWRITE_FAILED_CONVERSATION_SELECTED_COLOR)
                    gap()
            add("Outcome", app.heading_font)
            if record.error:
                add(record.error, color=FAILED_CONVERSATION_SELECTED_COLOR)
                gap()
            for outcome in record.outcomes:
                if record.error and " ".join(outcome.split()) == " ".join(record.error.split()):
                    continue
                add(outcome)
                gap()
            if not record.outcomes and not record.error:
                add("No outcome recorded yet.", color=MUTED)
                gap()
            add("Dialogue", app.heading_font)
            gap()
            available = [usable_rewrite(turn.parts, turn.rewritten)
                         for turn in record.turns if turn.speaker != "You"]
            if not app.original_dialogue and available and not all(available):
                add("No voice rewrite was stored; showing original dialogue." if not any(available)
                    else "Some lines use original dialogue because no voice rewrite was stored.",
                    app.small_font, MUTED)
                gap()
            if not record.turns and not record.failed_turns:
                add("No generated draft was retained for this failure." if record.error else "No dialogue generated yet.", color=MUTED)
            for turn in record.display_turns():
                message(turn,
                        original=uses_original_dialogue(turn.parts, turn.rewritten, original=app.original_dialogue))
            if record.failed_turns:
                add("Generated draft (not applied)", app.heading_font)
                add("Generation failed. These words did not change the simulation.", app.small_font, MUTED)
                gap()
                for turn in record.display_turns(failed=True):
                    message(turn)
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
            add(f"{SNAKE_NAME} claims divine authority. Claims about destinations and other zones are unverified.", app.small_font, MUTED)
            gap()
            add("Warnings", app.heading_font)
            for actor_id in npc.offenses:
                add(f"{app.simulation.actor_by_id(actor_id).name}: warned of being eaten.")
            gap()
            add("Known accounts (observer inspection)", app.heading_font)
            for account in npc.snapshot()["known_accounts"]:
                add(account["assertion"], app.small_font)
                add(f"Source: {account['immediate_source_name']}; original: {account['original_speaker_name']}", app.small_font, MUTED)
            return lines
        if self.tab == "Statistics":
            add("Last on their mind" if npc.departure else "On their mind", app.heading_font)
            add(concern_description(npc, npc.current_concern))
            if npc.current_concern and npc.current_concern.remaining == 0 and not npc.departure:
                add("Will choose a new subject when free.", app.small_font, MUTED)
            gap()
            for field in PROFILE_FIELDS:
                add(field, app.heading_font)
                add(npc.character[field] or "Not specified")
                gap()
            add(f"Arrival: day {npc.arrival_day}")
            add(f"Activity: {npc.status}")
            add(f"Room {npc.room_number} / {npc.location} / fatigue {npc.social_fatigue:.0f}/100")
            add(f"Creations: {sum(work.creator_id == npc.id for work in app.simulation.creations)}")
            if npc.departure:
                source = next(f for f in app.simulation.facts if f.source_id == npc.departure["event_id"] and f.kind == "eating")
                top = height()
                add("View departure encounter", app.small_font, ACCENT)
                self.promise_links.append((top, height(), source.conversation_id))
            add(f"Why: {npc.decision_reason}")
            add(f"Control: {npc.owner.replace('_', ' ')}", app.small_font, MUTED)
            if npc.interaction_blocked:
                add("Player request pending; new decisions are deferred.", app.small_font, MUTED)
            gap()
            add(f"Satisfaction: {npc.social:.0f}/100")
            inventory_label = "Inventory at departure" if npc.departure else "Carrying"
            add(f"{inventory_label}: {npc.inventory.get('fruit', 0)} apples (unlimited storage)")
            add(f"Owned works: {len(app.simulation.owned_works(npc.id))} (unlimited storage)")
            add(f"Opinion of player: {npc.player_opinion.opinion:+d}. {npc.player_opinion.reason}")
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
                add(f"Attitudes toward {SNAKE_NAME}", app.heading_font)
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
                    source = app.simulation.participant_by_id(belief.source_id)
                    originator = app.simulation.participant_by_id(claim.originator_id)
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
                giver = app.simulation.participant_by_id(promise.promisor_id)
                recipient = app.simulation.participant_by_id(promise.recipient_id)
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
            for field in (*IDENTITY_FIELDS, "Hair", "Eyes", *DEFAULT_COLORS):
                value = npc.name if field == "Name" else npc.character.get(field, "")
                add(f"{field}: {value or 'Not specified'}")
        else:
            add("Opinion scores range from -100 to 100.", app.small_font, MUTED)
            gap()
            opinions = npc.named_opinions()
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
        detail_height = (self.gallery.layout(app)[1] if self.tab == "Creations" else
                         sum(font.get_linesize() for _, font, _, _ in self._detail_lines(app)))
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
        if self.tab == "Statistics" and isinstance(self.selected_npc, NPC) and self.inventory_button.clicked(event):
            app.game.inventory.open(self.selected_npc)
            return
        if app.game.handle_clock_event(app, event):
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
                    for top, bottom, record_id in self.promise_links:
                        if top <= content_y < bottom:
                            record = next(entry for entry in app.conversation_history if entry.id == record_id)
                            self.open_record(record)
                            return
                track, thumb = self._scrollbar(rect, scroll, limits[pane])
                if limits[pane] and track.collidepoint(pos):
                    if pane == "detail" and self.tab == "Model logs":
                        self.follow_log = False
                    offset = pos[1] - track.top - thumb.height / 2
                    self._scroll(pane, offset * limits[pane] / (track.height - thumb.height), limits)
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
        app.game.draw_clock(app)
        pygame.draw.rect(surface, PAPER, (184, 134, 296, 506))
        pygame.draw.rect(surface, PAPER, (496, 134, 600, 506))
        list_title = ("Model requests" if self.tab == "Model logs" else
                      f"Conversations ({len(app.conversation_history)})" if self.tab == "Conversations" else
                      f"Living {app.simulation.population} / Departed {app.simulation.departed_count}")
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
            elif self.tab == "Conversations" and entry.id in generating_ids:
                color = GENERATING_CONVERSATION_SELECTED_COLOR if selected else GENERATING_CONVERSATION_COLOR
            elif self.tab == "Conversations" and entry.id in queued_ids:
                color = QUEUED_CONVERSATION_SELECTED_COLOR if selected else QUEUED_CONVERSATION_COLOR
            elif self.tab == "Conversations" and (entry.status == CONVERSATION_FAILED or entry.dialogue_errors):
                color = FAILED_CONVERSATION_SELECTED_COLOR if selected else FAILED_CONVERSATION_COLOR
            elif self.tab == "Conversations" and has_failed_voice_rewrite(entry):
                color = REWRITE_FAILED_CONVERSATION_SELECTED_COLOR if selected else REWRITE_FAILED_CONVERSATION_COLOR
            elif self.tab == "Conversations" and entry.id in gift_ids and not entry.turns and not entry.failed_turns:
                color = GIFT_CONVERSATION_SELECTED_COLOR if selected else GIFT_CONVERSATION_COLOR
            else:
                color = ACCENT if selected else GRASS
            pygame.draw.rect(surface, color, row)
            if selected and self.tab == "Model logs":
                pygame.draw.rect(surface, INK, row, 2)
            for index, line in enumerate(lines):
                surface.blit(app.font.render(line, True, PAPER if selected and self.tab != "Model logs" else INK),
                             (row.x + 8, row.y + 10 + index * app.font.get_linesize()))
        surface.set_clip(self.detail_rect.clip(previous_clip))
        lines = self._detail_lines(app)
        for top, bottom in self.dialogue_backgrounds:
            pygame.draw.rect(surface, ORIGINAL_DIALOGUE_PAPER,
                (self.detail_rect.x - 4, self.detail_rect.top + top - self.detail_scroll,
                 self.detail_rect.width - 8, bottom - top))
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
        surface.set_clip(previous_clip)
        if self.tab == "Statistics" and isinstance(self.selected_npc, NPC):
            self.inventory_button.draw(surface, app.small_font)
        if self.tab == "Creations":
            self.gallery.draw(app, self.detail_scroll)
        for pane, rect in (("list", self.list_rect), ("detail", self.detail_rect)):
            if limits[pane]:
                track, thumb = self._scrollbar(rect, getattr(self, f"{pane}_scroll"), limits[pane])
                pygame.draw.rect(surface, GRASS, track)
                pygame.draw.rect(surface, MUTED, thumb)
        surface.blit(app.small_font.render("Scroll or use Up/Down and Page Up/Down. Escape returns to the game.", True, MUTED), (184, 655))

    def _position_detail(self):
        top = STATS_INVENTORY_DETAIL_TOP if self.tab == "Statistics" and isinstance(self.selected_npc, NPC) else 200
        self.detail_rect.top = top
        self.detail_rect.height = 624 - top
