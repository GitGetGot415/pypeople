import pygame
from interactions import player_request_pending
from time import monotonic
from character_data import DEFAULT_COLORS, IDENTITY_FIELDS, PROFILE_FIELDS
from conversation_history import (has_failed_voice_rewrite, split_opinion_outcome, usable_rewrite,
                                 uses_original_dialogue, voice_rewrite_errors)
from emotions import portrait_expression
from npc import NPC, ConversationActor
from social_records import claim_by_id, fact_by_id, phase_label
from information import assertion_text, selected_beliefs
from social import concern_description
from .creations import CreationsGallery
from .button import Button
from .text import paginate_turn, wrap_text, scrollbar_rects, ScrollbarDrag
from .character_sprite import draw_actor_portrait
from constants import (STATS_PORTRAIT_PADDING, CONVERSATION_FAILED, ACCENT, FAILED_CONVERSATION_COLOR,
                       FAILED_CONVERSATION_SELECTED_COLOR, GENERATING_CONVERSATION_COLOR,
                       GENERATING_CONVERSATION_SELECTED_COLOR, GIFT_CONVERSATION_COLOR,
                       GIFT_CONVERSATION_SELECTED_COLOR, QUEUED_CONVERSATION_COLOR,
                       QUEUED_CONVERSATION_SELECTED_COLOR, REWRITE_FAILED_CONVERSATION_COLOR,
                       REWRITE_FAILED_CONVERSATION_SELECTED_COLOR, GRASS, INK, MODEL_LOG_COLORS,
                       MODEL_LOG_REFRESH_SECONDS, MUTED, ORIGINAL_DIALOGUE_PAPER, PAPER, SNAKE_NAME,
                       STATS_TAB_RECTS, STATS_FOLLOW_RECT, STATS_INVENTORY_RECT, STATS_INVENTORY_DETAIL_TOP,
                       STATS_SORT_RECT, OPINION_BAR_LEFT, OPINION_BAR_WIDTH, OPINION_POSITIVE_COLOR,
                       OPINION_NEGATIVE_COLOR, PLAYER_LABEL, STATS_VIEW_RECT, STATS_SUBTAB_RECTS, STATS_COLLAPSED_ROWS,
                       GRID_LABEL_WIDTH, GRID_MAX_CELL, STATS_DEAD_COLOR, STATS_DEAD_SELECTED_COLOR)

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

    def _detail_lines(self, app):
        lines = []
        self.private_toggle_top = None
        self.opinion_bars = []
        self.hits = []
        self.dividers = []
        self.grid = None
        self.dialogue_portraits = []
        self.dialogue_backgrounds = []
        self.dead_backgrounds = []

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

        def rule():
            self.dividers.append(height() + app.small_font.get_linesize() // 2)
            gap()

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
            rule()
            add("What they wanted to discuss", app.heading_font)
            if record.intended_subjects:
                for name in record.participants:
                    if name in record.intended_subjects:
                        key = ("subject", record.id, name)
                        opened = key in self.expanded
                        top = height()
                        add(f"{'-' if opened else '+'} {name}", color=ACCENT)
                        self._link(top, height(), lambda key=key: self.expanded.symmetric_difference_update({key}))
                        if opened:
                            add(record.intended_subjects[name])
                        gap()
                add("Click a name to reveal their interest at the start; the conversation may follow another subject.", app.small_font, MUTED)
            else:
                add("Individual subjects were not recorded for this exchange.", app.small_font, MUTED)
            rule()
            if record.dialogue_errors:
                add("Base dialogue validation", app.heading_font)
                for reason in record.dialogue_errors:
                    add(reason, app.small_font, FAILED_CONVERSATION_SELECTED_COLOR)
                add("Validation was off; dialogue accepted and outcomes applied.", app.small_font, MUTED)
                rule()
            rewrite_errors = {}
            for index, speaker, reason in voice_rewrite_errors(record):
                rewrite_errors.setdefault(reason, []).append(f"{speaker} (turn {index + 1})")
            if rewrite_errors:
                add("Voice rewrite validation", app.heading_font)
                for reason, affected in rewrite_errors.items():
                    add(", ".join(affected), app.small_font, MUTED)
                    add(reason, app.small_font, REWRITE_FAILED_CONVERSATION_SELECTED_COLOR)
                    gap()
                rule()
            add("Outcome", app.heading_font)
            if record.error:
                add(record.error, color=FAILED_CONVERSATION_SELECTED_COLOR)
                gap()
            for index, outcome in enumerate(record.outcomes):
                if record.error and " ".join(outcome.split()) == " ".join(record.error.split()):
                    continue
                change = split_opinion_outcome(outcome)
                if change is None:
                    add(outcome)
                    gap()
                    continue
                summary, reason, direction = change
                key = ("outcome", record.id, index)
                opened = key in self.expanded
                top = height()
                add(f"{'-' if opened else '+'} {summary}",
                    color=OPINION_POSITIVE_COLOR if direction > 0 else OPINION_NEGATIVE_COLOR if direction < 0 else INK)
                self._link(top, height(), lambda key=key: self.expanded.symmetric_difference_update({key}))
                if opened:
                    add(reason, app.small_font, MUTED, 16)
                gap()
            if not record.outcomes and not record.error:
                add("No outcome recorded yet.", color=MUTED)
                gap()
            elif any(split_opinion_outcome(outcome) for outcome in record.outcomes):
                add("Click an opinion change to read the reasoning.", app.small_font, MUTED)
                gap()
            rule()
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
                rule()
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
        add(PLAYER_LABEL if npc is app.simulation.player else npc.name, app.heading_font)
        gap()
        if self.tab == "Opinions" and not npc.alive:
            add("Departed. These opinions stopped changing at death.", app.small_font, MUTED)
            gap()
        if npc is app.simulation.snake and self.tab == "Statistics":
            add(f"Activity: {npc.status}")
            add(f"Tree stock: {app.tree.fruit}")
            add(f"{SNAKE_NAME} seeks influence through known accounts. Religious explanations remain its claims.", app.small_font, MUTED)
            gap()
            add("Known accounts (observer inspection)", app.heading_font)
            for account in npc.snapshot()["private_context"]["known_accounts"]:
                add(account["assertion"], app.small_font)
                add(f"Source: {account['immediate_source_name']}; original: {account['original_speaker_name']}", app.small_font, MUTED)
            return lines
        if self.tab == "Statistics" and npc is app.simulation.player:
            self._epitaph(app, npc, add, gap, rule, height)
        elif self.tab == "Statistics":
            self._statistics(app, npc, add, gap, rule, height)
        elif self.opinion_view == "Grid" and npc is not app.simulation.player:
            self._opinion_grid(app, npc, add, gap, height)
        else:
            self._opinion_table(app, npc, add, gap, height)
        return lines

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

    def _epitaph(self, app, actor, add, gap, rule, height):
        """What remains of someone who died: how, when, and what they left behind."""
        sim = app.simulation
        death = next(f for f in sim.facts if f.kind == "killing" and f.object_id == actor.id)
        killer = sim.participant_by_id(death.subject_id)
        add(f"Killed by {PLAYER_LABEL if killer is sim.player else killer.name} / {phase_label(death.phase)}")
        if death.conversation_id:
            top = height()
            add("View confrontation", app.small_font, ACCENT)
            self._link(top, height(), lambda: self._open_conversation(app, death.conversation_id))
        rule()
        if isinstance(actor, NPC):
            add("Last on their mind", app.heading_font)
            add(concern_description(actor, actor.current_concern))
            rule()
        add("Left behind", app.heading_font)
        add(f"Inventory: {actor.inventory.get('fruit', 0)} apples")
        add(f"Owned works: {len(sim.owned_works(actor.id))} / creations made: {sum(work.creator_id == actor.id for work in sim.creations)}")
        if isinstance(actor, NPC):
            add(f"Opinion of player: {actor.player_opinion.opinion:+d}. {actor.player_opinion.reason}")

    def _statistics(self, app, npc, add, gap, rule, height):
        sim = app.simulation
        page = self.subtab
        if page == "Overview" and not npc.alive:
            self._epitaph(app, npc, add, gap, rule, height)
        elif page == "Overview":
            add("On their mind", app.heading_font)
            add(concern_description(npc, npc.current_concern))
            if npc.current_concern and npc.current_concern.remaining == 0:
                add("Will choose a new subject when free.", app.small_font, MUTED)
            rule()
            add("Status", app.heading_font)
            add(f"Activity: {npc.status}")
            add(f"Why: {npc.decision_reason}")
            add(f"Control: {npc.owner.replace('_', ' ')}", app.small_font, MUTED)
            if npc.interaction_blocked:
                add("Player request pending; new decisions are deferred.", app.small_font, MUTED)
            add(f"Arrival: day {npc.arrival_day} / Room {npc.room_number} / {npc.location}")
            rule()
            add("Condition", app.heading_font)
            add(f"Satisfaction: {npc.social:.0f}/100 / fatigue {npc.social_fatigue:.0f}/100")
            add(f"Carrying: {npc.inventory.get('fruit', 0)} apples (unlimited storage)")
            add(f"Owned works: {len(sim.owned_works(npc.id))} (unlimited storage) / creations made: {sum(work.creator_id == npc.id for work in sim.creations)}")
            add(f"Opinion of player: {npc.player_opinion.opinion:+d}. {npc.player_opinion.reason}")
            add(f"Sociability: {npc.preferences['sociability']:.0%}; generosity: {npc.preferences['generosity']:.0%}", app.small_font, MUTED)
            rule()
            add("Personal objective", app.heading_font)
            goal = npc.goal
            if goal:
                add(goal.objective)
                add(f"{goal.status.title()} / due {phase_label(goal.deadline)}", app.small_font, MUTED)
                if goal.outcome:
                    add(goal.outcome, app.small_font)
            else:
                add("No feasible personal objective right now.", color=MUTED)
        elif page == "Profile":
            for field in PROFILE_FIELDS:
                add(field, app.heading_font)
                add(npc.character[field] or "Not specified")
                gap()
            rule()
            for field in (*IDENTITY_FIELDS, "Hair", "Eyes", *DEFAULT_COLORS):
                value = npc.name if field == "Name" else npc.character.get(field, "")
                add(f"{field}: {value or 'Not specified'}")
        elif page == "Social":
            add("Apple promises", app.heading_font)
            promises = [p for p in sim.promises if npc.id in (p.promisor_id, p.recipient_id)]
            promises.sort(key=lambda p: (p.status == "pending", p.created_phase), reverse=True)
            if not promises:
                add("No promises yet.", color=MUTED)
            for promise in (promises if "promises" in self.expanded else promises[:STATS_COLLAPSED_ROWS]):
                giver = sim.participant_by_id(promise.promisor_id)
                recipient = sim.participant_by_id(promise.recipient_id)
                top = height()
                add(f"{giver.name} -> {recipient.name}: one apple / {promise.status} [view]", app.small_font, ACCENT)
                self._link(top, height(), lambda cid=promise.conversation_id: self._open_conversation(app, cid))
                add(f"Due {phase_label(promise.deadline)}", app.small_font, MUTED)
                if promise.outcome:
                    add(promise.outcome, app.small_font)
            if len(promises) > STATS_COLLAPSED_ROWS:
                self._toggle(app, add, height, "promises",
                             "Show fewer promises" if "promises" in self.expanded else f"Show all {len(promises)} promises")
            if npc.pending_social_opinions:
                add("An opinion could not be saved; the simulation will retry.", app.small_font, MUTED)
            rule()
            add("Recent memories", app.heading_font)
            memories = list(reversed(npc.memories))
            if not memories:
                add("No personal experiences remembered yet.", color=MUTED)
            for memory in (memories if "memories" in self.expanded else memories[:STATS_COLLAPSED_ROWS]):
                add(f"Day {memory.day} / {memory.phase.title()}: {memory.summary}")
                add(f"Influence: {memory.weight:+g}", app.small_font, MUTED)
            if len(memories) > STATS_COLLAPSED_ROWS:
                self._toggle(app, add, height, "memories",
                             "Show fewer memories" if "memories" in self.expanded else f"Show all {len(memories)} memories")
        else:
            add("Observer view", app.heading_font)
            add("Private motives and accounts are hidden until you choose to show them.", app.small_font, MUTED)
            self.private_toggle_top = height()
            add(f"{'Hide' if self.show_private else 'Show'} private details (observer only)", app.small_font, ACCENT)
            if not self.show_private:
                return
            goal = npc.goal
            if goal:
                rule()
                add("Motive and approach", app.heading_font)
                add(f"Motive: {goal.motive}")
                add(f"Approach: {goal.approach}")
                add(f"Boundary: {goal.boundary}")
            rule()
            add("Known accounts and sources", app.heading_font)
            if not npc.beliefs:
                add("No firsthand accounts or rumors yet.", app.small_font, MUTED)
            else:
                add("Click an account to expand its source and the observer's verdict.", app.small_font, MUTED)
            for belief in selected_beliefs(npc):
                claim = claim_by_id(sim, belief.claim_id)
                fact = fact_by_id(sim, claim.fact_id)
                key = ("account", belief.claim_id)
                opened = key in self.expanded
                top = height()
                add(f"{'-' if opened else '+'} {assertion_text(sim, fact, claim.affirmed)}", app.small_font)
                add(f"{belief.stance.title()} / {'firsthand' if belief.direct else 'personal assertion' if belief.source_id == npc.id else 'hearsay'}", app.small_font, MUTED)
                self._link(top, height(), lambda key=key: self.expanded.symmetric_difference_update({key}))
                if opened:
                    source = sim.participant_by_id(belief.source_id)
                    originator = sim.participant_by_id(claim.originator_id)
                    add(f"Immediate source: {source.name}; original account: {originator.name}", app.small_font, MUTED, 16)
                    add(f"Observer: {'supported by the actual event or intention' if claim.affirmed else 'contradicts the actual event or intention'}.", app.small_font, MUTED, 16)
                    if claim.deliberate:
                        add(f"Observer: the original assertion was deliberately fabricated by {originator.name}.", app.small_font, MUTED, 16)
                    if belief.disclosure_id:
                        disclosure = next(d for d in sim.disclosures if d.id == belief.disclosure_id)
                        link = height()
                        add("View source conversation", app.small_font, ACCENT, 16)
                        self._link(link, height(), lambda cid=disclosure.conversation_id: self._open_conversation(app, cid))
                gap()

    @staticmethod
    def _person_label(app, person):
        return PLAYER_LABEL if person is app.simulation.player else person.name

    def _opinion_grid(self, app, npc, add, gap, height):
        """Every living resident's and the python's opinion of every other, as a colored matrix."""
        sim = app.simulation
        people = [*sim.actors, sim.player]

        def opinion(a, b):
            if a is b or a is sim.player:
                return None
            if b is sim.player:
                held = getattr(a, "player_opinion", None)
                return held.opinion if held and (held.opinion or held.reason) else None
            return a.opinions.get(b.id, {}).get("opinion")

        add("Each row is that person's opinion of the person in the column. Green likes, red dislikes, "
            "blank has no opinion. Click a cell to read the pair.", app.small_font, MUTED)
        gap()
        if len(people) < 2:
            add("Not enough residents for a grid yet.", color=MUTED)
            return
        cell = min(GRID_MAX_CELL, (self.detail_rect.width - 40 - GRID_LABEL_WIDTH) // len(people))
        header = app.small_font.get_linesize() + 4
        top = height()
        total = header + cell * len(people)
        self.grid = dict(top=top, cell=cell, header=header, people=people,
                         values=[[opinion(a, b) for b in people] for a in people])
        pad = -(-total // app.small_font.get_linesize())
        for _ in range(pad):
            gap()

    def _opinion_table(self, app, npc, add, gap, height):
        """Each person with this NPC's opinion of them and their opinion of this NPC, side by side."""
        sim = app.simulation
        is_player = npc is sim.player
        first = PLAYER_LABEL if is_player else npc.name.split()[0]
        rows = {}
        dead_names = {self._person_label(app, person) for person in (*sim.actors, sim.player) if not person.alive}
        if is_player:
            # The player holds no opinions of their own; show what each resident thought of them.
            for other in (*sim.npcs, *sim.archived):
                if other.player_opinion.opinion or other.player_opinion.reason:
                    rows[other.name] = {"mine": None, "theirs": {"opinion": other.player_opinion.opinion,
                                                                 "reason": other.player_opinion.reason}}
        else:
            for person, entry in npc.named_opinions().items():
                rows[person] = {"mine": entry, "theirs": None}
            for other in sim.actors:
                entry = other.named_opinions().get(npc.name) if other is not npc else None
                if entry is not None:
                    rows.setdefault(other.name, {"mine": None, "theirs": None})["theirs"] = entry
        if not is_player and (npc.player_opinion.opinion or npc.player_opinion.reason):
            rows[PLAYER_LABEL] = {"mine": {"opinion": npc.player_opinion.opinion, "reason": npc.player_opinion.reason},
                                  "theirs": None, "player": True}
        add("Bars run from -100 (left) to +100 (right). Click a person to read both reasons.", app.small_font, MUTED)
        gap()
        if not rows:
            add("No opinions recorded yet.", color=MUTED)

        def order(item):
            person, row = item
            if self.opinion_sort == "Score":
                score = row["mine"]["opinion"] if row["mine"] else None
                return (person in dead_names, score is None, -(score or 0), person.casefold())
            return (person in dead_names, False, 0, person.casefold())

        has_dead = any(person in dead_names for person in rows)
        category = None
        for person, row in sorted(rows.items(), key=order):
            dead = person in dead_names
            if has_dead and category != dead:
                add("Dead" if dead else "Living", app.heading_font)
                gap()
                category = dead
            top = height()
            add(person, app.font)
            for label, entry in ((f"{first} on them", row["mine"]), (f"They on {first}", row["theirs"])):
                if entry is None and (row.get("player") or is_player):
                    continue
                self.opinion_bars.append((height(), label, entry["opinion"] if entry else None))
                gap()
            if self.selected_opinion == person:
                if row["mine"]:
                    add(f"{first}'s reason: {row['mine']['reason']}", app.small_font)
                if row["theirs"]:
                    add(f"Their reason: {row['theirs']['reason']}", app.small_font)
                if not dead:
                    self.dialogue_backgrounds.append((top, height()))
            if dead:
                self.dead_backgrounds.append((top, height(), self.selected_opinion == person))
            self._link(top, height(), lambda person=person: setattr(
                self, "selected_opinion", None if self.selected_opinion == person else person))
            gap()

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

        if self.tab == "Statistics" and isinstance(self.selected_npc, NPC):
            for name, button in self.subtabs.items():
                if button.clicked(event):
                    self.subtab = name
                    self.detail_scroll = 0
                    return

        if self.tab == "Opinions" and isinstance(self.selected_npc, ConversationActor) and self.view_toggle.clicked(event):
            self.opinion_view = "Grid" if self.opinion_view == "Table" else "Table"
            self.detail_scroll = 0
            return

        if (self.tab == "Opinions" and self.opinion_view == "Table" and isinstance(self.selected_npc, ConversationActor)
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
                            self.selected_opinion = self._person_label(app, grid["people"][column]) if 0 <= column < len(grid["people"]) and column != row else None
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
        app.game.draw_clock(app)
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
        lines = self._detail_lines(app)
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
            self._draw_grid(app, surface)
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
        if self.tab == "Opinions" and isinstance(self.selected_npc, ConversationActor):
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

    def _draw_grid(self, app, surface):
        grid = self.grid
        cell, header, people = grid["cell"], grid["header"], grid["people"]
        x0 = self.detail_rect.x + GRID_LABEL_WIDTH
        y0 = self.detail_rect.top + grid["top"] - self.detail_scroll
        line = app.small_font.get_linesize()
        for column in range(len(people)):
            label = app.small_font.render(str(column + 1), True, MUTED)
            surface.blit(label, label.get_rect(center=(x0 + column * cell + cell // 2, y0 + line // 2)))
        for row, person in enumerate(people):
            y = y0 + header + row * cell
            if y + cell < self.detail_rect.top or y > self.detail_rect.bottom:
                continue
            name = self._person_label(app, person)
            name = name if len(name) <= 18 else name[:17] + "."
            color = INK if person is self.selected_npc else MUTED
            surface.blit(app.small_font.render(f"{row + 1}. {name}", True, color), (self.detail_rect.x + 4, y + (cell - line) // 2))
            for column, value in enumerate(grid["values"][row]):
                rect = pygame.Rect(x0 + column * cell, y, cell - 2, cell - 2)
                if value is None:
                    pygame.draw.rect(surface, MUTED if row == column else GRASS, rect, 0 if row == column else 1)
                    continue
                target = OPINION_POSITIVE_COLOR if value >= 0 else OPINION_NEGATIVE_COLOR
                strength = abs(value) / 100
                fill = tuple(round(PAPER[i] + (target[i] - PAPER[i]) * strength) for i in range(3))
                pygame.draw.rect(surface, fill, rect)
            if person is self.selected_npc:
                pygame.draw.rect(surface, ACCENT, (x0 - 2, y - 1, cell * len(people), cell), 2)

    def _position_detail(self):
        top = STATS_INVENTORY_DETAIL_TOP if self.tab == "Statistics" and isinstance(self.selected_npc, NPC) else 200
        self.detail_rect.top = top
        self.detail_rect.height = 624 - top
