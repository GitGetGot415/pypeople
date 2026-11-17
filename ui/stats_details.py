import pygame
from time import monotonic
from persistence.characters import DEFAULT_COLORS, IDENTITY_FIELDS, PROFILE_FIELDS
from dialogue.history import (
    split_opinion_outcome, usable_rewrite, uses_original_dialogue, voice_rewrite_errors,
)
from dialogue.emotions import portrait_expression
from simulation.npc import NPC
from simulation.social.records import claim_by_id, fact_by_id
from simulation.social.information import assertion_text, selected_beliefs
from simulation.social.concerns import concern_description
from ui.text import paginate_turn, wrap_text
from constants import (
    ACCENT, FAILED_CONVERSATION_SELECTED_COLOR, REWRITE_FAILED_CONVERSATION_SELECTED_COLOR, GRASS, INK,
    MODEL_LOG_REFRESH_SECONDS, MUTED, PAPER, SNAKE_NAME, OPINION_POSITIVE_COLOR, OPINION_NEGATIVE_COLOR,
    PLAYER_LABEL, STATS_COLLAPSED_ROWS, GRID_LABEL_WIDTH, GRID_MAX_CELL,
)


class StatsDetails:
    def __init__(self, screen):
        self.screen = screen

    def lines(self, app):
        lines = []
        self.screen.private_toggle_top = None
        self.screen.opinion_bars = []
        self.screen.hits = []
        self.screen.dividers = []
        self.screen.grid = None
        self.screen.dialogue_portraits = []
        self.screen.dialogue_backgrounds = []
        self.screen.dead_backgrounds = []

        if self.screen.tab == "Creations":
            return lines

        def add(text, font=app.font, color=INK, indent=0):
            for paragraph in str(text).splitlines() or [""]:
                for line in wrap_text(paragraph, font, self.screen.detail_rect.width - 20 - indent) or [""]:
                    lines.append((line, font, color, indent))

        def gap():
            lines.append(("", app.small_font, MUTED, 0))

        def height():
            return sum(font.get_linesize() for _, font, _, _ in lines)

        def rule():
            self.screen.dividers.append(height() + app.small_font.get_linesize() // 2)
            gap()

        def message(turn, *, original=True):
            speaker = turn.speaker
            npc = app.simulation.participant_by_id(turn.speaker_id)
            if npc is app.simulation.player:
                npc = None
            indent = 112 if npc is not None else 0
            for emotion, page in paginate_turn(turn, app.font, self.screen.detail_rect.width - 20 - indent,
                    original=app.original_dialogue, condensed=getattr(app, "condensed_dialogue", False)):
                top = height()
                add(speaker, color=ACCENT, indent=indent)
                add(page, indent=indent)
                if npc is not None:
                    expression = portrait_expression(emotion["name"], emotion["intensity"]) if emotion is not None else None
                    label = (expression or "neutral").replace("_", " ").title() if emotion is not None else "Not recorded"
                    labels = wrap_text(label, app.small_font, 100)
                    self.screen.dialogue_portraits.append((top, npc, expression, labels))
                    portrait_height = 94 + len(labels) * app.small_font.get_linesize()
                    while height() - top < portrait_height:
                        gap()
                if original:
                    self.screen.dialogue_backgrounds.append((top, height()))
                gap()

        if self.screen.tab == "Model logs":
            log = self.screen.selected_log
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
            if self.screen.log_cache is not None:
                previous_key, previous_lines, refreshed_at = self.screen.log_cache
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
            self.screen.log_cache = (cache_key, lines, now)
            return lines

        if self.screen.tab == "Conversations":
            record = self.screen.selected_conversation
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
                        opened = key in self.screen.expanded
                        top = height()
                        add(f"{'-' if opened else '+'} {name}", color=ACCENT)
                        self.screen.link(top, height(), lambda key=key: self.screen.expanded.symmetric_difference_update({key}))
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
                opened = key in self.screen.expanded
                top = height()
                add(f"{'-' if opened else '+'} {summary}",
                    color=OPINION_POSITIVE_COLOR if direction > 0 else OPINION_NEGATIVE_COLOR if direction < 0 else INK)
                self.screen.link(top, height(), lambda key=key: self.screen.expanded.symmetric_difference_update({key}))
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
            # Errors appear where they happened, between the turns recorded before and after them.
            notes = sorted(record.notes, key=lambda note: note["turn"])

            def show_notes(before):
                while notes and notes[0]["turn"] <= before:
                    add(f"Error: {notes.pop(0)['text']}", app.small_font, FAILED_CONVERSATION_SELECTED_COLOR)
                    gap()

            for index, turn in enumerate(record.display_turns()):
                show_notes(index)
                message(turn,
                        original=uses_original_dialogue(turn.parts, turn.rewritten, original=app.original_dialogue))
            show_notes(len(record.turns))
            if record.failed_turns:
                rule()
                add("Generated draft (not applied)", app.heading_font)
                add("Generation failed. These words did not change the simulation.", app.small_font, MUTED)
                gap()
                for turn in record.display_turns(failed=True):
                    message(turn)
            return lines

        npc = self.screen.selected_npc
        if npc is None:
            add("Select an NPC to view their statistics and opinions.", color=MUTED)
            return lines
        add(PLAYER_LABEL if npc is app.simulation.player else npc.name, app.heading_font)
        gap()
        if self.screen.tab == "Opinions" and not npc.alive:
            add("Opinions held by dead NPCs are preserved at death.", app.small_font, MUTED)
            gap()
        if npc is app.simulation.snake and self.screen.tab == "Statistics":
            add(f"Activity: {npc.status}")
            add(f"Tree stock: {app.tree.fruit}")
            add(f"{SNAKE_NAME} seeks influence through known accounts. Religious explanations remain its claims.", app.small_font, MUTED)
            gap()
            add("Known accounts (observer inspection)", app.heading_font)
            for account in npc.snapshot()["private_context"]["known_accounts"]:
                add(account["assertion"], app.small_font)
                add(f"Source: {account['immediate_source_name']}; original: {account['original_speaker_name']}", app.small_font, MUTED)
            return lines
        if self.screen.tab == "Statistics" and npc is app.simulation.player:
            self._epitaph(app, npc, add, gap, rule, height)
        elif self.screen.tab == "Statistics":
            self._statistics(app, npc, add, gap, rule, height)
        elif self.screen.opinion_view == "Grid":
            self._opinion_grid(app, npc, add, gap, height)
        else:
            self._opinion_table(app, npc, add, gap, height)
        return lines

    def _epitaph(self, app, actor, add, gap, rule, height):
        """What remains of someone who died: how, when, and what they left behind."""
        sim = app.simulation
        death = next(f for f in sim.facts if f.kind == "killing" and f.object_id == actor.id)
        killer = sim.participant_by_id(death.subject_id)
        add(f"Killed by {PLAYER_LABEL if killer is sim.player else killer.name} / {death.day}")
        if death.conversation_id:
            top = height()
            add("View confrontation", app.small_font, ACCENT)
            self.screen.link(top, height(), lambda: self.screen.open_conversation(app, death.conversation_id))
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
        page = self.screen.subtab
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
            add(f"Arrival: {npc.arrival_day} / Room {npc.room_number} / {npc.location}")
            rule()
            add("Condition", app.heading_font)
            add(f"Satisfaction: {npc.social:.0f}/100 / fatigue {npc.social_fatigue:.0f}/100")
            if npc.alive:
                from simulation.nutrition import daily_satisfaction_loss, starves_next_day
                add(f"Next day: -{daily_satisfaction_loss(npc.social)} satisfaction", app.small_font, MUTED)
                if starves_next_day(npc):
                    add("Will die of starvation next day unless they eat.", app.small_font, INK)
            add(f"Carrying: {npc.inventory.get('fruit', 0)} apples (unlimited storage)")
            add(f"Owned works: {len(sim.owned_works(npc.id))} (unlimited storage) / creations made: {sum(work.creator_id == npc.id for work in sim.creations)}")
            add(f"Opinion of player: {npc.player_opinion.opinion:+d}. {npc.player_opinion.reason}")
            add(f"Sociability: {npc.preferences['sociability']:.0%}; generosity: {npc.preferences['generosity']:.0%}", app.small_font, MUTED)
            rule()
            add("Personal objective", app.heading_font)
            goal = npc.goal
            if goal:
                add(goal.objective)
                add(f"{goal.status.title()} / due {goal.deadline}", app.small_font, MUTED)
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
            promises.sort(key=lambda p: (p.status == "pending", p.created_day), reverse=True)
            if not promises:
                add("No promises yet.", color=MUTED)
            for promise in (promises if "promises" in self.screen.expanded else promises[:STATS_COLLAPSED_ROWS]):
                giver = sim.participant_by_id(promise.promisor_id)
                recipient = sim.participant_by_id(promise.recipient_id)
                top = height()
                add(f"{giver.name} -> {recipient.name}: one apple / {promise.status} [view]", app.small_font, ACCENT)
                self.screen.link(top, height(), lambda cid=promise.conversation_id: self.screen.open_conversation(app, cid))
                add(f"Due {promise.deadline}", app.small_font, MUTED)
                if promise.outcome:
                    add(promise.outcome, app.small_font)
            if len(promises) > STATS_COLLAPSED_ROWS:
                self.screen.toggle(app, add, height, "promises",
                             "Show fewer promises" if "promises" in self.screen.expanded else f"Show all {len(promises)} promises")
            if npc.pending_social_opinions:
                add("An opinion could not be saved; the simulation will retry.", app.small_font, MUTED)
            rule()
            add("Recent memories", app.heading_font)
            memories = list(reversed(npc.memories))
            if not memories:
                add("No personal experiences remembered yet.", color=MUTED)
            for memory in (memories if "memories" in self.screen.expanded else memories[:STATS_COLLAPSED_ROWS]):
                add(f"{memory.day}: {memory.summary}")
                add(f"Influence: {memory.weight:+g}", app.small_font, MUTED)
            if len(memories) > STATS_COLLAPSED_ROWS:
                self.screen.toggle(app, add, height, "memories",
                             "Show fewer memories" if "memories" in self.screen.expanded else f"Show all {len(memories)} memories")
        else:
            add("Observer view", app.heading_font)
            add("Private motives and accounts are hidden until you choose to show them.", app.small_font, MUTED)
            self.screen.private_toggle_top = height()
            add(f"{'Hide' if self.screen.show_private else 'Show'} private details (observer only)", app.small_font, ACCENT)
            if not self.screen.show_private:
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
                opened = key in self.screen.expanded
                top = height()
                add(f"{'-' if opened else '+'} {assertion_text(sim, fact, claim.affirmed)}", app.small_font)
                add(f"{belief.stance.title()} / {'firsthand' if belief.direct else 'personal assertion' if belief.source_id == npc.id else 'hearsay'}", app.small_font, MUTED)
                self.screen.link(top, height(), lambda key=key: self.screen.expanded.symmetric_difference_update({key}))
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
                        self.screen.link(link, height(), lambda cid=disclosure.conversation_id: self.screen.open_conversation(app, cid))
                gap()

    @staticmethod
    def person_label(app, person):
        return PLAYER_LABEL if person is app.simulation.player else person.name

    def _opinion_grid(self, app, npc, add, gap, height):
        """Living and dead residents' and the python's opinions, as a colored matrix."""
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
        cell = min(GRID_MAX_CELL, (self.screen.detail_rect.width - 40 - GRID_LABEL_WIDTH) // len(people))
        header = app.small_font.get_linesize() + 4
        top = height()
        total = header + cell * len(people)
        self.screen.grid = dict(top=top, cell=cell, header=header, people=people,
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
        dead_names = {self.person_label(app, person) for person in (*sim.actors, sim.player) if not person.alive}
        if is_player:
            # The player holds no opinions of their own; show what each actor thinks of them.
            for other in sim.actors:
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
                                  "theirs": None}
        add("Bars run from -100 (left) to +100 (right). Click a person to read both reasons.", app.small_font, MUTED)
        if is_player or PLAYER_LABEL in rows:
            add("Your own opinions are not recorded and appear as n/a.", app.small_font, MUTED)
        gap()
        if not rows:
            add("No opinions recorded yet.", color=MUTED)

        def order(item):
            person, row = item
            if self.screen.opinion_sort == "Score":
                entry = row["theirs"] if is_player else row["mine"]
                score = entry["opinion"] if entry else None
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
                self.screen.opinion_bars.append((height(), label, entry["opinion"] if entry else None))
                gap()
            if self.screen.selected_opinion == person:
                if row["mine"]:
                    add(f"{first}'s reason: {row['mine']['reason']}", app.small_font)
                if row["theirs"]:
                    add(f"Their reason: {row['theirs']['reason']}", app.small_font)
                if not dead:
                    self.screen.dialogue_backgrounds.append((top, height()))
            if dead:
                self.screen.dead_backgrounds.append((top, height(), self.screen.selected_opinion == person))
            self.screen.link(top, height(), lambda person=person: setattr(
                self.screen, "selected_opinion", None if self.screen.selected_opinion == person else person))
            gap()

    def draw_grid(self, app, surface):
        grid = self.screen.grid
        cell, header, people = grid["cell"], grid["header"], grid["people"]
        x0 = self.screen.detail_rect.x + GRID_LABEL_WIDTH
        y0 = self.screen.detail_rect.top + grid["top"] - self.screen.detail_scroll
        line = app.small_font.get_linesize()
        for column in range(len(people)):
            label = app.small_font.render(str(column + 1), True, MUTED)
            surface.blit(label, label.get_rect(center=(x0 + column * cell + cell // 2, y0 + line // 2)))
        for row, person in enumerate(people):
            y = y0 + header + row * cell
            if y + cell < self.screen.detail_rect.top or y > self.screen.detail_rect.bottom:
                continue
            name = self.person_label(app, person)
            name = name if len(name) <= 18 else name[:17] + "."
            color = INK if person is self.screen.selected_npc else MUTED
            surface.blit(app.small_font.render(f"{row + 1}. {name}", True, color), (self.screen.detail_rect.x + 4, y + (cell - line) // 2))
            for column, value in enumerate(grid["values"][row]):
                rect = pygame.Rect(x0 + column * cell, y, cell - 2, cell - 2)
                if value is None:
                    pygame.draw.rect(surface, MUTED if row == column else GRASS, rect, 0 if row == column else 1)
                    continue
                target = OPINION_POSITIVE_COLOR if value >= 0 else OPINION_NEGATIVE_COLOR
                strength = abs(value) / 100
                fill = tuple(round(PAPER[i] + (target[i] - PAPER[i]) * strength) for i in range(3))
                pygame.draw.rect(surface, fill, rect)
            if person is self.screen.selected_npc:
                pygame.draw.rect(surface, ACCENT, (x0 - 2, y - 1, cell * len(people), cell), 2)
