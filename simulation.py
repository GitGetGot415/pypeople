"""A small world with player-controlled day phases."""

from dataclasses import dataclass
import math

from constants import (
    APPLE_ADVANCE_SECONDS, COMPANY_SOCIAL_RECOVERY, CROWD_LIMIT, DECISION_SECONDS,
    FAILURE_COOLDOWN, HEARING_RADIUS, MAX_SIMULATION_STEP_SECONDS, PROTECTED_POPULATION,
    RECOVERY_SECONDS, SATISFACTION_PHASE_LOSS, SOCIAL_RADIUS, SNAKE_NAME,
    STARTING_DAY, STARTING_PAUSED, STARTING_PHASE, WITNESS_RADIUS,
)


@dataclass
class PendingAction:
    npc_id: str
    action: str
    revision: int
    target_id: str | None = None
    topic: str = ""
    record: int | None = None
    item: str = ""
    quantity: int = 0


class Simulation:
    def __init__(self, npcs, tree=None, *, assign_goals=True, npc_trait_mode="normal"):
        from npc import Tree
        from snake import Snake

        self.npcs = npcs
        self.npc_trait_mode = npc_trait_mode
        self.tree = tree if tree is not None else Tree(fruit=len(npcs))
        self.snake = Snake()
        self.snake.world = self
        self.archived = []
        self.arrival_error = ""
        self.pending_arrival_days = []
        self.processed_arrival_days = []
        self.introductions = []
        self.roster_revision = 0
        self.day = STARTING_DAY
        self.phase = STARTING_PHASE
        self.paused = STARTING_PAUSED
        self.pending_actions = []
        self.history = []
        self.events = []
        self.promises = []
        self.facts = []
        self.claims = []
        self.disclosures = []
        self.social_rotation = 0
        for npc in self.npcs:
            npc.world = self
        if assign_goals:
            from purgatory import remember_description
            from social import assign_goal
            for npc in self.npcs:
                remember_description(npc)
            for npc in self.npcs:
                assign_goal(npc, self)
        from social import refresh_concern
        for npc in self.npcs:
            if npc.current_concern is None:
                refresh_concern(npc)
                npc.decision_remaining = 0.0

    @property
    def phase_index(self):
        from social import phase_index
        return phase_index(self.day, self.phase)

    @property
    def phase_label(self):
        return f"Day {self.day} / {self.phase.title()}"

    def next_phase(self):
        from social import advance_social_phase
        if self.paused:
            return
        if self.phase == "day":
            self.phase = "evening"
        elif self.phase == "evening":
            self.phase = "night"
        else:
            self.phase = "day"
            self.day += 1
            self.snake.eaten_today = 0
            if self.day not in self.pending_arrival_days and self.day not in self.processed_arrival_days:
                self.pending_arrival_days.append(self.day)
        self.snake.pardons.clear()
        for npc in self.npcs:
            npc.social = max(0.0, npc.social - SATISFACTION_PHASE_LOSS)
        self.tree.fruit = len(self.npcs)
        advance_social_phase(self)
        self.publish("phase", [], f"Time changed to {self.phase}.")

    def publish(self, kind, participants, text, record=None):
        from conversation_history import WorldEvent

        reference = next((i for i, entry in enumerate(self.history) if entry is record), None)
        event = WorldEvent(self.day, self.phase, kind, [npc.id for npc in participants], text, reference)
        self.events.append(event)
        from social import record_event_fact
        record_event_fact(self, event, record)
        del self.events[:-200]
        return event

    def start_conversation(self, initiator, recipient, topic, *, origin="player", purpose="chat", spacing=0,
                           opening_kind=None, focus_claim_id=None, goal_id=None):
        from npc import Conversation

        if any(self.npc_by_id(p.id) is not p for p in (initiator, recipient)):
            raise ValueError("Only living residents can begin this conversation.")
        interaction = Conversation(initiator, recipient, topic, self.tree, spacing, origin=origin, purpose=purpose,
                                   opening_kind=opening_kind, focus_claim_id=focus_claim_id, goal_id=goal_id)
        self.history.append(interaction.record)
        return interaction

    def npc_by_id(self, npc_id):
        return next((npc for npc in self.npcs if npc.id == npc_id), None)

    def advance(self, dt, *, active=True, action_handler=None, viewed_conversation=None):
        if not active or self.paused:
            return
        if action_handler is not None:
            actions, self.pending_actions = self.pending_actions, []
            for action in actions:
                npc = self.npc_by_id(action.npc_id)
                if npc is not None and npc.selected and npc.end_conversation:
                    # Keep the closing reply visible until the player dismisses it.
                    self.pending_actions.append(action)
                    continue
                action_handler(action)
        # Ignore long window stalls; there is no elapsed world time to catch up.
        self.step(min(MAX_SIMULATION_STEP_SECONDS, max(0.0, dt)), viewed_conversation=viewed_conversation)

    def company_available(self, npc):
        from npc import Behavior

        return (npc.departure is None
                and npc.interaction is None and not npc.selected and not npc.request_pending
                and npc.owner == "autonomous" and not npc.resume_pending
                and not any(npc.id in (action.npc_id, action.target_id) for action in self.pending_actions)
                and npc.behavior in (Behavior.WANDER, Behavior.WAIT, Behavior.COMPANY))

    def step(self, dt, *, viewed_conversation=None):
        from autonomy import lone_introduction, reconsider, social_pair
        from social import refresh_concern
        from npc import Behavior

        self.tree.update(dt)
        self.deliver_arrivals()
        self.snake.update(dt)
        encounter = self.snake.encounter
        if encounter is not None:
            encounter.update(dt, reading=encounter is viewed_conversation)
        for npc in self.npcs:
            npc.update_cooldowns(dt)
            if npc.pending_social_opinions and npc.decision_remaining == 0:
                from social import apply_opinion_effects
                if not apply_opinion_effects(npc):
                    npc.decision_remaining = DECISION_SECONDS
        # Finish interactions (including atomic request-to-gift handoffs) before decisions.
        interactions = {id(npc.conversation or npc.gift): npc.conversation or npc.gift
                        for npc in self.npcs if npc.conversation or npc.gift}
        for interaction in interactions.values():
            if interaction is viewed_conversation:
                interaction.update(dt, reading=True)
            else:
                interaction.update(dt)
        # Check necessary work for every actor before an idle initiator can reserve them.
        for npc in self.npcs:
            if npc.resume_pending and npc.interaction is None:
                npc.resume_activity(self)
            reconsider(npc, self, pressing_only=True)
        due = {npc.id for npc in self.npcs if npc.decision_remaining == 0 and self.company_available(npc)}
        for npc in self.npcs:
            if npc.id in due:
                refresh_concern(npc)
        while due:
            choice = social_pair(self, due=due)
            if choice is None:
                break
            initiator, recipient, purpose, topic, reason, opening, focus, goal_id = choice
            self.start_conversation(initiator, recipient, topic, origin="autonomous", purpose=purpose,
                                    opening_kind=opening, focus_claim_id=focus, goal_id=goal_id)
            initiator.reason = reason
            recipient.reason = (f"Answering {initiator.name}'s request for an apple" if purpose == "food_request"
                                else f"Chatting with {initiator.name}")
            due.difference_update((initiator.id, recipient.id))
            self.social_rotation = (self.npcs.index(initiator) + 1) % len(self.npcs)
        for npc in self.npcs:
            if npc.id in due:
                lone_introduction(npc, self)
            reconsider(npc, self)
            npc.update(dt, self.tree)
        # Each person benefits once, regardless of how many neighbors are present.
        socializing = []
        for npc in self.npcs:
            if not self.company_available(npc):
                continue
            if any(other is not npc and self.company_available(other)
                   and (npc.behavior == Behavior.COMPANY or other.behavior == Behavior.COMPANY)
                   and math.hypot(npc.x - other.x, npc.y - other.y) <= SOCIAL_RADIUS
                   for other in self.npcs):
                socializing.append(npc)
        for npc in socializing:
            npc.social = min(100.0, npc.social + COMPANY_SOCIAL_RECOVERY * dt)

    @property
    def residents(self):
        return [*self.npcs, *self.archived]

    @property
    def actors(self):
        return [*self.residents, self.snake]

    def actor_by_id(self, actor_id):
        return next((actor for actor in self.actors if actor.id == actor_id), None)

    @property
    def available_fruit(self):
        encounter = self.snake.encounter
        return max(0, self.tree.fruit - int(bool(encounter and encounter.active and encounter.reserved)))

    def listeners(self, actor, radius=None):
        radius = HEARING_RADIUS if radius is None else radius
        return [p for p in self.npcs if math.dist((p.x, p.y), (actor.x, actor.y)) <= radius]

    def observed_event(self, kind, subject, target, text, record=None, *, witnesses=None):
        from social import add_fact
        from conversation_history import remember
        if witnesses is None:
            witnesses = [subject, target, *self.listeners(subject, WITNESS_RADIUS)]
            if math.dist((self.snake.x, self.snake.y), (subject.x, subject.y)) <= WITNESS_RADIUS:
                witnesses.append(self.snake)
        witnesses = list({p.id: p for p in witnesses}.values())
        event = self.publish(kind, (subject, target), text, record)
        fact = add_fact(self, kind, subject, target, event.id, record, text, witnesses)
        for actor in witnesses:
            if actor is self.snake:
                continue
            partner = target if actor is subject else subject
            remember(actor, partner, event, kind, text, -4 if kind == "eating" else 0)
        return event, fact

    def seek_bargain(self, npc):
        from snake import SnakeEncounter
        if self.npc_by_id(npc.id) is not npc or math.dist((npc.x, npc.y), self.tree.interaction_point) > 8:
            return None
        collecting = npc.id in self.snake.advances
        if ((not collecting and (npc.inventory_space < 1 or self.available_fruit < 1)) or self.snake.encounter
                or self.snake.selected or self.snake.request_pending
                or self.snake.recovery_remaining or self.snake.id in npc.cooldowns):
            npc.cooldowns[self.tree.id] = FAILURE_COOLDOWN
            npc.reason = "Waiting for an available apple bargain"
            return None
        try:
            contested = npc.id in self.snake.offenses or self.snake.standing.get(npc.id, 0) < -8
            if (npc.id not in self.snake.pardons and self.eating_permitted(npc)
                    and (contested or len(self.npcs) > CROWD_LIMIT)):
                reason = f"{SNAKE_NAME} disputes this visitor's right to take an apple under its own standards."
                encounter = SnakeEncounter(self, npc, "judgment", reason)
                self.snake.judged_phase = self.phase_index
                return encounter
            reason = ("Collecting the offering for an apple already given upfront" if collecting else
                      "An apple upfront, with information or public support owed later")
            return SnakeEncounter(self, npc, "bargain", reason)
        except ValueError:
            return None

    def intercept_harvest(self, npc):
        if math.dist((npc.x, npc.y), self.tree.interaction_point) > 8:
            return None
        event, fact = self.observed_event("defiance", npc, self.snake,
            f"{npc.name} attempted to take guarded fruit without an offering; no apple was taken.")
        self.snake.offenses[npc.id] = fact.id
        self.snake.standing[npc.id] = max(-100, self.snake.standing.get(npc.id, 0) - 6)
        return self.seek_bargain(npc)

    def eating_permitted(self, target=None, *, encounter=None, reachable=False):
        if len(self.npcs) <= PROTECTED_POPULATION or self.snake.recovery_remaining > 0:
            return False
        if self.snake.eaten_today >= 1 and len(self.npcs) <= CROWD_LIMIT:
            return False
        if target is not None:
            if self.npc_by_id(target.id) is not target:
                return False
            if reachable and math.dist((self.snake.x, self.snake.y), (target.x, target.y)) > 32:
                return False
        if encounter is not None:
            return (encounter.active and encounter is self.snake.encounter and encounter.committed
                    and encounter.outcome == "eat" and encounter.stage == "eating")
        return True

    def commit_snake_response(self, encounter, response):
        from social import claim_by_id, learn_account
        npc, snake = encounter.recipient, self.snake
        decision = response.decision
        if not encounter.accepts_response or encounter.committed or not encounter.arrived:
            raise ValueError("This encounter is no longer available.")
        if decision not in encounter.request_context["permitted_decisions"]:
            raise ValueError("That decision is not available for this encounter.")
        advance = snake.advances.get(npc.id)
        expected = encounter.request_context["advance"]
        if bool(advance) != bool(expected) or advance and any(
                advance[key] != expected[key] for key in ("offering", "account_id", "conversation_id")):
            raise ValueError("The resident's apple agreement changed. Retry the encounter.")
        giving_apple = decision in ("advance_secret", "advance_endorsement")
        trust_before = npc.attitudes["trust"]
        audience = [p for p in self.listeners(npc) if p is not npc]
        if giving_apple:
            if advance:
                raise ValueError("This resident already owes an offering for an upfront apple.")
            if not encounter.reserved or self.tree.fruit < 1 or npc.inventory_space < 1:
                raise ValueError("The reserved apple or inventory space is no longer available. Retry the bargain.")
            if (math.dist((npc.x, npc.y), self.tree.interaction_point) > 8
                    or math.dist((npc.x, npc.y), (snake.x, snake.y)) > HEARING_RADIUS):
                raise ValueError("The resident must be at the guarded tree to receive the apple.")
            if decision == "advance_endorsement" and len(self.npcs) < 2:
                raise ValueError("There is no other resident to hear a later endorsement.")
        elif decision in ("secret", "endorsement"):
            if (not advance or advance["offering"] != decision
                    or decision == "secret" and response.account_id != advance["account_id"]):
                raise ValueError("The offering must match the existing upfront apple agreement.")
            if decision == "endorsement" and (not audience or not set(encounter.request_context["audience_ids"]) & {p.id for p in audience}):
                raise ValueError("No intended listener can hear the endorsement now. Retry when someone is nearby.")
        elif decision == "eat" and not self.eating_permitted(npc):
            raise ValueError("Eating permission changed. Retry the judgment.")
        if decision in ("secret", "advance_secret"):
            allowed = {a["claim_id"] for a in encounter.request_context["offering_accounts"]}
            owned = {b.claim_id for b in npc.beliefs if b.stance != "disbelieved"}
            if response.account_id not in allowed & owned:
                raise ValueError("That account is no longer owned or available to share.")
            claim = claim_by_id(self, response.account_id)
            if giving_apple and any((claim_by_id(self, b.claim_id).fact_id, claim_by_id(self, b.claim_id).affirmed)
                                   == (claim.fact_id, claim.affirmed) for b in snake.beliefs):
                raise ValueError(f"{SNAKE_NAME} already knows this account.")
        if response.snake_account_id:
            if response.snake_account_id not in {b.claim_id for b in snake.beliefs}:
                raise ValueError(f"{SNAKE_NAME} no longer knows that account.")
        # All rejection checks precede this single simulation-thread commit.
        encounter.committed, encounter.outcome = True, decision
        encounter.reserved = False
        if giving_apple:
            self.tree.fruit -= 1
            npc.inventory["fruit"] = npc.inventory.get("fruit", 0) + 1
            offering = "secret" if decision == "advance_secret" else "endorsement"
            snake.advances[npc.id] = {"offering": offering, "account_id": response.account_id,
                                      "conversation_id": encounter.record.id, "remaining": APPLE_ADVANCE_SECONDS}
            text = (f"{SNAKE_NAME} gave {npc.name} one apple upfront. {npc.name} agreed to provide "
                    f"{'private information' if offering == 'secret' else 'a public endorsement'} later; the offering is still owed.")
            event, _ = self.observed_event("bargain", snake, npc, text, encounter.record)
            encounter.outcome_event_id = event.id
        elif decision in ("secret", "endorsement"):
            if decision == "secret":
                learn_account(self, npc, snake, response.account_id, encounter.record, 1, "uncertain", private=True)
            else:
                text = f"{npc.name} publicly endorsed {SNAKE_NAME} after receiving an apple upfront; the endorsement is their claim."
                self.observed_event("endorsement", npc, snake, text, encounter.record, witnesses=[npc, snake, *audience])
            snake.advances.pop(npc.id)
            text = f"{npc.name} fulfilled the offering owed for {SNAKE_NAME}'s earlier upfront apple. No additional apple was transferred."
            event, _ = self.observed_event("bargain", snake, npc,
                                           text, encounter.record)
            next(r for r in self.history if r.id == advance["conversation_id"]).outcomes.append(text)
            encounter.outcome_event_id = event.id
            snake.standing[npc.id] = min(100, snake.standing.get(npc.id, 0) + 5)
        elif decision in ("refuse", "negotiate", "challenge"):
            action = {"challenge": f"challenged {SNAKE_NAME}'s control of the tree",
                      "refuse": "declined the current bargain", "negotiate": f"asked to negotiate {SNAKE_NAME}'s price"}[decision]
            broken = advance is not None and decision in ("refuse", "challenge")
            if broken:
                offering = "private recollection" if advance["offering"] == "secret" else "public endorsement"
                text = f"{npc.name} refused the agreed {offering} after receiving an apple upfront from {SNAKE_NAME}. The offering remained unpaid."
                snake.advances.pop(npc.id)
                next(r for r in self.history if r.id == advance["conversation_id"]).outcomes.append(text)
            elif advance:
                text = f"{npc.name}'s offering for an earlier upfront apple is waiting for a feasible opportunity; no additional apple was given."
                advance["remaining"] = APPLE_ADVANCE_SECONDS
            else:
                text = f"{npc.name} {action}; no payment or apple changed hands."
            event, fact = self.observed_event("defiance" if broken or decision == "challenge" else "bargain", npc, snake, text, encounter.record)
            encounter.outcome_event_id = event.id
            if broken or decision == "challenge":
                snake.offenses[npc.id] = fact.id
                snake.standing[npc.id] = max(-100, snake.standing.get(npc.id, 0) - (12 if broken else 8))
                if npc.id in snake.pardons:
                    snake.pardons.remove(npc.id)
        if encounter.purpose == "judgment":
            event, _ = self.observed_event("judgment", snake, npc,
                f"{SNAKE_NAME} accused {npc.name}: {encounter.accusation} The judgment was {'condemnation' if decision == 'eat' else 'a pardon or accepted remedy'}.", encounter.record)
            encounter.outcome_event_id = event.id
            if decision != "eat":
                if npc.id not in snake.pardons:
                    snake.pardons.append(npc.id)
                snake.offenses.pop(npc.id, None)
                if decision == "spare" and advance:
                    snake.advances.pop(npc.id)
                    text = f"{SNAKE_NAME} waived {npc.name}'s outstanding offering for the earlier upfront apple."
                    next(r for r in self.history if r.id == advance["conversation_id"]).outcomes.append(text)
                self.observed_event("pardon", snake, npc, f"{SNAKE_NAME} spared {npc.name} for this phase.", encounter.record)
        # A real encounter supports an attitude change, independently of compliance.
        if response.attitudes is not None:
            updated = response.attitudes.model_dump()
            updated["trust"] = max(npc.attitudes["trust"] - 6, min(npc.attitudes["trust"] + 6, updated["trust"]))
            for key in ("belief_reason", "trust_reason", "authority_reason"):
                updated[key] = (f"After the {encounter.purpose} on day {self.day}: " + updated[key])[:400]
            npc.attitudes = updated
        if giving_apple:
            npc.attitudes["trust"] = min(100, max(npc.attitudes["trust"], trust_before + 4))
            npc.attitudes["trust_reason"] = f"Received a real apple upfront from {SNAKE_NAME} on day {self.day}, before providing the agreed offering."
        if response.snake_account_id:
            learn_account(self, snake, npc, response.snake_account_id, encounter.record, 2, response.account_stance)
        if encounter.request_context["first_meeting"]:
            self.observed_event("snake_claim", snake, npc,
                f"{npc.name} heard {SNAKE_NAME} present claims about authority and the garden; those claims remain unverified.", encounter.record,
                witnesses=[snake, npc, *self.listeners(snake)])
        encounter.record.outcomes.append(f"Decision: {decision}. " + ("One apple transferred upfront; the offering is still owed." if giving_apple else "No additional apple transferred."))

    def interrupt_resident(self, target, reason, *, departing=False):
        """Cancel only unfinished work; preserve every committed handoff and record."""
        from npc import Behavior
        interaction = target.conversation or target.gift
        if interaction:
            interaction.active = False
            interaction.record.status = "Cancelled"
            interaction.record.error = reason
            interaction.record.outcomes.append(reason)
            if hasattr(interaction, "pending_opinions"):
                interaction.pending_opinions.clear()
            for person, saved in zip((interaction.initiator, interaction.recipient), interaction.saved_states):
                person.conversation = person.gift = None
                other = interaction.recipient if person is interaction.initiator else interaction.initiator
                person.restore_after_interaction(saved, other.id)
                person.pause_after_conversation(bool(interaction.record.turns))
                if not interaction.record.turns:
                    from social import dialogue_failure_cooldown
                    person.cooldowns[other.id] = dialogue_failure_cooldown(person, other)
                person.command_revision += 1
        encounter = target.snake_encounter
        if encounter:
            encounter.finish(error=reason)
        self.tree.release(target)
        for record in self.history:
            if target.name in record.participants and "You" in record.participants and record.status in ("Waiting for reply", "Reply pending when saved"):
                record.status, record.error = "Cancelled", reason
                record.outcomes.append(reason)
        if departing:
            target.pending_social_opinions.clear()
        # An unrelated interaction can have suspended a task aimed at this resident.
        interactions = {id(p.conversation or p.gift): p.conversation or p.gift
                        for p in self.npcs if p.conversation or p.gift}
        states = [state for interaction in interactions.values() for state in interaction.saved_states]
        if self.snake.encounter is not None:
            states.append(self.snake.encounter.saved_state)
        for state in states:
            state["cooldowns"].pop(target.id, None)
            if state["target_id"] == target.id:
                state.update(behavior=Behavior.WAIT, owner="autonomous", target_id=None,
                             destination=None, activity_elapsed=0.0, resume_pending=False,
                             reason=reason, decision_remaining=0.0)
        remaining = []
        for action in self.pending_actions:
            if target.id in (action.npc_id, action.target_id):
                owner = self.npc_by_id(action.npc_id)
                if owner:
                    owner.request_pending = False
                    owner.command_revision += 1
                if action.record is not None:
                    record = self.history[action.record]
                    record.status, record.error = "Cancelled", reason
                    record.outcomes.append(reason)
            else:
                remaining.append(action)
        self.pending_actions[:] = remaining
        for person in self.npcs:
            if person.target_id == target.id:
                person.finish_activity(self, failed=True)
            person.cooldowns.pop(target.id, None)
            if person is target:
                person.command_revision += 1
                person.request_pending = False
                person.selected = False
                person.destination = person.target_id = None
                person.behavior = Behavior.WAIT
                person.owner = "autonomous"
                person.activity_elapsed = 0.0
                person.resume_pending = False

    def depart_resident(self, target, encounter):
        from social import resolve_goal
        if target.departure or not self.eating_permitted(target, encounter=encounter, reachable=True):
            return False
        # Mark before cleanup so a second call cannot remove or count this actor twice.
        target.departure = {"day": self.day, "phase": self.phase, "cause": "eaten by the snake", "event_id": ""}
        reason = f"Cancelled because {target.name} was eaten; this is not a deliberate betrayal."
        # The eating encounter is the preserved source, not an abandoned bargain.
        target.snake_encounter = None
        for promise in self.promises:
            if promise.status == "pending" and target.id in (promise.promisor_id, promise.recipient_id):
                promise.status, promise.resolved_phase, promise.outcome = "cancelled", self.phase_index, reason
                event = self.publish("promise_cancelled", (self.actor_by_id(promise.promisor_id), self.actor_by_id(promise.recipient_id)), reason,
                                     next(r for r in self.history if r.id == promise.conversation_id))
                promise.resolution_event_id = promise.consequence_event_id = event.id
        for person in self.npcs:
            goal = person.goal
            if goal and goal.status == "active" and (person is target or target.id in (goal.target_id, goal.third_npc_id)):
                resolve_goal(person, self, "cancelled", reason, encounter.record)
        self.interrupt_resident(target, reason, departing=True)
        self.npcs[:] = [p for p in self.npcs if p is not target]
        self.archived.append(target)
        self.roster_revision += 1
        self.introductions[:] = [i for i in self.introductions if i != target.id]
        self.social_rotation %= max(1, len(self.npcs))
        event, _ = self.observed_event("eating", self.snake, target,
            f"{SNAKE_NAME} ate {target.name}. {SNAKE_NAME} claims this sends a soul to hell; that destination remains unverified.", encounter.record)
        target.departure["event_id"] = event.id
        for witness in self.listeners(self.snake, 220):
            friend = witness.opinions.get(target.name, {}).get("opinion", 0)
            adjustment = -6 if friend > 0 or witness.attitudes["authority"] == "rejected" else 2 if witness.attitudes["belief"] == "believed" else -3
            witness.attitudes["trust"] = max(-100, min(100, witness.attitudes["trust"] + adjustment))
            witness.attitudes["trust_reason"] = f"Witnessed {SNAKE_NAME} eat {target.name} on day {self.day}."
        self.snake.eaten_today += 1
        self.snake.recovery_remaining = RECOVERY_SECONDS
        self.snake.offenses.pop(target.id, None)
        advance = self.snake.advances.pop(target.id, None)
        if advance:
            next(r for r in self.history if r.id == advance["conversation_id"]).outcomes.append(
                f"The outstanding offering was cancelled because {target.name} was eaten.")
        self.snake.pardons[:] = [i for i in self.snake.pardons if i != target.id]
        if self.snake.retry_request and self.snake.retry_request["resident_id"] == target.id:
            self.snake.retry_request = None
        encounter.outcome_event_id = event.id
        encounter.finish()
        encounter.record.status = "Eaten; resident archived"
        encounter.record.outcomes.append(f"{target.name} departed on day {self.day}. Their destination is unverified.")
        return True

    def deliver_arrivals(self):
        from character_data import random_character
        from npc import NPC, assign_npc_names
        from purgatory import remember_description
        from social import assign_goal
        if not self.pending_arrival_days or self.arrival_error:
            return None
        day = self.pending_arrival_days[0]
        if day in self.processed_arrival_days:
            self.pending_arrival_days.pop(0)
            return None
        try:
            character = random_character(trait_mode=self.npc_trait_mode)
        except (OSError, ValueError) as exc:
            self.arrival_error = f"Could not create the next arrival: {exc}"
            return None
        npc = NPC(character=character, arrival_day=day)
        npc.sprite_size = self.npcs[0].sprite_size if self.npcs else (40, 70)
        assign_npc_names([npc], used_names=(actor.name for actor in self.actors))
        left, right, top, bottom = npc.movement_bounds
        positions = [(x, y) for y in range(math.ceil(top), math.floor(bottom) + 1, 36)
                     for x in range(math.ceil(left), math.floor(right) + 1, 48)
                     if math.dist((x, y), (self.tree.x, self.tree.y)) > 160]
        npc.x, npc.y = max(positions, key=lambda point: min((math.dist(point, (p.x, p.y)) for p in self.npcs), default=400))
        npc.world = self
        remember_description(npc)
        self.npcs.append(npc)
        self.processed_arrival_days.append(day)
        self.pending_arrival_days.pop(0)
        self.roster_revision += 1
        assign_goal(npc, self)
        from social import refresh_concern
        refresh_concern(npc)
        npc.decision_remaining = 0.0
        observers = [npc, *self.listeners(npc, 220)]
        if math.dist((npc.x, npc.y), (self.snake.x, self.snake.y)) <= 220:
            observers.append(self.snake)
        self.observed_event("arrival", npc, self.snake, f"{npc.name} arrived for dawn on day {day}, remembering a former life.", witnesses=observers)
        return npc
