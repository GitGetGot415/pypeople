"""Killings and deaths: eligibility, the committed killing, and archiving the victim."""

import math
from constants import CONVERSATION_CANCELLED, PROTECTED_POPULATION
from simulation.maps import active_player


class DeathMixin:
    """The Simulation's killing and departure rules, kept apart from its clock and movement."""

    def killing_unavailable(self, attacker, victim):
        """Shared physical restrictions; each killer has their own daily cooldown."""
        from constants import GARDEN_LOCATION, KILLING_COOLDOWN_DAYS, KILLING_DISTANCE, PLAYER_KILLING_DISTANCE, KILLING_INDOOR_REASON, SNAKE_NAME
        if victim is self.snake:
            return f"{SNAKE_NAME} cannot be killed."
        living_victim = (victim is self.player and active_player(self) or
                         self.npc_by_id(victim.id) is victim and victim.alive)
        if (not living_victim or attacker is victim
                or not (self.npc_by_id(attacker.id) is attacker
                        or attacker is self.player and active_player(self))):
            return "Choose a living resident."
        from simulation.autonomy import arrival_greeting_reserved
        if any(arrival_greeting_reserved(person, self) for person in (attacker, victim)):
            return "Judy and the new arrival are protected until their welcome finishes."
        if attacker.location != GARDEN_LOCATION or victim.location != GARDEN_LOCATION:
            return KILLING_INDOOR_REASON
        if self.population <= PROTECTED_POPULATION:
            return f"You cannot kill while there are {PROTECTED_POPULATION} or less people remaining."
        last_kill = self.last_killing_day_for(attacker.id)
        if last_kill is not None and self.day - last_kill < KILLING_COOLDOWN_DAYS:
            if attacker is self.player:
                return "You already killed someone today."
            return "This attacker has already killed someone today. Wait for the next day."
        distance = PLAYER_KILLING_DISTANCE if attacker is self.player else KILLING_DISTANCE
        if math.dist((attacker.x, attacker.y), (victim.x, victim.y)) > distance:
            return "Move closer to the resident first."
        return ""

    def player_killing_unavailable(self, victim):
        if not active_player(self):
            return "The player is unavailable."
        if self.paused:
            return "Resume the simulation first."
        if self.player_interaction and self.player_interaction.incoming and self.player_interaction.stage == "approach":
            return "The resident is approaching. Wait until they reach you."
        if (self.player.journey or self.visit and self.visit.stage == "travel"
                or self.player_interaction and (self.player_interaction.npc_id != victim.id
                                               or self.player_interaction.kind == "door")):
            return "Finish the current journey or interaction first."
        return self.killing_unavailable(self.player, victim)

    def commit_player_killing(self, victim):
        from constants import CONVERSATION_COMPLETED, PLAYER_LABEL
        from dialogue.history import ConversationRecord
        reason = self.player_killing_unavailable(victim)
        if reason:
            raise ValueError(reason)
        record = ConversationRecord((PLAYER_LABEL, victim.name), f"Killed {victim.name}", CONVERSATION_COMPLETED,
                                    origin="player", purpose="killing", participant_ids=[self.player.id, victim.id])
        self.history.append(record)
        return self._finish_killing(self.player, victim, record)

    def player_confrontation_record(self, npc, interaction, grievance, *, approaching=False, food=False):
        from constants import CONVERSATION_COMPLETED, CONVERSATION_WALKING_TO_MEET, PLAYER_LABEL
        from dialogue.history import ConversationRecord
        from simulation.social.relationships import confrontation_topic
        topic = confrontation_topic(npc, grievance)
        if food:
            topic += " Seek an apple now because I cannot survive the next day without eating; a later promise has not supplied food."
        if npc.player_call:
            npc.player_call.kind, npc.player_call.reference = "food_confrontation" if food else "news", grievance
            npc.player_call.key = f"food_confrontation:{self.day}" if food else f"confrontation:{grievance}:{self.day}"
            npc.player_call.reason = topic
        record = ConversationRecord((npc.name, PLAYER_LABEL), topic,
                                    CONVERSATION_WALKING_TO_MEET if approaching else CONVERSATION_COMPLETED,
                                    origin="autonomous", purpose="confrontation",
                                    participant_ids=[npc.id, self.player.id], grievance_id=grievance)
        record.food_confrontation_day = self.day if food else None
        record.intended_subjects = {npc.name: topic, PLAYER_LABEL: "Responding to the resident's accusation."}
        self.history.append(record)
        interaction.record_id = record.id
        return record

    def begin_incoming_conversation(self, npc, interaction):
        """Resolve the opening once on arrival, including an interrupted victim's partner."""
        from constants import CONFRONTATION_OPINION, CONVERSATION_COMPLETED, CONVERSATION_WALKING_TO_MEET
        from simulation.social.relationships import grievance_against, killing_chance
        if (self.paused or self.player_interaction is not interaction or not interaction.incoming
                or interaction.stage != "conversation"
                or not (npc is self.snake or self.npc_by_id(npc.id) is npc)):
            return
        if npc is not self.snake:
            from simulation.player_food import food_arrival
            if food_arrival(self, npc, interaction):
                return
        record = self.record_by_id(interaction.record_id)
        if record:
            if record.purpose != "confrontation" or record.status != CONVERSATION_WALKING_TO_MEET:
                return
            record.status = CONVERSATION_COMPLETED
        else:
            grievance = grievance_against(npc, self.player) if npc is not self.snake else None
            if not grievance:
                return
            record = self.player_confrontation_record(npc, interaction, grievance)
        if npc is self.snake:
            record.outcomes.append("The python confronted the player without physical action.")
            return
        unavailable = self.killing_unavailable(npc, self.player)
        if npc.opinion_score(self.player.id) > CONFRONTATION_OPINION:
            record.outcomes.append("The resident was not hostile enough to attack.")
        elif unavailable:
            record.outcomes.append(f"The resident could not attack: {unavailable}")
        elif npc.rng.random() < killing_chance(npc.opinion_score(self.player.id)):
            record.outcomes.append("The resident attacked immediately on reaching the player.")
            self._finish_killing(npc, self.player, record)
        else:
            record.outcomes.append("The resident's opening attack roll failed.")

    def escalation_choices(self, meeting, opinion=None):
        """opinion: the confronter's view to judge eligibility by, such as the one an exchange is about to commit."""
        from constants import CONFRONTATION_OPINION, GRIEVANCE_FACT_KINDS
        from simulation.social.records import claim_by_id, fact_by_id
        first, target = meeting.initiator, meeting.recipient
        if (meeting.purpose != "confrontation" or self.npc_by_id(first.id) is not first
                or self.npc_by_id(target.id) is not target
                or not meeting.grievance_id):
            return None
        actions = ["leave"]
        if meeting.food_confrontation:
            from simulation.nutrition import starves_next_day
            from simulation.autonomy import food_alternative
            if not starves_next_day(first) or food_alternative(first, self):
                return {"target_id": target.id, "grievance_id": meeting.grievance_id, "actions": actions}
        belief = next((b for b in first.beliefs if b.claim_id == meeting.grievance_id), None)
        claim = claim_by_id(self, meeting.grievance_id)
        fact = fact_by_id(self, claim.fact_id)
        # Frozen information proves the grievance was retained before this meeting.
        retained = (meeting.information_context is None or any(
            b.claim_id == meeting.grievance_id and b.stance == "believed"
            for b in meeting.information_context.beliefs["initiator"]))
        if (belief and belief.stance == "believed" and claim.affirmed and fact.subject_id == target.id
                and fact.kind in GRIEVANCE_FACT_KINDS
                and retained and (first.opinion_score(target.id) if opinion is None else opinion) <= CONFRONTATION_OPINION
                and not self.killing_unavailable(first, target)):
            actions.append("attack")
        return {"target_id": target.id, "grievance_id": meeting.grievance_id, "actions": actions}

    def commit_killing(self, meeting):
        from simulation.locations import available
        decision = meeting.escalation
        attacker, victim = meeting.initiator, meeting.recipient
        choices = self.escalation_choices(meeting)
        valid = (meeting.active and meeting.arrived and meeting.dialogue.dialogue_state == "ready"
                 and meeting.social_effects_applied and decision is not None and decision.action == "attack"
                 and choices and "attack" in choices["actions"]
                 and decision.target_id == victim.id and decision.grievance_id == meeting.grievance_id
                 and bool(decision.evidence.strip()) and any(t.speaker == "recipient" and decision.evidence in t.text for t in meeting.turns)
                 and all(self.reservations.get(n.id, "conversation") is meeting
                         and meeting.accepted_revisions.get(n.id) == n.command_revision
                         and n.alive and available(n) and not n.interaction_blocked
                         and not n.player_call for n in (attacker, victim)))
        meeting.escalation = None
        if not valid:
            meeting.record.outcomes.append("The accepted attack was cancelled because eligibility or ownership changed.")
            return False
        return self._finish_killing(attacker, victim, meeting.record, meeting=meeting)

    def _finish_killing(self, attacker, victim, record, *, meeting=None):
        """Commit one authorised death, preserving the same witnesses and consequences for either killer."""
        if not victim.alive or (victim is not self.player and self.npc_by_id(victim.id) is not victim):
            return False
        from constants import KILLING_WITNESS_PENALTY
        from simulation.maps import observers
        from simulation.social.records import claim_by_id
        from simulation.social.information import add_fact
        conversation = getattr(victim, "conversation", None)
        partner = (conversation.other(victim) if attacker is self.player and conversation
                   and conversation.active and conversation.arrived else None)
        witnesses = observers(self, attacker, victim)
        text = f"{attacker.name} killed {victim.name}."
        event = self.publish("killing", (attacker, victim), text, record)
        fact = add_fact(self, "killing", attacker, victim, event.id, record, text, witnesses,
                        location=victim.location)
        if meeting:
            meeting.outcome_event_id = event.id
        record.outcomes.append(text)
        from dialogue.history import remember
        for witness in witnesses:
            remember(witness, victim if witness is attacker else attacker, event, "killing", text, 0)
        for witness in witnesses:
            if witness in (attacker, victim):
                continue
            belief = next(b for b in witness.beliefs if claim_by_id(self, b.claim_id).fact_id == fact.id)
            belief.assessed = True  # One firsthand consequence; reports cannot repeat it.
            delta = -witness.opinion_score(victim.id) - KILLING_WITNESS_PENALTY
            before = witness.opinion_score(attacker.id)
            reason = f"Witnessed {text} My relationship with {victim.name} informed my reaction."
            if attacker is self.player:
                from simulation.player import update_player_opinion
                update_player_opinion(witness, delta, reason, scaled=False)
            else:
                witness.remember_opinion(attacker.id, max(-100, min(100, witness.opinion_score(attacker.id) + delta)), reason)
            from dialogue.history import describe_opinion_change
            change = describe_opinion_change(witness.name, attacker.name, before, witness.opinion_score(attacker.id), reason)
            record.outcomes.append(change)
            self.publish("reputation_changed", (witness, attacker), change, record)
        if victim is self.player:
            self.depart_player()
            self._take_killed_apples(attacker, victim, record, witnesses)
            return True
        departed = self.depart_resident(victim, event, record, attacker, meeting=meeting)
        if departed:
            self._take_killed_apples(attacker, victim, record, witnesses)
        if departed and partner is not None:
            self.confront_interrupted_partner(partner, fact)
        return departed

    def _take_killed_apples(self, attacker, victim, record, witnesses):
        """A killing can seize carried fruit, without consent or gift/promise rewards."""
        quantity = victim.inventory.get("fruit", 0)
        if not quantity:
            return
        victim.inventory["fruit"] = 0
        attacker.inventory["fruit"] = attacker.inventory.get("fruit", 0) + quantity
        record.seized_apples = quantity
        text = f"{attacker.name} took {quantity} apple{'s' if quantity != 1 else ''} from {victim.name} after killing them."
        record.outcomes.append(text)
        event = self.publish("apples_taken", (attacker, victim), text, record)
        from simulation.social.information import add_fact
        from dialogue.history import remember
        witnesses = [person for person in witnesses if person is not victim and person.alive]
        add_fact(self, "apples_taken", attacker, victim, event.id, record, text, witnesses,
                 quantity=quantity, location=victim.location)
        for witness in witnesses:
            remember(witness, victim if witness is attacker else attacker, event, "apples_taken", text, 0)

    def confront_interrupted_partner(self, partner, fact):
        from simulation.player import start_approach
        from simulation.social.records import claim_by_id
        if not active_player(self) or not partner.alive or partner.location != self.player.location:
            return
        belief = next(b for b in partner.beliefs if b.direct and claim_by_id(self, b.claim_id).fact_id == fact.id)
        interaction = start_approach(self, partner, incoming=True)
        source = self.player_confrontation_record(partner, interaction, belief.claim_id, approaching=True)
        partner.reason = "Confronting the player about killing my conversation partner."
        self.publish("player_call", (partner,), f"{partner.name} is confronting you about killing their conversation partner.", source)

    def cancel_death_promises(self, target, reason):
        for promise in self.promises:
            if promise.status == "pending" and target.id in (promise.promisor_id, promise.recipient_id):
                promise.status, promise.resolved_day, promise.outcome = "cancelled", self.day, reason
                cancellation = self.publish("promise_cancelled", (self.participant_by_id(promise.promisor_id), self.participant_by_id(promise.recipient_id)), reason,
                                            self.record_by_id(promise.conversation_id))
                promise.resolution_event_id = promise.consequence_event_id = cancellation.id

    def depart_player(self):
        from copy import deepcopy
        from simulation.player import end_interaction, clear_call
        from simulation.apartments import finish_visit
        self.player.alive, self.mode = False, "spectator"
        self.bodies.append({"appearance": deepcopy(self.player.appearance), "x": self.player.x, "y": self.player.y,
                            "location": self.player.location, "facing": self.player.facing, "elapsed": 0.0})
        reason = "Cancelled because the player died; this is not a deliberate betrayal."
        self.cancel_death_promises(self.player, reason)
        end_interaction(self, reason)
        finish_visit(self)
        self.player.journey = None
        self.player_acknowledgments.clear()
        for offer in self.player_invitations:
            if offer.status == "offered":
                offer.status = "withdrawn"
        for npc in self.npcs:
            if npc.player_food_request:
                npc.player_food_request.pursuing = False
            if npc.gift and self.player in (npc.gift.initiator, npc.gift.recipient):
                npc.gift.finish(reason)
            clear_call(npc)
        retained = []
        for action in self.pending_actions:
            if action.target_id != self.player.id:
                retained.append(action)
                continue
            npc = self.npc_by_id(action.npc_id)
            if npc:
                npc.player_reply.release()
                npc.command_revision += 1
            source = self.record_by_id(action.record)
            if source:
                source.status, source.error = CONVERSATION_CANCELLED, reason
                source.outcomes.append(reason)
        self.pending_actions[:] = retained
        self.roster_revision += 1

    def depart_resident(self, target, event, record, attacker, *, meeting=None):
        """Archive a committed victim and release unfinished work through its owners."""
        if (self.npc_by_id(target.id) is not target or not target.alive or event.kind != "killing"
                or event.history != record.id or record.purpose not in ("confrontation", "killing")
                or not any(f.kind == "killing" and f.source_id == event.id and f.subject_id == attacker.id
                           and f.object_id == target.id for f in self.facts)):
            return False
        return self._archive_dead_resident(target, event,
            "killed by player" if attacker is self.player else "killed by resident",
            record=record, attacker=attacker, meeting=meeting)

    def starve_resident(self, target):
        """Record natural death without inventing an attacker or a witnessed cause."""
        if self.npc_by_id(target.id) is not target or not target.alive or target.social > 0:
            return False
        from simulation.maps import observers
        from simulation.social.information import add_fact
        from dialogue.history import remember
        witnesses = [person for person in observers(self, target) if person is not target]
        event = self.publish("starvation", (target,), f"{target.name} died of starvation.")
        visible = f"{target.name} collapsed and died. The cause was not established by seeing the collapse."
        add_fact(self, "death", target, None, event.id, None, visible, witnesses, location=target.location)
        for witness in witnesses:
            remember(witness, target, event, "death", visible, 0)
        return self._archive_dead_resident(target, event, "starvation")

    def _archive_dead_resident(self, target, event, cause, *, record=None, attacker=None, meeting=None):
        from copy import deepcopy
        from simulation.social.goals import resolve_goal
        target.departure = {"day": self.day, "cause": cause, "event_id": event.id,
                            "attacker_id": attacker.id if attacker else None,
                            "conversation_id": record.id if record else None}
        self.bodies.append({"appearance": deepcopy(target.character), "x": target.x, "y": target.y,
                            "location": target.location, "facing": target.facing, "elapsed": 0.0})
        # The source is accepted and complete; mark death before any release can re-enter.
        if meeting:
            meeting.finish(self.tree)
            meeting.outcome_event_id = event.id
        reason = f"Cancelled because {target.name} died; this is not a deliberate betrayal."
        self.cancel_death_promises(target, reason)
        for person in self.npcs:
            goal = person.goal
            if goal and goal.status == "active" and (person is target or target.id in (goal.target_id, goal.third_npc_id)):
                resolve_goal(person, self, "cancelled", reason, record)
        self.interrupt_resident(target, reason, mode="departure")
        for invitation in self.player_invitations:
            if invitation.speaker_id == target.id and invitation.status == "offered":
                invitation.status = "withdrawn"
        target.home_remaining = 0.0
        if target.player_food_request:
            target.player_food_request.pursuing = False
        self.npcs[:] = [p for p in self.npcs if p is not target]
        self.archived.append(target)
        self.roster_revision += 1
        self.introductions[:] = [i for i in self.introductions if i != target.id]
        self.social_rotation %= max(1, len(self.npcs))
        return True
