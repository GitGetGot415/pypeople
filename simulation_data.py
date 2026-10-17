"""Self-contained simulation saves. No character file paths are loaded."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Literal
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from character_data import normalize_character, normalize_opinions
from conversation_history import ConversationRecord, Memory, WorldEvent
from llm import (AcceptedProposal, ConversationPart, ConversationTurn, DisclosureChoice,
                 EmotionReaction, LeverageUse, ThirdPartyOpinion)
from npc import ACTIVITY_FIELDS, Behavior, Conversation, Gift, NPC, Tree, WORLD_HEIGHT, WORLD_WIDTH
from simulation import DIALOGUE_WAIT_SECONDS, EAT_SECONDS, PendingAction, SHAKE_SECONDS, Simulation
from snake import SNAKE_FRAME_SECONDS, SNAKE_MARGIN, Snake
from social import (Belief, Claim, Disclosure, Fact, FactKind, Goal, GoalKind, OpinionEffect, Promise, Proposal,
                    Stance, information_from_dict, phase_index, validate_agreements, validate_information)

SAVE_DIRECTORY = Path(__file__).resolve().parent / "saves"


class SaveModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ActivityState(SaveModel):
    behavior: Behavior
    destination: tuple[float, float] | None
    pause_remaining: float = Field(ge=0)
    facing: int = Field(ge=0, le=7)
    owner: Literal["autonomous", "player_task", "player_hold"]
    target_id: str | None
    activity_elapsed: float = Field(ge=0)
    reason: str
    decision_remaining: float = Field(ge=0)
    cooldowns: dict[str, float]
    resume_pending: bool

    @model_validator(mode="after")
    def validate_activity(self):
        if any(value < 0 for value in self.cooldowns.values()):
            raise ValueError("Invalid cooldown duration.")
        if self.destination is not None and not (0 <= self.destination[0] <= WORLD_WIDTH and 0 <= self.destination[1] <= WORLD_HEIGHT):
            raise ValueError("Invalid destination.")
        if (self.owner == "player_hold") != (self.behavior == Behavior.STOP):
            raise ValueError("A player hold must be stopped.")
        player_behaviors = (Behavior.GO_TO_TREE, Behavior.SHAKE_TREE, Behavior.SHAKING)
        if self.behavior != Behavior.EAT and (self.owner == "player_task") != (self.behavior in player_behaviors):
            raise ValueError("Invalid player task.")
        if self.behavior == Behavior.EAT and (self.destination is not None or self.activity_elapsed >= EAT_SECONDS):
            raise ValueError("Invalid eating progress.")
        return self


class PreferencesState(SaveModel):
    sociability: float = Field(ge=0, le=1)
    generosity: float = Field(ge=0, le=1)


class MemoryState(SaveModel):
    id: str = Field(min_length=1)
    day: int = Field(ge=1)
    phase: Literal["day", "evening", "night"]
    partner_id: str
    kind: Literal["gave_fruit", "received_fruit", "refused", "kept_fruit", "conversation",
                  "promise_created", "promise_fulfilled", "promise_broken", "goal_completed", "goal_failed",
                  "claim_shared", "heard_claim", "claim_opinion", "leverage"]
    summary: str = Field(max_length=400)
    weight: float = Field(ge=-12, le=12)


class EventState(SaveModel):
    id: str = Field(min_length=1)
    day: int = Field(ge=1)
    phase: Literal["day", "evening", "night"]
    kind: Literal["phase", "food_requested", "food_refused", "fruit_shared", "conversation_completed", "interaction_failed",
                  "promise_created", "promise_fulfilled", "promise_broken", "goal_completed", "goal_failed",
                  "claim_shared", "claim_assessed", "reputation_changed", "leverage_used"]
    participants: list[str] = Field(max_length=2)
    text: str
    history: int | None = Field(ge=0)


class GoalState(SaveModel):
    id: str = Field(min_length=1)
    kind: GoalKind
    target_id: str
    objective: str = Field(min_length=1, max_length=400)
    motive: str = Field(min_length=1, max_length=400)
    approach: str = Field(min_length=1, max_length=400)
    boundary: str = Field(min_length=1, max_length=400)
    created_phase: int = Field(ge=0)
    deadline: int = Field(ge=0)
    baseline_opinion: int = Field(ge=-100, le=100)
    status: Literal["active", "completed", "failed"]
    resolution_event_id: str | None
    resolved_phase: int | None = Field(ge=0)
    outcome: str
    last_attempt_phase: int | None = Field(ge=0)
    third_npc_id: str | None
    claim_id: str | None


class ProposalState(SaveModel):
    id: str = Field(min_length=1)
    promisor_id: str
    recipient_id: str
    deadline: int = Field(ge=0)


class PromiseState(ProposalState):
    conversation_id: str = Field(min_length=1)
    created_phase: int = Field(ge=0)
    item: Literal["fruit"]
    quantity: Literal[1]
    status: Literal["pending", "fulfilled", "broken"]
    resolution_event_id: str | None
    resolved_phase: int | None = Field(ge=0)
    outcome: str
    consequence_event_id: str | None


class OpinionEffectState(SaveModel):
    event_id: str = Field(min_length=1)
    target_id: str
    delta: int = Field(ge=-6, le=6)
    reason: str = Field(min_length=1, max_length=400)
    claim_id: str | None
    actor_id: str | None
    conversation_id: str | None
    phase: int | None = Field(ge=0)


class FactState(SaveModel):
    id: str = Field(min_length=1)
    kind: FactKind
    subject_id: str
    object_id: str
    source_id: str = Field(min_length=1)
    conversation_id: str | None
    phase: int = Field(ge=0)
    summary: str = Field(min_length=1)
    witness_ids: list[str] = Field(min_length=1, max_length=2)


class ClaimState(SaveModel):
    id: str = Field(min_length=1)
    fact_id: str
    subject_id: str
    originator_id: str
    affirmed: bool
    parent_claim_id: str | None
    deliberate: bool
    phase: int = Field(ge=0)
    conversation_id: str | None


class BeliefState(SaveModel):
    npc_id: str
    claim_id: str
    stance: Stance
    source_id: str
    disclosure_id: str | None
    direct: bool
    phase: int = Field(ge=0)
    assessed: bool
    opinion_delta: int = Field(ge=-6, le=6)


class DisclosureState(SaveModel):
    id: str = Field(min_length=1)
    claim_id: str
    conversation_id: str
    speaker_id: str
    listener_id: str
    turn: int = Field(ge=0, le=7)
    intent: Literal["share", "challenge", "lie"]
    stance: Stance
    source_disclosure_id: str | None
    phase: int = Field(ge=0)
    event_id: str = Field(min_length=1)


class ClaimOptionState(SaveModel):
    id: str = Field(min_length=1)
    fact_id: str
    subject_id: str
    originator_id: str
    affirmed: bool
    parent_claim_id: str | None
    creator_id: str | None
    kind: FactKind
    assertion: str = Field(min_length=1)


class InformationState(SaveModel):
    participants: dict[str, str]
    beliefs: dict[str, list[BeliefState]]
    options: list[ClaimOptionState]
    can_lie: dict[str, bool]
    opinions: dict[str, dict[str, int]]
    allowed: dict[str, list[str]]
    focus_claim_id: str | None


class NPCState(ActivityState):
    id: str = Field(min_length=1)
    character: dict
    display_name: str | None
    x: float = Field(ge=0, le=WORLD_WIDTH)
    y: float = Field(ge=0, le=WORLD_HEIGHT)
    reply: str
    emotion: EmotionReaction
    social: float = Field(ge=0, le=100)
    inventory: dict[Literal["fruit"], int]
    inventory_capacity: int = Field(ge=1)
    preferences: PreferencesState
    memories: list[MemoryState] = Field(max_length=20)
    goal: GoalState | None
    pending_social_opinions: list[OpinionEffectState]
    beliefs: list[BeliefState]
    command_revision: int = Field(ge=0)

    @field_validator("character")
    @classmethod
    def validate_character(cls, value):
        return normalize_character(value)

    @model_validator(mode="after")
    def validate_inventory(self):
        if any(value < 0 for value in self.inventory.values()) or sum(self.inventory.values()) > self.inventory_capacity:
            raise ValueError("Invalid inventory.")
        return self


class HistoryState(SaveModel):
    id: str = Field(min_length=1)
    participants: tuple[str, str]
    topic: str
    status: str
    turns: list[tuple[str, str]]
    dialogue_parts: dict[int, list[ConversationPart]] = Field(default_factory=dict)
    outcomes: list[str]
    error: str
    origin: Literal["player", "autonomous"]
    purpose: Literal["chat", "food_request", "gossip"]


class ConversationState(SaveModel):
    id: str = Field(min_length=1)
    origin: Literal["player", "autonomous"]
    purpose: Literal["chat", "food_request", "gossip"]
    focus_claim_id: str | None
    dialogue_state: Literal["waiting", "pending", "ready"]
    wait_elapsed: float = Field(ge=0, le=DIALOGUE_WAIT_SECONDS + 0.1)
    talking_remaining: float = Field(ge=0, le=4)
    food_outcome: Literal["", "accepted", "need", "refused", "unavailable"]
    food_reason: str
    handoff_thanks: list[ConversationPart] = Field(max_length=3)
    outcome_event_id: str | None
    reactions: list[tuple[str, float]] = Field(max_length=2)
    initiator: int = Field(ge=0)
    recipient: int = Field(ge=0)
    record: int = Field(ge=0)
    topic: str
    arrived: bool
    turns: list[ConversationTurn] = Field(max_length=8)
    saved_states: tuple[ActivityState, ActivityState]
    pending_opinions: list[tuple[int, dict]]
    proposals: list[ProposalState] = Field(max_length=2)
    accepted_proposals: list[AcceptedProposal] = Field(max_length=2)
    social_effects_applied: bool
    context_references: dict[str, Literal["active", "completed", "failed", "pending", "fulfilled", "broken"]]
    information_context: InformationState | None
    accepted_disclosures: list[DisclosureChoice] = Field(max_length=4)
    third_party_opinions: list[ThirdPartyOpinion] = Field(max_length=4)
    leverage_uses: list[LeverageUse] = Field(max_length=2)

    @field_validator("pending_opinions")
    @classmethod
    def validate_opinions(cls, value):
        return [(index, normalize_opinions(opinions)) for index, opinions in value]


class GiftState(SaveModel):
    initiator: int = Field(ge=0)
    recipient: int = Field(ge=0)
    record: int = Field(ge=0)
    item: Literal["fruit"]
    quantity: int = Field(ge=1)
    arrived: bool
    elapsed: float = Field(ge=0, lt=2)
    transferred: bool
    outcome_event_id: str | None
    thanks: list[ConversationPart] = Field(max_length=3)
    saved_states: tuple[ActivityState, ActivityState]


class TreeState(SaveModel):
    id: str = Field(min_length=1)
    x: float = Field(ge=60, le=WORLD_WIDTH)
    y: float = Field(ge=0, le=WORLD_HEIGHT - 50)
    fruit: int = Field(ge=0)
    capacity: int = Field(ge=1)
    shake_elapsed: float | None = Field(ge=0, le=SHAKE_SECONDS)
    shaker: str | None
    reserved_by: str | None


class SnakeState(SaveModel):
    x: float = Field(default=720.0, ge=SNAKE_MARGIN, le=WORLD_WIDTH - SNAKE_MARGIN)
    y: float = Field(default=330.0, ge=SNAKE_MARGIN, le=WORLD_HEIGHT - SNAKE_MARGIN)
    destination: tuple[float, float] | None = None
    facing: int = Field(default=4, ge=0, le=7)
    pause_remaining: float = Field(default=0.0, ge=0)
    animation_elapsed: float = Field(default=0.0, ge=0, lt=2 * SNAKE_FRAME_SECONDS)

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value):
        if value is not None and not (
            SNAKE_MARGIN <= value[0] <= WORLD_WIDTH - SNAKE_MARGIN
            and SNAKE_MARGIN <= value[1] <= WORLD_HEIGHT - SNAKE_MARGIN
        ):
            raise ValueError("Invalid snake destination.")
        return value


class PendingActionState(SaveModel):
    npc_id: str
    action: Literal["WANDER", "STOP", "GO_TO_TREE", "SHAKE_TREE", "TALK_TO_NPC", "GIVE_ITEM", "EAT"]
    revision: int = Field(ge=0)
    target_id: str | None
    topic: str
    record: int | None = Field(ge=0)
    item: Literal["", "fruit"] = ""
    quantity: int = Field(default=0, ge=0)


class SimulationState(SaveModel):
    day: int = Field(ge=1)
    phase: Literal["day", "evening", "night"]
    events: list[EventState] = Field(max_length=200)
    promises: list[PromiseState]
    facts: list[FactState]
    claims: list[ClaimState]
    disclosures: list[DisclosureState]
    social_opportunity: bool
    social_rotation: int = Field(ge=0)
    npcs: list[NPCState] = Field(min_length=1)
    tree: TreeState
    snake: SnakeState = Field(default_factory=SnakeState)
    pending_actions: list[PendingActionState]
    history: list[HistoryState]
    conversations: list[ConversationState]
    gifts: list[GiftState] = Field(default_factory=list)
    camera_offset: tuple[float, float]
    camera_zoom: float = Field(default=1.0, ge=0.5, le=2.5)

    @model_validator(mode="after")
    def validate_references(self):
        count = len(self.npcs)
        names = [(npc.display_name or npc.character["Name"]).casefold() for npc in self.npcs]
        ids = [npc.id for npc in self.npcs]
        all_ids = ids + [self.tree.id]
        if len(set(names)) != count or len(set(all_ids)) != len(all_ids):
            raise ValueError("Names and object IDs must be unique.")
        if self.tree.fruit > self.tree.capacity:
            raise ValueError("Invalid fruit resource.")
        if self.social_rotation >= count:
            raise ValueError("Invalid social rotation.")
        event_ids = [event.id for event in self.events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("Duplicate event IDs.")
        now = phase_index(self.day, self.phase)
        history_by_id = {entry.id: entry for entry in self.history}
        if len(history_by_id) != len(self.history):
            raise ValueError("Duplicate history IDs.")
        promise_ids, pending_pairs, pending_counts = set(), set(), {}
        goal_ids = [npc.goal.id for npc in self.npcs if npc.goal]
        if len(set(goal_ids)) != len(goal_ids):
            raise ValueError("Duplicate goal IDs.")
        retained_events = {event.id: event for event in self.events}
        facts = {fact.id: fact for fact in self.facts}
        claims = {claim.id: claim for claim in self.claims}
        disclosures = {entry.id: entry for entry in self.disclosures}
        if len(facts) != len(self.facts) or len(claims) != len(self.claims) or len(disclosures) != len(self.disclosures):
            raise ValueError("Duplicate information record IDs.")
        if len({(fact.source_id, fact.kind) for fact in self.facts}) != len(self.facts):
            raise ValueError("Duplicate factual sources.")

        def history_pair(record_id, pair):
            record = history_by_id.get(record_id)
            return (record is not None and {name.casefold() for name in record.participants}
                    == {names[ids.index(npc_id)] for npc_id in pair})

        for fact in self.facts:
            if (fact.subject_id not in ids or fact.object_id not in ids or fact.subject_id == fact.object_id
                    or fact.phase > now or len(set(fact.witness_ids)) != len(fact.witness_ids)
                    or set(fact.witness_ids) != ({fact.subject_id} if fact.kind == "private_motive"
                                                else {fact.subject_id, fact.object_id})):
                raise ValueError("Invalid fact participants or visibility.")
            if fact.kind == "private_motive":
                if fact.conversation_id is not None:
                    raise ValueError("A private intention is not a conversation event.")
            elif not history_pair(fact.conversation_id, (fact.subject_id, fact.object_id)):
                raise ValueError("Invalid factual conversation source.")
            event = retained_events.get(fact.source_id)
            if event:
                expected = {"apple_given": "fruit_shared", "apple_refused": "food_refused",
                            "promise_kept": "promise_fulfilled", "promise_broken": "promise_broken"}.get(fact.kind)
                pair = [fact.object_id, fact.subject_id] if fact.kind == "apple_refused" else [fact.subject_id, fact.object_id]
                if event.kind != expected or event.participants != pair or fact.phase != phase_index(event.day, event.phase):
                    raise ValueError("Fact does not match its committed event.")
        for claim in self.claims:
            fact = facts.get(claim.fact_id)
            if (fact is None or claim.subject_id != fact.subject_id or claim.originator_id not in ids
                    or not fact.phase <= claim.phase <= now):
                raise ValueError("Invalid claim references.")
            if claim.parent_claim_id is None:
                if not claim.affirmed or claim.deliberate or claim.originator_id not in fact.witness_ids:
                    raise ValueError("An original firsthand account must describe an actual fact.")
            else:
                parent = claims.get(claim.parent_claim_id)
                record = history_by_id.get(claim.conversation_id)
                if (parent is None or parent.fact_id != fact.id or not parent.affirmed
                        or claim.affirmed or not claim.deliberate or claim.phase < parent.phase
                        or record is None or names[ids.index(claim.originator_id)] not in {name.casefold() for name in record.participants}):
                    raise ValueError("A fabricated assertion must negate a known, existing account.")
        seen_disclosures = set()
        for entry in self.disclosures:
            claim = claims.get(entry.claim_id)
            record = history_by_id.get(entry.conversation_id)
            if (claim is None or entry.speaker_id == entry.listener_id
                    or entry.speaker_id not in ids or entry.listener_id not in ids
                    or not history_pair(entry.conversation_id, (entry.speaker_id, entry.listener_id))
                    or not claim.phase <= entry.phase <= now or entry.turn >= len(record.turns)
                    or record.turns[entry.turn][0].casefold() != names[ids.index(entry.speaker_id)]):
                raise ValueError("Invalid disclosure source or participants.")
            if entry.source_disclosure_id:
                source = disclosures.get(entry.source_disclosure_id)
                if (source is None or source.id not in seen_disclosures or source.listener_id != entry.speaker_id
                        or source.claim_id not in (claim.id, claim.parent_claim_id)
                        or source.phase > entry.phase
                        or source.conversation_id == entry.conversation_id and source.turn >= entry.turn):
                    raise ValueError("Disclosure provenance must follow an earlier source.")
            elif entry.speaker_id != claim.originator_id:
                raise ValueError("An account repeated by another speaker needs its source.")
            event = retained_events.get(entry.event_id)
            if event and (event.kind != "claim_shared" or event.participants != [entry.speaker_id, entry.listener_id]):
                raise ValueError("Invalid disclosure event.")
            seen_disclosures.add(entry.id)

        def validate_beliefs(entries, npc_id):
            if len({belief.claim_id for belief in entries}) != len(entries):
                raise ValueError("Duplicate personal beliefs.")
            assessed, totals = set(), {}
            for belief in entries:
                claim = claims.get(belief.claim_id)
                if (belief.npc_id != npc_id or claim is None or belief.source_id not in ids
                        or not claim.phase <= belief.phase <= now or not belief.assessed and belief.opinion_delta):
                    raise ValueError("Invalid personal belief references.")
                if belief.direct:
                    if (npc_id not in facts[claim.fact_id].witness_ids or claim.originator_id != npc_id
                            or not claim.affirmed or belief.stance != "believed"
                            or belief.source_id != npc_id or belief.disclosure_id is not None):
                        raise ValueError("Invalid firsthand knowledge.")
                elif belief.disclosure_id is not None:
                    source = disclosures.get(belief.disclosure_id)
                    if (source is None or source.listener_id != npc_id or source.claim_id != claim.id
                            or source.speaker_id != belief.source_id or source.phase < belief.phase):
                        raise ValueError("Invalid belief provenance.")
                elif not (claim.deliberate and claim.originator_id == npc_id and belief.source_id == npc_id
                          and belief.stance == "disbelieved"):
                    raise ValueError("Hearsay needs a disclosure source.")
                if belief.assessed:
                    key = (claim.fact_id, claim.affirmed)
                    if key in assessed:
                        raise ValueError("Repeated assertions cannot multiply reputation effects.")
                    assessed.add(key)
                totals[claim.fact_id] = totals.get(claim.fact_id, 0) + belief.opinion_delta
            if any(abs(total) > 6 for total in totals.values()):
                raise ValueError("Invalid combined reputation influence.")

        def validate_resolution(record, active):
            if record.created_phase > now or record.deadline < record.created_phase:
                raise ValueError("Invalid social deadline.")
            if active:
                if record.deadline < now or record.resolved_phase is not None or record.resolution_event_id or record.outcome:
                    raise ValueError("Invalid active social outcome.")
            elif (record.resolved_phase is None or not record.resolution_event_id or not record.outcome
                  or not record.created_phase <= record.resolved_phase <= now):
                raise ValueError("Invalid resolved social outcome.")

        for promise in self.promises:
            pair = (promise.promisor_id, promise.recipient_id)
            origin = history_by_id.get(promise.conversation_id)
            if (promise.id in promise_ids or pair[0] == pair[1] or any(npc_id not in ids for npc_id in pair)
                    or origin is None or origin.purpose == "food_request"
                    or {name.casefold() for name in origin.participants} != {names[ids.index(npc_id)] for npc_id in pair}):
                raise ValueError("Invalid promise references.")
            validate_resolution(promise, promise.status == "pending")
            if (promise.consequence_event_id is not None) != (promise.status != "pending"):
                raise ValueError("Invalid promise consequence reference.")
            if promise.status == "broken" and promise.resolved_phase <= promise.deadline:
                raise ValueError("A promise breaks only after its deadline phase.")
            if promise.status == "fulfilled" and promise.resolved_phase > promise.deadline:
                raise ValueError("A late gift cannot fulfill an expired promise.")
            event = retained_events.get(promise.resolution_event_id)
            if event is not None and (event.kind != ("fruit_shared" if promise.status == "fulfilled" else "promise_broken")
                                      or event.participants != list(pair)):
                raise ValueError("Invalid promise resolution event.")
            event = retained_events.get(promise.consequence_event_id)
            if event is not None and (event.kind != f"promise_{promise.status}" or event.participants != list(pair)):
                raise ValueError("Invalid promise consequence event.")
            if promise.status == "pending":
                pending_counts[pair[0]] = pending_counts.get(pair[0], 0) + 1
                if pair in pending_pairs or pending_counts[pair[0]] > 2:
                    raise ValueError("Invalid outstanding promises.")
                pending_pairs.add(pair)
            promise_ids.add(promise.id)
        for event in self.events:
            if (any(participant not in ids for participant in event.participants)
                    or len(set(event.participants)) != len(event.participants)
                    or (event.history is not None and event.history >= len(self.history))):
                raise ValueError("Invalid event references.")
        for npc in self.npcs:
            if len({memory.id for memory in npc.memories}) != len(npc.memories):
                raise ValueError("Duplicate memories.")
            if any(memory.partner_id not in ids or memory.partner_id == npc.id for memory in npc.memories):
                raise ValueError("Invalid memory partner.")
            if npc.goal:
                if npc.goal.target_id not in ids or npc.goal.target_id == npc.id:
                    raise ValueError("Invalid goal target.")
                validate_resolution(npc.goal, npc.goal.status == "active")
                if not any(fact.kind == "private_motive" and fact.source_id == npc.goal.id
                           and fact.subject_id == npc.id and npc.goal.motive in fact.summary for fact in self.facts):
                    raise ValueError("A private goal needs its owner-only intention fact.")
                if npc.goal.kind == "sabotage_relationship":
                    claim = claims.get(npc.goal.claim_id)
                    if (npc.goal.third_npc_id not in ids or npc.goal.third_npc_id in (npc.id, npc.goal.target_id)
                            or claim is None or claim.subject_id != npc.goal.third_npc_id
                            or npc.goal.claim_id not in {belief.claim_id for belief in npc.beliefs}):
                        raise ValueError("A sabotage goal needs a known account about a third NPC.")
                elif npc.goal.third_npc_id is not None or npc.goal.claim_id is not None:
                    raise ValueError("Unexpected sabotage references.")
                if npc.goal.last_attempt_phase is not None and not npc.goal.created_phase <= npc.goal.last_attempt_phase <= now:
                    raise ValueError("Invalid goal attempt phase.")
                if npc.goal.status == "completed" and npc.goal.resolved_phase > npc.goal.deadline:
                    raise ValueError("A goal cannot complete after its deadline.")
                if npc.goal.status == "failed" and npc.goal.resolved_phase <= npc.goal.deadline:
                    raise ValueError("A goal expires after its deadline phase.")
                event = retained_events.get(npc.goal.resolution_event_id)
                if event is not None and (event.kind != f"goal_{npc.goal.status}"
                                          or event.participants != [npc.id, npc.goal.target_id]):
                    raise ValueError("Invalid goal resolution event.")
            if (len({effect.event_id for effect in npc.pending_social_opinions}) != len(npc.pending_social_opinions)
                    or any(effect.target_id not in ids or effect.target_id == npc.id for effect in npc.pending_social_opinions)):
                raise ValueError("Invalid pending social opinion effects.")
            for effect in npc.pending_social_opinions:
                if effect.claim_id is not None:
                    claim = claims.get(effect.claim_id)
                    belief = next((b for b in npc.beliefs if b.claim_id == effect.claim_id), None)
                    source = next((d for d in self.disclosures if d.conversation_id == effect.conversation_id
                                   and d.claim_id == effect.claim_id and d.speaker_id == effect.actor_id and d.listener_id == npc.id), None)
                    if (claim is None or claim.subject_id != effect.target_id or belief is None or not belief.assessed
                            or belief.opinion_delta != effect.delta or source is None or effect.delta == 0
                            or effect.phase is None or effect.phase != source.phase):
                        raise ValueError("A reputation adjustment needs its heard account and assessment.")
                    continue
                promise = next((p for p in self.promises if p.consequence_event_id == effect.event_id), None)
                if (promise is None or promise.recipient_id != npc.id or promise.promisor_id != effect.target_id
                        or effect.delta != (4 if promise.status == "fulfilled" else -6)
                        or effect.reason != promise.outcome or effect.actor_id is not None or effect.conversation_id is not None
                        or effect.phase is not None):
                    raise ValueError("Opinion effect does not match a committed promise outcome.")
            validate_beliefs(npc.beliefs, npc.id)
        if (self.tree.shaker is None) != (self.tree.shake_elapsed is None):
            raise ValueError("A shaking tree needs a shaker and elapsed time.")
        if self.tree.reserved_by is not None and self.tree.reserved_by not in ids:
            raise ValueError("Invalid tree reservation.")
        if self.tree.shaker is not None and self.tree.shaker != self.tree.reserved_by:
            raise ValueError("The shaker must own the tree slot.")
        reserved, records = set(), set()
        if len({conversation.id for conversation in self.conversations}) != len(self.conversations):
            raise ValueError("Duplicate interaction IDs.")

        def validate_target(activity, npc):
            if any(target not in all_ids or target == npc.id for target in activity.cooldowns):
                raise ValueError("Invalid cooldown target.")
            target = activity.target_id
            if activity.behavior in (Behavior.HARVEST, Behavior.GO_TO_TREE, Behavior.SHAKE_TREE, Behavior.SHAKING):
                if target != self.tree.id:
                    raise ValueError("Invalid tree target.")
                point = (self.tree.x - 60, self.tree.y + 50)
                if activity.destination is not None and activity.destination != point:
                    raise ValueError("Invalid tree destination.")
                if activity.behavior in (Behavior.GO_TO_TREE, Behavior.SHAKE_TREE) and activity.destination is None:
                    raise ValueError("A walking task needs a destination.")
                if activity.behavior in (Behavior.HARVEST, Behavior.SHAKING) and activity.activity_elapsed > SHAKE_SECONDS:
                    raise ValueError("Invalid shake progress.")
            elif activity.behavior == Behavior.COMPANY:
                if target not in ids or target == npc.id:
                    raise ValueError("Invalid companion.")
            elif target is not None:
                raise ValueError("Unexpected activity target.")

        for conversation in self.conversations:
            pair = (conversation.initiator, conversation.recipient)
            if pair[0] == pair[1] or any(index >= count or index in reserved for index in pair):
                raise ValueError("Invalid conversation participants.")
            if conversation.record >= len(self.history) or conversation.record in records:
                raise ValueError("Invalid conversation history reference.")
            if self.history[conversation.record].id != conversation.id:
                raise ValueError("Conversation identity does not match its history.")
            if not conversation.arrived and self.npcs[pair[0]].destination is None:
                raise ValueError("A meeting in progress needs a destination.")
            if self.npcs[pair[1]].destination is not None:
                raise ValueError("The conversation recipient must be waiting.")
            if conversation.dialogue_state == "pending" and not conversation.arrived:
                raise ValueError("Dialogue requests start only after meeting.")
            if any(not reason.strip() or not -4 <= weight <= 4 for reason, weight in conversation.reactions):
                raise ValueError("Invalid conversation reaction.")
            if conversation.arrived and any(self.npcs[index].destination is not None for index in pair):
                raise ValueError("Arrived conversation participants must be still.")
            if (self.history[conversation.record].origin != conversation.origin
                    or self.history[conversation.record].purpose != conversation.purpose):
                raise ValueError("Conversation history purpose does not match.")
            ready = conversation.dialogue_state == "ready"
            if conversation.social_effects_applied != ready:
                raise ValueError("Conversation social effects must match accepted dialogue.")
            if len({p.id for p in conversation.proposals}) != len(conversation.proposals):
                raise ValueError("Duplicate proposal IDs.")
            pair_ids = {self.npcs[index].id for index in pair}
            context = conversation.information_context
            if context is not None:
                roles = {"initiator", "recipient"}
                if (set(context.participants) != roles or set(context.beliefs) != roles
                        or set(context.allowed) != roles or set(context.can_lie) != roles or set(context.opinions) != roles
                        or context.participants != {"initiator": self.npcs[pair[0]].id, "recipient": self.npcs[pair[1]].id}):
                    raise ValueError("Information context belongs to the wrong speakers.")
                options = {option.id: option for option in context.options}
                if len(options) != len(context.options):
                    raise ValueError("Duplicate permitted accounts.")
                if context.focus_claim_id is not None and context.focus_claim_id not in context.allowed["initiator"]:
                    raise ValueError("The gossip focus must be a permitted account.")
                for option in context.options:
                    fact = facts.get(option.fact_id)
                    if (fact is None or option.subject_id != fact.subject_id or option.kind != fact.kind
                            or option.originator_id not in ids):
                        raise ValueError("Invalid permitted account references.")
                    if option.creator_id:
                        parent = claims.get(option.parent_claim_id)
                        if (option.creator_id not in pair_ids or option.originator_id != option.creator_id
                                or option.affirmed or parent is None or not parent.affirmed or parent.fact_id != fact.id):
                            raise ValueError("Invalid permitted denial.")
                    else:
                        claim = claims.get(option.id)
                        if (claim is None or (claim.fact_id, claim.subject_id, claim.originator_id, claim.affirmed, claim.parent_claim_id)
                                != (option.fact_id, option.subject_id, option.originator_id, option.affirmed, option.parent_claim_id)):
                            raise ValueError("Permitted account does not match its saved claim.")
                for role, npc_id in context.participants.items():
                    validate_beliefs(context.beliefs[role], npc_id)
                    known = {b.claim_id for b in context.beliefs[role]}
                    if not known.issubset(options):
                        raise ValueError("Information context is missing a known account.")
                    for claim_id in context.allowed[role]:
                        option = options.get(claim_id)
                        if option is None or (claim_id not in known and not
                                              (option.creator_id == npc_id and option.parent_claim_id in known and context.can_lie[role])):
                            raise ValueError("Speaker was permitted to disclose an unknown account.")
                    if (set(context.opinions[role]) != set(ids) - {npc_id}
                            or any(not -100 <= score <= 100 for score in context.opinions[role].values())):
                        raise ValueError("Invalid private opinion context.")
            if conversation.purpose != "food_request" and (ready or conversation.dialogue_state == "pending") and context is None:
                raise ValueError("A generated chat needs its private information context.")
            if conversation.purpose == "food_request" and context is not None:
                raise ValueError("Food requests do not negotiate disclosures.")
            if conversation.purpose == "gossip":
                if (conversation.focus_claim_id not in claims or conversation.focus_claim_id not in
                        {b.claim_id for b in self.npcs[pair[0]].beliefs}):
                    raise ValueError("Gossip needs an account known by its initiator.")
            elif conversation.focus_claim_id is not None:
                raise ValueError("Unexpected gossip focus.")
            if context is not None and context.focus_claim_id != conversation.focus_claim_id:
                raise ValueError("Information context has the wrong gossip focus.")
            if not ready and (conversation.accepted_disclosures or conversation.third_party_opinions or conversation.leverage_uses):
                raise ValueError("Unaccepted speech cannot have information effects.")
            if ready:
                validate_information(SimpleNamespace(turns=conversation.turns, disclosures=conversation.accepted_disclosures,
                                                     third_party_opinions=conversation.third_party_opinions, leverage=conversation.leverage_uses),
                                     information_from_dict(context.model_dump()) if context else None)
            applied = [entry for entry in self.disclosures if entry.conversation_id == conversation.id]
            if {(entry.turn, entry.claim_id, entry.intent, entry.stance) for entry in applied} != {
                    (entry.turn, entry.claim_id, entry.intent, entry.listener_stance) for entry in conversation.accepted_disclosures}:
                raise ValueError("Accepted disclosures do not match applied information effects.")
            if any({p.promisor_id, p.recipient_id} != pair_ids for p in conversation.proposals):
                raise ValueError("Invalid proposal participants.")
            validate_agreements(conversation.accepted_proposals, [p.model_dump() for p in conversation.proposals],
                                conversation.turns, self.npcs[pair[0]].id, self.npcs[pair[1]].id)
            if not ready and conversation.accepted_proposals:
                raise ValueError("Unaccepted dialogue cannot have agreements.")
            if {p.id for p in self.promises if p.conversation_id == conversation.id} != {
                    agreement.proposal_id for agreement in conversation.accepted_proposals}:
                raise ValueError("Conversation agreements do not match applied promises.")
            for agreement in conversation.accepted_proposals:
                promise = next((p for p in self.promises if p.id == agreement.proposal_id), None)
                proposal = next(p for p in conversation.proposals if p.id == agreement.proposal_id)
                if (promise is None or promise.conversation_id != self.history[conversation.record].id
                        or (promise.promisor_id, promise.recipient_id, promise.deadline)
                        != (proposal.promisor_id, proposal.recipient_id, proposal.deadline)):
                    raise ValueError("Accepted agreement is missing its applied promise.")
            if ready != bool(conversation.turns) or (ready and not conversation.arrived):
                raise ValueError("Invalid dialogue progress.")
            if ready and len(conversation.reactions) != 2:
                raise ValueError("A generated conversation needs both reactions.")
            turn_count = len(conversation.turns)
            if ready and ((conversation.purpose == "food_request" and turn_count != 2)
                          or (conversation.purpose != "food_request" and turn_count < 4)):
                raise ValueError("Invalid conversation turn count.")
            if conversation.purpose == "food_request":
                if (conversation.proposals or conversation.accepted_proposals
                        or conversation.origin != "autonomous" or bool(conversation.food_outcome) != conversation.arrived
                        or bool(conversation.food_reason) != conversation.arrived
                        or bool(conversation.handoff_thanks) != (ready and conversation.food_outcome == "accepted")
                        or bool(conversation.outcome_event_id) != (ready and conversation.food_outcome in ("need", "refused", "unavailable"))):
                    raise ValueError("Invalid food request progress.")
            elif conversation.food_outcome or conversation.food_reason or conversation.handoff_thanks:
                raise ValueError("Invalid conversation outcome.")
            if any(index not in pair for index, _ in conversation.pending_opinions):
                raise ValueError("Invalid pending opinion participant.")
            for index, activity in zip(pair, conversation.saved_states):
                validate_target(activity, self.npcs[index])
            reserved.update(pair)
            records.add(conversation.record)
        for gift in self.gifts:
            pair = (gift.initiator, gift.recipient)
            if pair[0] == pair[1] or any(index >= count or index in reserved for index in pair):
                raise ValueError("Invalid gift participants.")
            if gift.record >= len(self.history) or gift.record in records:
                raise ValueError("Invalid gift history reference.")
            giver, recipient = (self.npcs[index] for index in pair)
            if self.history[gift.record].purpose == "food_request" and gift.quantity != 1:
                raise ValueError("A food request shares exactly one apple.")
            if bool(gift.thanks) != (self.history[gift.record].purpose == "food_request"):
                raise ValueError("Food request gifts must carry generated thanks.")
            if gift.arrived != (giver.destination is None) or (not gift.arrived and gift.elapsed != 0):
                raise ValueError("Invalid gift progress.")
            if recipient.destination is not None:
                raise ValueError("The gift recipient must be waiting.")
            if gift.transferred != (gift.elapsed >= 1) or gift.transferred != bool(gift.outcome_event_id):
                raise ValueError("Invalid gift transfer progress.")
            for index, activity in zip(pair, gift.saved_states):
                validate_target(activity, self.npcs[index])
            reserved.update(pair)
            records.add(gift.record)
        for index, npc in enumerate(self.npcs):
            if index in reserved:
                if npc.behavior != Behavior.WAIT or npc.owner != "autonomous" or npc.target_id is not None or npc.resume_pending:
                    raise ValueError("Invalid activity during an interaction.")
                if npc.id == self.tree.reserved_by:
                    raise ValueError("Interaction participants cannot reserve the tree.")
                continue
            validate_target(npc, npc)
            if npc.behavior == Behavior.EAT and npc.inventory.get("fruit", 0) < 1:
                raise ValueError("Eating needs an apple in inventory.")
            if npc.resume_pending:
                if npc.id == self.tree.reserved_by:
                    raise ValueError("A suspended activity cannot reserve the tree.")
                continue
            if npc.behavior in (Behavior.HARVEST, Behavior.SHAKING) and self.tree.reserved_by != npc.id:
                raise ValueError("An active harvest or shake must reserve the tree.")
            if npc.behavior == Behavior.HARVEST and (self.tree.fruit == 0 or sum(npc.inventory.values()) >= npc.inventory_capacity):
                raise ValueError("An active harvest needs available fruit and inventory space.")
            if self.tree.reserved_by == npc.id:
                if npc.target_id != self.tree.id or npc.behavior not in (Behavior.HARVEST, Behavior.GO_TO_TREE, Behavior.SHAKE_TREE, Behavior.SHAKING):
                    raise ValueError("Invalid tree owner activity.")
            if self.tree.shaker == npc.id:
                if npc.destination is not None or npc.behavior not in (Behavior.HARVEST, Behavior.SHAKING):
                    raise ValueError("Invalid shaker activity.")
        pending = set()
        for action in self.pending_actions:
            if action.npc_id not in ids or action.npc_id in pending:
                raise ValueError("Invalid pending action participant.")
            if action.action in ("TALK_TO_NPC", "GIVE_ITEM"):
                if action.target_id not in ids or action.target_id == action.npc_id:
                    raise ValueError("Invalid pending recipient.")
            elif action.target_id is not None:
                raise ValueError("Unexpected pending action target.")
            if action.action == "GIVE_ITEM":
                if action.item != "fruit" or action.quantity < 1:
                    raise ValueError("Invalid pending gift.")
            elif action.item or action.quantity:
                raise ValueError("Only giving actions may include items.")
            if action.record is not None and action.record >= len(self.history):
                raise ValueError("Invalid pending action history.")
            pending.add(action.npc_id)
        return self


class SimulationSave(SaveModel):
    name: str = ""
    created_at: datetime
    updated_at: datetime
    state: SimulationState


@dataclass
class SaveEntry:
    id: str
    name: str
    updated_at: datetime | None = None
    npc_count: int = 0
    error: str = ""


class InvalidSaveName(ValueError):
    pass


class SaveStore:
    def __init__(self, directory=SAVE_DIRECTORY):
        self.directory = Path(directory)

    def _path(self, save_id):
        if (not isinstance(save_id, str) or not save_id.strip() or save_id in (".", "..")
                or save_id != save_id.strip() or save_id.endswith(".")
                or any(ord(char) < 32 or char in '<>:"/\\|?*' for char in save_id)):
            raise InvalidSaveName('Use a filename without <>:"/\\|?* or trailing spaces/dots.')
        if re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", save_id.split(".")[0], re.IGNORECASE):
            raise InvalidSaveName("That filename is reserved. Choose another name.")
        path = self.directory / f"{save_id}.json"
        if path.resolve().parent != self.directory.resolve():
            raise ValueError("Invalid save location.")
        return path

    def load(self, save_id):
        save = SimulationSave.model_validate_json(self._path(save_id).read_text(encoding="utf-8"))
        save.name = save_id
        return save

    def list_saves(self):
        entries = []
        for path in self.directory.glob("*.json"):
            try:
                save = self.load(path.stem)
                entries.append(SaveEntry(path.stem, save.name, save.updated_at, len(save.state.npcs)))
            except (OSError, ValueError):
                entries.append(SaveEntry(path.stem, path.stem, error="This save cannot be loaded. You can delete it."))
        return sorted(entries, key=lambda entry: entry.updated_at.timestamp() if entry.updated_at else 0, reverse=True)

    def next_name(self):
        used = {path.stem.casefold() for path in self.directory.glob("*.json")}
        number = 1
        while f"simulation {number}" in used:
            number += 1
        return f"Simulation {number}"

    def _require_available(self, name, *, exclude=None):
        target = self._path(name)
        if self.directory.exists():
            for path in self.directory.iterdir():
                if path.name.casefold() == target.name.casefold() and path != exclude:
                    raise FileExistsError("A save with that name already exists. Choose another name.")

    def file_identity(self, name):
        stat = self._path(name).stat()
        return stat.st_dev, stat.st_ino

    def current_name(self, name, identity):
        """Follow a file renamed outside the app without recreating its old filename."""
        try:
            if self.file_identity(name) == identity:
                return name
        except FileNotFoundError:
            pass
        for path in self.directory.glob("*.json"):
            try:
                if self.file_identity(path.stem) == identity:
                    return path.stem
            except FileNotFoundError:
                continue
        return name

    def _write(self, save_id, save, *, overwrite=True):
        path = self._path(save_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(save.model_dump_json(indent=2) + "\n", encoding="utf-8")
            if not overwrite:
                self._require_available(save_id)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, state, *, save_id=None, name=None):
        now = datetime.now(timezone.utc)
        try:
            created_at = self.load(save_id).created_at if save_id else now
        except FileNotFoundError:
            # A save removed outside the app can be recreated from the live world.
            created_at = now
        overwrite = save_id is not None
        save_id = save_id if overwrite else (name if name is not None else self.next_name())
        if not overwrite:
            self._require_available(save_id)
        save = SimulationSave(name=save_id, created_at=created_at, updated_at=now, state=state)
        self._write(save_id, save, overwrite=overwrite)
        return save_id

    def rename(self, save_id, name):
        old_path = self._path(save_id)
        new_path = self._path(name)
        self._require_available(name, exclude=old_path)
        old_path.rename(new_path)
        return name

    def delete(self, save_id):
        self._path(save_id).unlink()


def capture_simulation(app):
    """Capture main-thread state, including accepted actions awaiting a tick."""
    npc_indices = {id(npc): index for index, npc in enumerate(app.npcs)}
    records = {id(record): index for index, record in enumerate(app.conversation_history)}
    history = [asdict(record) for record in app.conversation_history]
    if app.request_history is not None and id(app.request_history) in records:
        record = history[records[id(app.request_history)]]
        record["status"] = "Reply pending when saved"
        record["outcomes"] = ["The reply was not saved. Send the request again after loading."]
    npcs, conversations, gifts = [], [], []
    for npc in app.npcs:
        npcs.append(NPCState(
            **npc.activity_state(), id=npc.id,
            character=npc.character, display_name=npc.display_name, x=npc.x, y=npc.y,
            reply=npc.reply, emotion=EmotionReaction(name=npc.emotion, intensity=npc.emotion_intensity),
            social=npc.social,
            inventory=npc.inventory.copy(), inventory_capacity=npc.inventory_capacity,
            preferences=npc.preferences.copy(), command_revision=npc.command_revision,
            memories=[asdict(memory) for memory in npc.memories],
            goal=asdict(npc.goal) if npc.goal else None,
            pending_social_opinions=[asdict(effect) for effect in npc.pending_social_opinions],
            beliefs=[asdict(belief) for belief in npc.beliefs],
        ))
        gift = npc.gift
        if gift is not None and gift.initiator is npc:
            gifts.append(GiftState(
                initiator=npc_indices[id(npc)], recipient=npc_indices[id(gift.recipient)],
                record=records[id(gift.record)], item=gift.item, quantity=gift.quantity,
                arrived=gift.arrived, elapsed=gift.elapsed,
                transferred=gift.transferred, outcome_event_id=gift.outcome_event_id,
                thanks=gift.thanks,
                saved_states=[ActivityState(**activity) for activity in gift.saved_states],
            ))
        conversation = npc.conversation
        if conversation is None or conversation.initiator is not npc:
            continue
        conversations.append(ConversationState(
            id=conversation.id, origin=conversation.origin, purpose=conversation.purpose,
            focus_claim_id=conversation.focus_claim_id,
            dialogue_state=conversation.dialogue_state, wait_elapsed=conversation.wait_elapsed,
            talking_remaining=conversation.talking_remaining, food_outcome=conversation.food_outcome,
            food_reason=conversation.food_reason, handoff_thanks=conversation.handoff_thanks,
            outcome_event_id=conversation.outcome_event_id, reactions=conversation.reactions,
            initiator=npc_indices[id(npc)], recipient=npc_indices[id(conversation.recipient)],
            record=records[id(conversation.record)], topic=conversation.topic,
            arrived=conversation.arrived, turns=conversation.turns,
            saved_states=[ActivityState(**activity) for activity in conversation.saved_states],
            pending_opinions=[(npc_indices[id(person)], opinions) for person, opinions in conversation.pending_opinions],
            proposals=[asdict(p) for p in conversation.proposals], accepted_proposals=conversation.accepted_proposals,
            social_effects_applied=conversation.social_effects_applied,
            context_references=conversation.context_references.copy(),
            information_context=asdict(conversation.information_context) if conversation.information_context else None,
            accepted_disclosures=conversation.accepted_disclosures, third_party_opinions=conversation.third_party_opinions,
            leverage_uses=conversation.leverage_uses,
        ))
    world = app.simulation
    return SimulationState(
        day=world.day, phase=world.phase,
        events=[asdict(event) for event in world.events],
        promises=[asdict(promise) for promise in world.promises],
        facts=[asdict(fact) for fact in world.facts], claims=[asdict(claim) for claim in world.claims],
        disclosures=[asdict(disclosure) for disclosure in world.disclosures],
        social_opportunity=world.social_opportunity, social_rotation=world.social_rotation,
        snake=SnakeState(x=world.snake.x, y=world.snake.y, destination=world.snake.destination,
                         facing=world.snake.facing, pause_remaining=world.snake.pause_remaining,
                         animation_elapsed=world.snake.animation_elapsed),
        npcs=npcs, history=history, conversations=conversations, gifts=gifts,
        tree=TreeState(id=app.tree.id, x=app.tree.x, y=app.tree.y, fruit=app.tree.fruit,
                       capacity=app.tree.capacity,
                       shake_elapsed=app.tree.shake_elapsed,
                       shaker=app.tree.shaker.id if app.tree.shaker is not None else None,
                       reserved_by=app.tree.reserved_by.id if app.tree.reserved_by is not None else None),
        pending_actions=[asdict(action) for action in world.pending_actions],
        camera_offset=tuple(app.game.camera_offset),
        camera_zoom=app.game.camera_zoom,
    )


def restore_simulation(state):
    """Build and validate a complete world before replacing the current simulation."""
    state = SimulationState.model_validate(state.model_dump())
    npcs = []
    for saved in state.npcs:
        npc = NPC(
            id=saved.id, character=saved.character, display_name=saved.display_name, x=saved.x, y=saved.y,
            reply=saved.reply, emotion=saved.emotion.name, emotion_intensity=saved.emotion.intensity,
            social=saved.social,
            inventory=saved.inventory.copy(), inventory_capacity=saved.inventory_capacity,
            preferences=saved.preferences.model_dump(), command_revision=saved.command_revision,
            memories=[Memory(**memory.model_dump()) for memory in saved.memories],
            goal=Goal(**saved.goal.model_dump()) if saved.goal else None,
            pending_social_opinions=[OpinionEffect(**effect.model_dump()) for effect in saved.pending_social_opinions],
            beliefs=[Belief(**belief.model_dump()) for belief in saved.beliefs],
        )
        npc.restore_activity({name: getattr(saved, name) for name in ACTIVITY_FIELDS})
        npcs.append(npc)
    by_id = {npc.id: npc for npc in npcs}
    tree = Tree(id=state.tree.id, x=state.tree.x, y=state.tree.y, fruit=state.tree.fruit,
                capacity=state.tree.capacity,
                shake_elapsed=state.tree.shake_elapsed, shaker=by_id.get(state.tree.shaker),
                reserved_by=by_id.get(state.tree.reserved_by))
    world = Simulation(npcs, tree, assign_goals=False)
    world.snake = Snake(**state.snake.model_dump())
    world.day, world.phase = state.day, state.phase
    world.pending_actions = [PendingAction(**action.model_dump()) for action in state.pending_actions]
    for action in world.pending_actions:
        by_id[action.npc_id].request_pending = True
    history = [ConversationRecord(**saved.model_dump()) for saved in state.history]
    world.history = history
    world.events = [WorldEvent(**event.model_dump()) for event in state.events]
    world.promises = [Promise(**promise.model_dump()) for promise in state.promises]
    world.facts = [Fact(**fact.model_dump()) for fact in state.facts]
    world.claims = [Claim(**claim.model_dump()) for claim in state.claims]
    world.disclosures = [Disclosure(**disclosure.model_dump()) for disclosure in state.disclosures]
    world.social_opportunity, world.social_rotation = state.social_opportunity, state.social_rotation
    for saved in state.conversations:
        # Loading bypasses __init__: it would interrupt the saved activities again.
        conversation = Conversation.__new__(Conversation)
        for name in ("id", "origin", "purpose", "dialogue_state", "wait_elapsed", "talking_remaining",
                     "food_outcome", "food_reason", "handoff_thanks", "outcome_event_id", "reactions",
                     "accepted_proposals", "social_effects_applied", "context_references",
                     "accepted_disclosures", "third_party_opinions", "leverage_uses", "focus_claim_id"):
            setattr(conversation, name, getattr(saved, name))
        conversation.initiator, conversation.recipient = npcs[saved.initiator], npcs[saved.recipient]
        conversation.topic, conversation.arrived, conversation.active = saved.topic, saved.arrived, True
        conversation.record = history[saved.record]
        conversation.turns = saved.turns
        conversation.proposals = [Proposal(**proposal.model_dump()) for proposal in saved.proposals]
        conversation.information_context = information_from_dict(saved.information_context.model_dump()) if saved.information_context else None
        conversation.pending_opinions = [(npcs[index], opinions) for index, opinions in saved.pending_opinions]
        conversation.saved_states = [activity.model_dump() for activity in saved.saved_states]
        for npc in (conversation.initiator, conversation.recipient):
            npc.conversation = conversation
        if conversation.dialogue_state == "pending":
            conversation.fail_dialogue("The outstanding dialogue request was not saved. No dialogue was generated.")
    for saved in state.gifts:
        gift = Gift.__new__(Gift)
        gift.initiator, gift.recipient = npcs[saved.initiator], npcs[saved.recipient]
        gift.item, gift.quantity = saved.item, saved.quantity
        gift.arrived, gift.elapsed = saved.arrived, saved.elapsed
        gift.active, gift.transferred, gift.outcome_event_id = True, saved.transferred, saved.outcome_event_id
        gift.thanks = saved.thanks
        gift.record = history[saved.record]
        gift.saved_states = [activity.model_dump() for activity in saved.saved_states]
        for npc in (gift.initiator, gift.recipient):
            npc.gift = gift
    # Claims are reacquired through the same safe path as interrupted activities.
    tree.reserved_by = None
    tree.cancel_shake()
    for npc in npcs:
        if npc.conversation is None and npc.gift is None and npc.target_id == tree.id:
            npc.resume_pending = True
    return world, history
