"""Self-contained simulation saves. No character file paths are loaded."""

from dataclasses import asdict, dataclass
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Literal
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from character_data import normalize_character, normalize_opinions
from constants import (
    APPLE_ADVANCE_SECONDS, DECISION_SECONDS, DIALOGUE_TIMEOUT_SECONDS, EAT_SECONDS, GIFT_GIVE_SECONDS, GIFT_SECONDS,
    MAX_SIMULATION_STEP_SECONDS, SHAKE_SECONDS, SNAKE_EAT_SECONDS,
    POST_DIALOGUE_SECONDS, RECOVERY_SECONDS, SNAKE_FRAME_SECONDS, SNAKE_NAME, WORLD_HEIGHT, WORLD_WIDTH,
)
from conversation_history import ConversationRecord, Memory, WorldEvent
from llm import (AcceptedProposal, ConversationPart, ConversationTurn, DisclosureChoice, NameIntroduction,
                 EmotionReaction, LeverageUse, ThirdPartyOpinion)
from npc import ACTIVITY_FIELDS, Behavior, Conversation, Gift, NPC, Tree
from simulation import PendingAction, Simulation
from snake import Snake, SnakeEncounter
from purgatory import Attitudes
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
                  "claim_shared", "heard_claim", "claim_opinion", "leverage", "arrival", "bargain", "endorsement", "defiance", "snake_claim", "judgment", "pardon", "eating", "goal_cancelled"]
    summary: str = Field(max_length=400)
    weight: float = Field(ge=-12, le=12)


class EventState(SaveModel):
    id: str = Field(min_length=1)
    day: int = Field(ge=1)
    phase: Literal["day", "evening", "night"]
    kind: Literal["phase", "food_requested", "food_refused", "fruit_shared", "conversation_completed", "interaction_failed",
                  "promise_created", "promise_fulfilled", "promise_broken", "goal_completed", "goal_failed",
                  "claim_shared", "claim_assessed", "reputation_changed", "leverage_used", "arrival", "bargain", "endorsement", "defiance", "snake_claim", "judgment", "pardon", "eating", "goal_cancelled", "promise_cancelled"]
    participants: list[str]
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
    status: Literal["active", "completed", "failed", "cancelled"]
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
    status: Literal["pending", "fulfilled", "broken", "cancelled"]
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
    witness_ids: list[str] = Field(min_length=1)


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


class DepartureState(SaveModel):
    day: int = Field(ge=1)
    phase: Literal["day", "evening", "night"]
    cause: Literal["eaten by the snake"]
    event_id: str = Field(min_length=1)


class LearnedNameState(SaveModel):
    name: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    turn: int = Field(ge=0)


class NPCState(ActivityState):
    id: str = Field(min_length=1)
    character: dict
    display_name: str | None
    x: float = Field(ge=0, le=WORLD_WIDTH)
    y: float = Field(ge=0, le=WORLD_HEIGHT)
    reply: str
    rewritten_reply: str = ""
    emotion: EmotionReaction
    social: float = Field(ge=0, le=100)
    inventory: dict[Literal["fruit"], int]
    inventory_capacity: int = Field(ge=1)
    preferences: PreferencesState
    memories: list[MemoryState] = Field(max_length=20)
    goal: GoalState | None
    pending_social_opinions: list[OpinionEffectState]
    beliefs: list[BeliefState]
    known_names: dict[str, LearnedNameState] = Field(default_factory=dict)
    command_revision: int = Field(ge=0)
    attitudes: Attitudes
    arrival_day: int = Field(ge=1)
    departure: DepartureState | None

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
    rewritten_parts: dict[int, list[str]] = Field(default_factory=dict)
    failed_turns: list[ConversationTurn] = Field(default_factory=list)
    name_forms: list[str] = Field(default_factory=list)
    name_introductions: dict[int, list[NameIntroduction]] = Field(default_factory=dict)
    name_witnesses: dict[int, list[str]] = Field(default_factory=dict)
    orientation_discussed: bool = False
    dialogue_errors: list[str] = Field(default_factory=list)
    outcomes: list[str]
    error: str
    origin: Literal["player", "autonomous"]
    purpose: Literal["chat", "orientation", "food_request", "gossip", "bargain", "judgment", "introduction", "snake_player"]

    @model_validator(mode="after")
    def validate_rewrites(self):
        for index, texts in self.rewritten_parts.items():
            if (index < 0 or index >= len(self.turns) or self.turns[index][0] == "You"
                    or len(texts) != len(self.dialogue_parts.get(index, []))
                    or not texts or any(not text.strip() for text in texts)):
                raise ValueError("A rewrite must match the original NPC dialogue parts.")
        return self


class ConversationState(SaveModel):
    id: str = Field(min_length=1)
    origin: Literal["player", "autonomous"]
    purpose: Literal["chat", "orientation", "food_request", "gossip"]
    focus_claim_id: str | None
    dialogue_state: Literal["waiting", "pending", "ready"]
    wait_elapsed: float = Field(ge=0, le=DIALOGUE_TIMEOUT_SECONDS + MAX_SIMULATION_STEP_SECONDS)
    talking_remaining: float = Field(ge=0, le=max(POST_DIALOGUE_SECONDS, DECISION_SECONDS))
    food_outcome: Literal["", "accepted", "need", "refused", "unavailable"]
    food_reason: str
    handoff_thanks: list[ConversationPart] = Field(max_length=3)
    rewritten_handoff_thanks: list[str] = Field(default_factory=list)
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
    context_references: dict[str, Literal["active", "completed", "failed", "pending", "fulfilled", "broken", "cancelled"]]
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
    elapsed: float = Field(ge=0, lt=GIFT_SECONDS)
    transferred: bool
    outcome_event_id: str | None
    thanks: list[ConversationPart] = Field(max_length=3)
    rewritten_thanks: list[str] = Field(default_factory=list)
    saved_states: tuple[ActivityState, ActivityState]


class TreeState(SaveModel):
    id: str = Field(min_length=1)
    x: float = Field(ge=60, le=WORLD_WIDTH)
    y: float = Field(ge=0, le=WORLD_HEIGHT - 50)
    fruit: int = Field(ge=0)
    shake_elapsed: float | None = Field(ge=0, le=SHAKE_SECONDS)
    shaker: str | None
    reserved_by: str | None


class SnakeRetryState(SaveModel):
    resident_id: str
    purpose: Literal["bargain", "judgment", "introduction"]
    reason: str
    origin: Literal["player", "autonomous"]
    error: str


class AppleAdvanceState(SaveModel):
    offering: Literal["secret", "endorsement"]
    account_id: str
    conversation_id: str
    remaining: float = Field(ge=0, le=APPLE_ADVANCE_SECONDS)


class SnakeState(SaveModel):
    id: str = Field(min_length=1)
    x: float = Field(ge=0, le=WORLD_WIDTH)
    y: float = Field(ge=0, le=WORLD_HEIGHT)
    destination: tuple[float, float] | None
    facing: int = Field(ge=0, le=7)
    pause_remaining: float = Field(ge=0)
    animation_elapsed: float = Field(ge=0, lt=2 * SNAKE_FRAME_SECONDS)
    activity: Literal["guarding", "recovering", "judging", "bargaining", "introducing", "approaching condemned resident", "eating"]
    target_id: str | None
    beliefs: list[BeliefState]
    standing: dict[str, int]
    offenses: dict[str, str]
    advances: dict[str, AppleAdvanceState] = Field(default_factory=dict)
    eaten_today: int = Field(ge=0)
    recovery_remaining: float = Field(ge=0, le=RECOVERY_SECONDS)
    pardons: list[str]
    judged_phase: int = Field(ge=-1)
    decision_remaining: float = Field(ge=0)
    command_revision: int = Field(ge=0)
    reply: str
    rewritten_reply: str = ""
    retry_request: SnakeRetryState | None

    @field_validator("destination")
    @classmethod
    def validate_destination(cls, value):
        if value is not None and not (0 <= value[0] <= WORLD_WIDTH and 0 <= value[1] <= WORLD_HEIGHT):
            raise ValueError("Invalid snake destination.")
        return value


class SnakeEncounterState(SaveModel):
    id: str
    resident_id: str
    record: int = Field(ge=0)
    purpose: Literal["bargain", "judgment", "introduction"]
    origin: Literal["player", "autonomous"]
    topic: str
    accusation: str
    saved_state: ActivityState
    reserved: bool
    arrived: bool
    dialogue_state: Literal["waiting", "pending", "ready"]
    wait_elapsed: float = Field(ge=0, le=DIALOGUE_TIMEOUT_SECONDS + MAX_SIMULATION_STEP_SECONDS)
    talking_remaining: float = Field(ge=0, le=max(POST_DIALOGUE_SECONDS, DECISION_SECONDS))
    committed: bool
    outcome: Literal["", "talk", "advance_secret", "advance_endorsement", "secret", "endorsement", "refuse", "negotiate", "challenge", "spare", "eat"]
    outcome_event_id: str | None
    stage: Literal["approach", "waiting", "dialogue", "condemned_approach", "eating"]
    elapsed: float = Field(ge=0, le=SNAKE_EAT_SECONDS + MAX_SIMULATION_STEP_SECONDS)
    turns: list[ConversationTurn]
    request_context: dict | None


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
    npc_trait_mode: Literal["normal", "extreme", "both"] = "normal"
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
    snake: SnakeState
    archived: list[NPCState]
    arrival_error: str
    pending_arrival_days: list[int]
    processed_arrival_days: list[int]
    introductions: list[str]
    roster_revision: int = Field(ge=0)
    snake_encounter: SnakeEncounterState | None
    pending_actions: list[PendingActionState]
    history: list[HistoryState]
    conversations: list[ConversationState]
    gifts: list[GiftState] = Field(default_factory=list)
    camera_offset: tuple[float, float]
    camera_zoom: float = Field(default=1.0, ge=0.5, le=2.5)

    @model_validator(mode="after")
    def validate_references(self):
        residents = [*self.npcs, *self.archived]
        live_ids = {n.id for n in self.npcs}
        actors = {n.id: (n.display_name or n.character["Name"]) for n in residents}
        actors[self.snake.id] = SNAKE_NAME
        authored_names = {actors[n.id]: n.character["Name"] for n in residents}
        authored_names[SNAKE_NAME] = SNAKE_NAME
        ids = set(actors)
        now = phase_index(self.day, self.phase)
        if len(ids) != len(residents) + 1 or self.tree.id in ids or len({v.casefold() for v in actors.values()}) != len(ids):
            raise ValueError("Actor IDs and stable display names must be unique.")
        if self.social_rotation >= len(self.npcs):
            raise ValueError("Invalid social rotation.")
        if any(n.departure is not None for n in self.npcs) or any(n.departure is None for n in self.archived):
            raise ValueError("Live and departed rosters disagree.")
        history = {h.id: h for h in self.history}
        events = {e.id: e for e in self.events}
        facts = {f.id: f for f in self.facts}
        claims = {c.id: c for c in self.claims}
        disclosures = {d.id: d for d in self.disclosures}
        for index, source in ((history, self.history), (events, self.events), (facts, self.facts), (claims, self.claims), (disclosures, self.disclosures)):
            if len(index) != len(source):
                raise ValueError("Duplicate social record IDs.")
        if len({(f.source_id, f.kind) for f in self.facts}) != len(self.facts):
            raise ValueError("Duplicate fact sources.")
        def record_for(record_id):
            record = history.get(record_id)
            if record is None:
                raise ValueError("Missing source conversation.")
            return record
        def participant(actor_id, record):
            return actor_id in ids and actors[actor_id] in record.participants
        from name_knowledge import asks_name, introduction_supported
        def introduction_evidence(record, index, intro):
            previous = record.turns[index - 1] if index else None
            requested = bool(previous and previous[0] != record.turns[index][0]
                and asks_name(previous[1], self_name=authored_names.get(previous[0], "")))
            return introduction_supported(record.turns[index][1], intro.name, intro.reference, name_requested=requested)
        for event in self.events:
            if not set(event.participants) <= ids or phase_index(event.day, event.phase) > now:
                raise ValueError("Invalid event actors or date.")
            if event.history is not None and event.history >= len(self.history):
                raise ValueError("Invalid event history link.")
        for record in self.history:
            if any(name != "You" and name not in actors.values() for name in record.participants):
                raise ValueError("Unknown historical speaker.")
            if any(speaker not in record.participants or not text.strip() for speaker, text in record.turns):
                raise ValueError("Invalid historical speech.")
            for index, parts in record.dialogue_parts.items():
                if index >= len(record.turns) or " ".join(p.text.strip() for p in parts) != record.turns[index][1]:
                    raise ValueError("Dialogue parts do not match their recorded turn.")
            for index, introductions in record.name_introductions.items():
                if not 0 <= index < len(record.turns):
                    raise ValueError("Invalid introduction turn.")
                # Diagnosed metadata is retained even when its name effect was
                # unusable. Learned-name references are checked separately below.
                if record.dialogue_errors:
                    continue
                if len({i.actor_id for i in introductions}) != len(introductions):
                    raise ValueError("Invalid introduction turn or duplicate introduced actor.")
                for intro in introductions:
                    missing_speech = not introduction_evidence(record, index, intro)
                    if (intro.actor_id not in ids or missing_speech
                            or intro.reference == "self" and record.turns[index][0] != actors[intro.actor_id]):
                        raise ValueError("Invalid introduction identity or spoken evidence.")
            for index, witnesses in record.name_witnesses.items():
                if (not 0 <= index < len(record.turns) or not set(witnesses) <= ids
                        or not all(any((f.kind == "endorsement" and index == 1 or f.kind == "snake_claim" and index == 0)
                                       and f.conversation_id == record.id
                                       and witness in f.witness_ids for f in self.facts) for witness in witnesses)):
                    raise ValueError("A witnessed introduction requires a recorded hearing event.")
        for fact in self.facts:
            if (fact.subject_id not in ids or fact.object_id not in ids or fact.subject_id == fact.object_id
                    or fact.phase > now or not set(fact.witness_ids) <= ids
                    or len(set(fact.witness_ids)) != len(fact.witness_ids)):
                raise ValueError("Invalid fact participants or witnesses.")
            if fact.kind in ("private_motive", "former_life"):
                if fact.witness_ids != [fact.subject_id] or fact.conversation_id is not None:
                    raise ValueError("A private recollection or motive cannot be public.")
            elif fact.conversation_id is not None:
                record_for(fact.conversation_id)
            if fact.source_id in events:
                event = events[fact.source_id]
                expected = {"apple_given": "fruit_shared", "apple_refused": "food_refused", "promise_kept": "promise_fulfilled"}.get(fact.kind, fact.kind)
                if event.kind != expected or fact.phase != phase_index(event.day, event.phase):
                    raise ValueError("A fact must match its actual source event.")
        for claim in self.claims:
            fact = facts.get(claim.fact_id)
            if fact is None or claim.subject_id != fact.subject_id or claim.originator_id not in ids or not fact.phase <= claim.phase <= now:
                raise ValueError("Invalid claim references.")
            if claim.parent_claim_id is None:
                if not claim.affirmed or claim.deliberate or claim.originator_id not in fact.witness_ids:
                    raise ValueError("Firsthand accounts require an actual witnessed source.")
            else:
                parent = claims.get(claim.parent_claim_id)
                if parent is None or parent.fact_id != fact.id or not parent.affirmed or claim.affirmed or not claim.deliberate or parent.phase > claim.phase:
                    raise ValueError("A denial must trace to an existing account.")
        seen = set()
        for disclosure in self.disclosures:
            record = record_for(disclosure.conversation_id)
            if (disclosure.claim_id not in claims or disclosure.speaker_id == disclosure.listener_id
                    or not participant(disclosure.speaker_id, record) or disclosure.listener_id not in ids
                    or disclosure.turn >= len(record.turns) or record.turns[disclosure.turn][0] != actors[disclosure.speaker_id]
                    or disclosure.phase > now):
                raise ValueError("Invalid disclosure actor, speech or account.")
            if disclosure.source_disclosure_id:
                source = disclosures.get(disclosure.source_disclosure_id)
                claim = claims[disclosure.claim_id]
                if source is None or source.id not in seen or source.listener_id != disclosure.speaker_id or source.claim_id not in (claim.id, claim.parent_claim_id):
                    raise ValueError("Disclosure provenance must follow an earlier source.")
            elif disclosure.speaker_id != claims[disclosure.claim_id].originator_id:
                fact = facts[claims[disclosure.claim_id].fact_id]
                if disclosure.speaker_id not in fact.witness_ids:
                    raise ValueError("A repeated account requires its actual source.")
            seen.add(disclosure.id)
        for actor in [*residents, self.snake]:
            if len({b.claim_id for b in actor.beliefs}) != len(actor.beliefs):
                raise ValueError("Duplicate owned account.")
            for belief in actor.beliefs:
                claim = claims.get(belief.claim_id)
                if claim is None or belief.npc_id != actor.id or belief.source_id not in ids or belief.phase > now:
                    raise ValueError("Invalid owned knowledge.")
                if belief.direct:
                    if actor.id not in facts[claim.fact_id].witness_ids or belief.disclosure_id is not None:
                        raise ValueError("Firsthand knowledge requires witnessing.")
                elif belief.disclosure_id:
                    source = disclosures.get(belief.disclosure_id)
                    if source is None or source.listener_id != actor.id or source.claim_id != claim.id or source.speaker_id != belief.source_id:
                        raise ValueError("Hearsay requires its actual disclosure source.")
        goal_ids = set()
        for npc in residents:
            from name_knowledge import forms
            for target_id, learned in npc.known_names.items():
                learned = learned.model_dump()
                target = next((n for n in residents if n.id == target_id), None)
                actual_name = SNAKE_NAME if target_id == self.snake.id else target.character["Name"] if target else ""
                if (target_id == npc.id or not actual_name or set(learned) != {"name", "conversation_id", "turn"}
                        or learned["name"] not in forms(actual_name)):
                    raise ValueError("Invalid learned name or actor reference.")
                source = record_for(learned["conversation_id"])
                turn = learned["turn"]
                if (type(turn) is not int or not 0 <= turn < len(source.turns)
                        or actors[npc.id] not in source.participants and npc.id not in source.name_witnesses.get(turn, [])):
                    raise ValueError("A learned name requires accepted spoken source evidence.")
                if not any(intro.actor_id == target_id and intro.name == learned["name"]
                           and (source.dialogue_errors or introduction_evidence(source, turn, intro))
                           and (intro.reference != "self" or source.turns[turn][0] == actors[target_id])
                           and source.turns[turn][0] != actors[npc.id]
                           for intro in source.name_introductions.get(turn, [])):
                    raise ValueError("A learned name must be supported by its source speech.")
            if npc.arrival_day > self.day or any(m.partner_id not in ids or phase_index(m.day, m.phase) > now for m in npc.memories):
                raise ValueError("Invalid arrival day or personal memory.")
            if npc.departure and (phase_index(npc.departure.day, npc.departure.phase) > now or not any(f.source_id == npc.departure.event_id and f.kind == "eating" and f.object_id == npc.id for f in self.facts)):
                raise ValueError("An archived resident needs its actual departure record.")
            goal = npc.goal
            if goal:
                if goal.id in goal_ids or goal.target_id not in ids or goal.third_npc_id is not None and goal.third_npc_id not in ids or goal.created_phase > now or goal.deadline < goal.created_phase:
                    raise ValueError("Invalid goal references.")
                goal_ids.add(goal.id)
                if goal.status == "active" and (npc.id not in live_ids or goal.target_id not in live_ids or goal.third_npc_id is not None and goal.third_npc_id not in live_ids):
                    raise ValueError("An active goal can target only living residents.")
                if (goal.status == "active") != (goal.resolved_phase is None):
                    raise ValueError("Invalid goal resolution.")
                if goal.claim_id is not None and goal.claim_id not in claims:
                    raise ValueError("Unknown goal account.")
            for effect in npc.pending_social_opinions:
                if effect.target_id not in ids or effect.claim_id is not None and effect.claim_id not in claims:
                    raise ValueError("Invalid queued social effect.")
        promise_ids = set()
        for promise in self.promises:
            record = record_for(promise.conversation_id)
            if promise.id in promise_ids or promise.promisor_id == promise.recipient_id or not all(participant(i, record) for i in (promise.promisor_id, promise.recipient_id)):
                raise ValueError("Invalid promise participants.")
            promise_ids.add(promise.id)
            if promise.status == "pending" and not {promise.promisor_id, promise.recipient_id} <= live_ids:
                raise ValueError("Impossible obligations must be cancelled.")
            if promise.deadline < promise.created_phase or promise.created_phase > now or (promise.status == "pending") != (promise.resolved_phase is None):
                raise ValueError("Invalid promise resolution.")
        def activity_target(activity, owner):
            target = activity.target_id
            if activity.behavior in (Behavior.GO_TO_TREE, Behavior.SHAKE_TREE, Behavior.SHAKING, Behavior.HARVEST):
                if target != self.tree.id:
                    raise ValueError("Invalid tree task target.")
            elif activity.behavior == Behavior.COMPANY:
                if target not in live_ids or target == owner.id:
                    raise ValueError("A company target must be living.")
            elif target is not None:
                raise ValueError("Unexpected action target.")
        reserved = set()
        for interaction in [*self.conversations, *self.gifts]:
            pair = (interaction.initiator, interaction.recipient)
            if pair[0] == pair[1] or any(i >= len(self.npcs) or i in reserved for i in pair) or interaction.record >= len(self.history):
                raise ValueError("Invalid live interaction participants.")
            for index, activity in zip(pair, interaction.saved_states):
                activity_target(activity, self.npcs[index])
            record = self.history[interaction.record]
            if set(record.participants) != {actors[self.npcs[i].id] for i in pair}:
                raise ValueError("An interaction must match its record.")
            reserved.update(pair)
            thanks = interaction.handoff_thanks if isinstance(interaction, ConversationState) else interaction.thanks
            rewritten = (interaction.rewritten_handoff_thanks if isinstance(interaction, ConversationState)
                         else interaction.rewritten_thanks)
            if rewritten and (len(rewritten) != len(thanks) or any(not text.strip() for text in rewritten)):
                raise ValueError("Rewritten thanks must match the original parts.")
            if isinstance(interaction, ConversationState):
                ready = interaction.dialogue_state == "ready"
                if record.id != interaction.id or ready != bool(interaction.turns) or ready != interaction.social_effects_applied:
                    raise ValueError("Invalid conversation commit progress.")
                if interaction.information_context:
                    context = interaction.information_context
                    if context.participants != {"initiator": self.npcs[pair[0]].id, "recipient": self.npcs[pair[1]].id}:
                        raise ValueError("Invalid information speakers.")
                    if any(o.fact_id not in facts for o in context.options):
                        raise ValueError("Missing snapshot facts.")
                validate_agreements(interaction.accepted_proposals, [p.model_dump() for p in interaction.proposals], interaction.turns, self.npcs[pair[0]].id, self.npcs[pair[1]].id)
                if ready:
                    validate_information(SimpleNamespace(turns=interaction.turns, disclosures=interaction.accepted_disclosures,
                        third_party_opinions=interaction.third_party_opinions, leverage=interaction.leverage_uses),
                        interaction.information_context.model_dump() if interaction.information_context else None)
            elif interaction.transferred != (interaction.elapsed >= GIFT_GIVE_SECONDS) or interaction.transferred != bool(interaction.outcome_event_id):
                raise ValueError("Invalid apple handoff progress.")
        encounter = self.snake_encounter
        if encounter:
            if encounter.resident_id not in live_ids or encounter.record >= len(self.history):
                raise ValueError("A snake encounter needs a living resident.")
            resident_index = next(i for i, n in enumerate(self.npcs) if n.id == encounter.resident_id)
            if resident_index in reserved or self.history[encounter.record].id != encounter.id or self.snake.target_id != encounter.resident_id:
                raise ValueError("Conflicting snake encounter reservation.")
            reserved.add(resident_index)
            if encounter.committed != bool(encounter.turns) or encounter.committed != (encounter.dialogue_state == "ready") or encounter.committed != bool(encounter.outcome):
                raise ValueError("Invalid snake encounter commit progress.")
            if encounter.reserved and (encounter.committed or encounter.purpose == "introduction" or self.tree.fruit < 1):
                raise ValueError("Invalid apple reservation.")
            if encounter.stage in ("condemned_approach", "eating") and encounter.outcome != "eat":
                raise ValueError("Eating requires a resolved condemnation.")
            activity_target(encounter.saved_state, self.npcs[resident_index])
        elif self.snake.target_id is not None:
            raise ValueError("A snake target requires an encounter.")
        if (not set(self.snake.standing) <= ids or any(not -100 <= score <= 100 for score in self.snake.standing.values())
                or not set(self.snake.pardons) <= live_ids or not set(self.snake.offenses) <= live_ids
                or any(f not in facts for f in self.snake.offenses.values())):
            raise ValueError("Invalid snake standing, offenses or pardons.")
        for resident_id, advance in self.snake.advances.items():
            record = record_for(advance.conversation_id)
            if (resident_id not in live_ids or not participant(resident_id, record)
                    or not participant(self.snake.id, record)
                    or not any(f.kind == "bargain" and f.conversation_id == record.id
                               and f.subject_id == self.snake.id and f.object_id == resident_id for f in self.facts)):
                raise ValueError("An apple advance needs its living recipient and actual handoff record.")
            if advance.offering == "secret":
                resident = next(n for n in self.npcs if n.id == resident_id)
                if advance.account_id not in claims or not any(b.claim_id == advance.account_id for b in resident.beliefs):
                    raise ValueError("A promised private account must belong to the recipient.")
            elif advance.account_id:
                raise ValueError("An endorsement advance cannot select a private account.")
        if self.snake.judged_phase > now or self.snake.retry_request and self.snake.retry_request.resident_id not in live_ids:
            raise ValueError("Invalid snake judgment phase or retry target.")
        for index, npc in enumerate(self.npcs):
            if index not in reserved:
                activity_target(npc, npc)
            if index in reserved and npc.behavior != Behavior.WAIT:
                raise ValueError("Encounter participants must wait.")
        for reference in (self.tree.reserved_by, self.tree.shaker):
            if reference is not None and reference not in live_ids:
                raise ValueError("A tree reservation requires a living owner.")
        if self.tree.shaker is not None and self.tree.shaker != self.tree.reserved_by or bool(self.tree.shaker) != (self.tree.shake_elapsed is not None):
            raise ValueError("Invalid tree shake owner.")
        pending_owners = set()
        for action in self.pending_actions:
            if action.npc_id not in live_ids or action.npc_id in pending_owners or action.target_id is not None and action.target_id not in live_ids:
                raise ValueError("Queued actions need living actors.")
            pending_owners.add(action.npc_id)
            if action.record is not None and action.record >= len(self.history):
                raise ValueError("Invalid queued history record.")
        pending, processed = self.pending_arrival_days, self.processed_arrival_days
        if pending != sorted(set(pending)) or processed != sorted(set(processed)) or set(pending) & set(processed) or set(pending) | set(processed) != set(range(2, self.day + 1)):
            raise ValueError("Every dawn must have exactly one pending or completed arrival.")
        arrivals = [n.arrival_day for n in residents if n.arrival_day >= 2]
        if sorted(arrivals) != processed:
            raise ValueError("Completed arrivals must match their preserved residents.")
        if not set(self.introductions) <= live_ids:
            raise ValueError("Invalid introduction identity.")
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


def _capture_fields(model, source, **overrides):
    """Use the save model's fields; spell out only reference/type conversions."""
    return model(**{name: overrides[name] if name in overrides else getattr(source, name)
                    for name in model.model_fields})


def _restore_fields(target, saved, *, exclude=(), **overrides):
    for name in type(saved).model_fields:
        if name not in exclude:
            setattr(target, name, overrides[name] if name in overrides else getattr(saved, name))


def capture_resident(npc):
    return _capture_fields(
        NPCState, npc, emotion=EmotionReaction(name=npc.emotion, intensity=npc.emotion_intensity),
        memories=[asdict(memory) for memory in npc.memories],
        goal=asdict(npc.goal) if npc.goal else None,
        pending_social_opinions=[asdict(effect) for effect in npc.pending_social_opinions],
        beliefs=[asdict(belief) for belief in npc.beliefs],
    )


def capture_simulation(app):
    """Capture main-thread state, including accepted actions awaiting a tick."""
    npc_indices = {id(npc): index for index, npc in enumerate(app.npcs)}
    records = {id(record): index for index, record in enumerate(app.conversation_history)}
    history = [asdict(record) for record in app.conversation_history]
    if app.request_history is not None and id(app.request_history) in records:
        record = history[records[id(app.request_history)]]
        record["status"] = "Reply pending when saved"
        record["outcomes"].insert(0, "The reply was not saved. Send the request again after loading.")
    npcs, conversations, gifts = [], [], []
    for npc in app.npcs:
        npcs.append(capture_resident(npc))
        gift = npc.gift
        if gift is not None and gift.initiator is npc:
            gifts.append(_capture_fields(GiftState, gift,
                initiator=npc_indices[id(npc)], recipient=npc_indices[id(gift.recipient)],
                record=records[id(gift.record)],
            ))
        conversation = npc.conversation
        if conversation is None or conversation.initiator is not npc:
            continue
        conversations.append(_capture_fields(ConversationState, conversation,
            initiator=npc_indices[id(npc)], recipient=npc_indices[id(conversation.recipient)],
            record=records[id(conversation.record)],
            pending_opinions=[(npc_indices[id(person)], opinions) for person, opinions in conversation.pending_opinions],
            proposals=[asdict(p) for p in conversation.proposals],
            information_context=asdict(conversation.information_context) if conversation.information_context else None,
        ))
    world = app.simulation
    encounter = world.snake.encounter
    return SimulationState(
        day=world.day, npc_trait_mode=world.npc_trait_mode, phase=world.phase,
        events=[asdict(event) for event in world.events],
        promises=[asdict(promise) for promise in world.promises],
        facts=[asdict(fact) for fact in world.facts], claims=[asdict(claim) for claim in world.claims],
        disclosures=[asdict(disclosure) for disclosure in world.disclosures],
        social_opportunity=world.social_opportunity, social_rotation=world.social_rotation,
        snake=_capture_fields(SnakeState, world.snake, beliefs=[asdict(b) for b in world.snake.beliefs]),
        archived=[capture_resident(npc) for npc in world.archived],
        arrival_error=world.arrival_error,
        pending_arrival_days=world.pending_arrival_days.copy(), processed_arrival_days=world.processed_arrival_days.copy(),
        introductions=world.introductions.copy(), roster_revision=world.roster_revision,
        snake_encounter=(_capture_fields(SnakeEncounterState, encounter,
                         resident_id=encounter.recipient.id, record=records[id(encounter.record)])
                         if encounter is not None else None),
        npcs=npcs, history=history, conversations=conversations, gifts=gifts,
        tree=_capture_fields(TreeState, app.tree,
                       shaker=app.tree.shaker.id if app.tree.shaker is not None else None,
                       reserved_by=app.tree.reserved_by.id if app.tree.reserved_by is not None else None),
        pending_actions=[asdict(action) for action in world.pending_actions],
        camera_offset=tuple(app.game.camera_offset),
        camera_zoom=app.game.camera_zoom,
    )


def restore_simulation(state):
    """Build and validate a complete world before replacing the current simulation."""
    state = SimulationState.model_validate(state.model_dump())
    all_residents = []
    for saved in [*state.npcs, *state.archived]:
        npc = NPC(
            id=saved.id, character=saved.character, display_name=saved.display_name, x=saved.x, y=saved.y,
            reply=saved.reply, emotion=saved.emotion.name, emotion_intensity=saved.emotion.intensity,
            social=saved.social,
            inventory=saved.inventory.copy(), inventory_capacity=saved.inventory_capacity,
            rewritten_reply=saved.rewritten_reply,
            preferences=saved.preferences.model_dump(), command_revision=saved.command_revision,
            memories=[Memory(**memory.model_dump()) for memory in saved.memories],
            goal=Goal(**saved.goal.model_dump()) if saved.goal else None,
            pending_social_opinions=[OpinionEffect(**effect.model_dump()) for effect in saved.pending_social_opinions],
            beliefs=[Belief(**belief.model_dump()) for belief in saved.beliefs],
            known_names=deepcopy(saved.model_dump()["known_names"]),
            attitudes=saved.attitudes.model_dump(),
            arrival_day=saved.arrival_day, departure=saved.departure.model_dump() if saved.departure else None,
        )
        npc.restore_activity({name: getattr(saved, name) for name in ACTIVITY_FIELDS})
        all_residents.append(npc)
    npcs = all_residents[:len(state.npcs)]
    by_id = {npc.id: npc for npc in npcs}
    tree = Tree(id=state.tree.id, x=state.tree.x, y=state.tree.y, fruit=state.tree.fruit,
                shake_elapsed=state.tree.shake_elapsed, shaker=by_id.get(state.tree.shaker),
                reserved_by=by_id.get(state.tree.reserved_by))
    world = Simulation(npcs, tree, assign_goals=False, npc_trait_mode=state.npc_trait_mode)
    world.archived = all_residents[len(state.npcs):]
    for npc in world.archived:
        npc.world = world
    snake_values = state.snake.model_dump()
    snake_values["beliefs"] = [Belief(**b) for b in snake_values["beliefs"]]
    world.snake = Snake(**snake_values)
    world.snake.world = world
    world.arrival_error = state.arrival_error
    world.pending_arrival_days = state.pending_arrival_days.copy()
    world.processed_arrival_days = state.processed_arrival_days.copy()
    world.introductions = state.introductions.copy()
    world.roster_revision = state.roster_revision
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
        _restore_fields(conversation, saved,
            initiator=npcs[saved.initiator], recipient=npcs[saved.recipient], record=history[saved.record],
            proposals=[Proposal(**proposal.model_dump()) for proposal in saved.proposals],
            information_context=information_from_dict(saved.information_context.model_dump()) if saved.information_context else None,
            pending_opinions=[(npcs[index], opinions) for index, opinions in saved.pending_opinions],
            saved_states=[activity.model_dump() for activity in saved.saved_states])
        conversation.active = True
        for npc in (conversation.initiator, conversation.recipient):
            npc.conversation = conversation
        if conversation.dialogue_state == "pending":
            conversation.dialogue_state = "waiting"
            conversation.wait_elapsed = 0.0
    for saved in state.gifts:
        gift = Gift.__new__(Gift)
        _restore_fields(gift, saved,
            initiator=npcs[saved.initiator], recipient=npcs[saved.recipient], record=history[saved.record],
            saved_states=[activity.model_dump() for activity in saved.saved_states])
        gift.active = True
        for npc in (gift.initiator, gift.recipient):
            npc.gift = gift
    saved = state.snake_encounter
    if saved:
        encounter = SnakeEncounter.__new__(SnakeEncounter)
        _restore_fields(encounter, saved, exclude=("resident_id",),
                        record=history[saved.record], saved_state=saved.saved_state.model_dump())
        encounter.world = world
        encounter.initiator, encounter.recipient = world.snake, by_id[saved.resident_id]
        encounter.active = True
        encounter.pending_opinions = []
        world.snake.encounter = encounter
        encounter.recipient.snake_encounter = encounter
        if encounter.dialogue_state == "pending":
            encounter.dialogue_state = "waiting"
            encounter.wait_elapsed = 0.0
            encounter.request_context = None
    # Claims are reacquired through the same safe path as interrupted activities.
    tree.reserved_by = None
    tree.cancel_shake()
    for npc in npcs:
        if npc.conversation is None and npc.gift is None and npc.target_id == tree.id:
            npc.resume_pending = True
    return world
