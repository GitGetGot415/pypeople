"""Session records of dialogue and its outcomes."""

from dataclasses import dataclass, field
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field

from dialogue import ConversationTurn
from dialogue_text import contains_serialized_dialogue, contains_speaker_label
from dialogue_rules import validate_rewrite_identity


@dataclass
class RecordedTurn:
    speaker: str
    parts: list[dict]
    rewritten: list[str] = field(default_factory=list)
    introductions: list[dict] = field(default_factory=list)
    witnesses: list[str] = field(default_factory=list)
    speaker_id: str | None = None

    @property
    def text(self):
        if len(self.parts) == 1 and "emotion" not in self.parts[0]:
            return self.parts[0]["text"]
        return " ".join(part["text"].strip() for part in self.parts)

    @property
    def speech(self):
        return self.speaker, self.text

    def display(self, *, original=False):
        return display_parts([(part["text"], part.get("emotion")) for part in self.parts],
                             self.rewritten, original=original)


@dataclass
class ConversationRecord:
    participants: tuple[str, str]
    topic: str
    status: str
    turns: list[RecordedTurn] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    error: str = ""
    origin: str = "player"
    purpose: str = "chat"
    id: str = field(default_factory=lambda: uuid4().hex)
    failed_turns: list[dict] = field(default_factory=list)
    name_forms: list[str] = field(default_factory=list)
    orientation_discussed: bool = False
    dialogue_errors: list[str] = field(default_factory=list)
    intended_subjects: dict[str, str] = field(default_factory=dict)
    participant_ids: list[str] = field(default_factory=list)
    transfers: list[dict] = field(default_factory=list)
    accepted_player_turns: list[int] = field(default_factory=list)
    authorization_id: str | None = None
    transfer_intent: dict | None = None
    player_promise_offers: list[dict] = field(default_factory=list)

    @property
    def transcript(self):
        return [turn.speech for turn in self.turns]

    def includes(self, actor):
        return actor.id in self.participant_ids if self.participant_ids else actor.name in self.participants

    def between(self, first, second):
        return (set(self.participant_ids) == {first.id, second.id} if self.participant_ids else
                set(self.participants) == {first.name, second.name})

    def add_turn(self, speaker, text, parts=None, rewritten=(), *, introductions=(), witnesses=(), speaker_id=None):
        if speaker_id is None and self.participant_ids and speaker in self.participants:
            speaker_id = self.participant_ids[self.participants.index(speaker)]
        turn = RecordedTurn(speaker, parts if parts is not None else [{"text": text}],
                            list(rewritten), list(introductions), list(witnesses), speaker_id)
        self.turns.append(turn)
        return turn

    def generated_turns(self):
        """Project stored speech into the speaker roles used by effect validators."""
        return [ConversationTurn(speaker="initiator" if turn.speaker == self.participants[0] else "recipient",
                                 parts=turn.parts, introductions=turn.introductions) for turn in self.turns]


def usable_rewrite(parts, rewritten, *, speaker="", listener="", names=(), introductions=()):
    """Whether generated strings can be associated with the original display parts."""
    return bool(rewritten) and len(rewritten) == len(parts) and all(
        isinstance(text, str) and text.strip() for text in rewritten)


def _part_text(part):
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        return part["text"]
    return part[0] if isinstance(part, tuple) else part.text


def rewrite_validation_error(parts, rewritten, *, speaker="", listener="", names=(), introductions=()):
    """Report a failed rewrite check without replacing any generated words."""
    if not rewritten:
        return ""
    if not usable_rewrite(parts, rewritten):
        return "A rewrite needs one nonempty text string per original part."
    if any(contains_serialized_dialogue(text) for text in rewritten):
        return "The voice rewrite contains serialized JSON or response wrappers instead of spoken words."
    if any(contains_speaker_label(text) for text in rewritten):
        return "The voice rewrite contains narrator or speaker labels."
    if speaker:
        try:
            validate_rewrite_identity(" ".join(_part_text(part) for part in parts), " ".join(rewritten),
                                      speaker=speaker, listener=listener, names=names, introductions=introductions)
        except ValueError as exc:
            return str(exc)
    return ""


def display_parts(parts, rewritten=(), *, original=False, speaker="", listener="", names=(), introductions=()):
    """Select presentation text while keeping the original part's metadata."""
    use_original = uses_original_dialogue(parts, rewritten, original=original, speaker=speaker, listener=listener,
                                         names=names, introductions=introductions)
    return [(part[0] if use_original else rewritten[index], part[1])
            for index, part in enumerate(parts)]


def voice_rewrite_errors(record):
    """Return each affected turn, speaker and reason without changing stored words."""
    errors = []
    for index, turn in enumerate(record.turns):
        error = rewrite_validation_error(
            turn.parts, turn.rewritten, speaker=turn.speaker,
            listener=next((name for name in record.participants if name != turn.speaker), "You"),
            names=record.name_forms, introductions=turn.introductions)
        if error:
            errors.append((index, turn.speaker, error))
    return errors


def has_failed_voice_rewrite(record):
    """Diagnose retained rewrites, including those loaded from saved history."""
    return bool(voice_rewrite_errors(record))


def uses_original_dialogue(parts, rewritten=(), *, original=False, speaker="", listener="", names=(), introductions=()):
    """Content diagnostics never gate display; only availability or the user's toggle does."""
    return original or not usable_rewrite(parts, rewritten)


@dataclass
class Memory:
    id: Annotated[str, Field(min_length=1)]
    day: Annotated[int, Field(ge=1)]
    phase: Literal["day", "night"]
    partner_id: str
    kind: Literal["gave_fruit", "received_fruit", "refused", "kept_fruit", "conversation",
                  "promise_created", "promise_fulfilled", "promise_broken", "goal_completed", "goal_failed",
                  "claim_shared", "heard_claim", "claim_opinion", "leverage", "arrival", "bargain", "endorsement", "defiance", "snake_claim", "judgment", "pardon", "eating", "goal_cancelled", "gave_work", "received_work", "visit"]
    summary: Annotated[str, Field(max_length=400)]
    weight: Annotated[float, Field(ge=-12, le=12)]


@dataclass
class WorldEvent:
    day: Annotated[int, Field(ge=1)]
    phase: Literal["day", "night"]
    kind: Literal["phase", "food_requested", "food_refused", "fruit_shared", "conversation_completed", "interaction_failed",
                  "promise_created", "promise_fulfilled", "promise_broken", "goal_completed", "goal_failed",
                  "claim_shared", "claim_assessed", "reputation_changed", "leverage_used", "arrival", "bargain", "endorsement", "defiance", "snake_claim", "judgment", "pardon", "eating", "goal_cancelled", "promise_cancelled", "work_shared", "player_call", "player_conversation", "tree_queue", "travel_failed", "visit"]
    participants: list[str]
    text: str
    history: Annotated[int | None, Field(ge=0)] = None
    id: Annotated[str, Field(min_length=1)] = field(default_factory=lambda: uuid4().hex)


def memory_bias(npc, partner_id):
    """Recent personal experiences contribute at most twelve decision points."""
    memories = [memory for memory in npc.memories if memory.partner_id == partner_id]
    return max(-12.0, min(12.0, sum(
        memory.weight * 0.8 ** age for age, memory in enumerate(reversed(memories))
    )))


def remember(npc, partner, event, kind, summary, weight):
    if any(memory.id == event.id for memory in npc.memories):
        return
    npc.memories.append(Memory(event.id, event.day, event.phase, partner.id, kind, summary[:400], weight))
    del npc.memories[:-20]


def relevant_memories(npc, partner_id=None):
    """Keep a small, deduplicated selection with obligations ahead of routine chat."""
    goal_target = npc.goal.target_id if npc.goal and npc.goal.status == "active" else None
    promises = [p for p in npc.world.promises if npc.id in (p.promisor_id, p.recipient_id)
                and p.status in ("pending", "broken")] if npc.world else []
    partners = {p.recipient_id if p.promisor_id == npc.id else p.promisor_id for p in promises}
    danger_ids = set()
    if npc.world:
        from social import claim_by_id, fact_by_id
        for belief in npc.beliefs:
            fact = fact_by_id(npc.world, claim_by_id(npc.world, belief.claim_id).fact_id)
            if fact.kind in ("eating", "judgment", "defiance"):
                danger_ids.add(fact.source_id)
                danger_ids.update(d.event_id for d in npc.world.disclosures if d.id == belief.disclosure_id)
    ranked = sorted(enumerate(npc.memories), key=lambda entry: (
        entry[1].kind in ("eating", "judgment", "defiance") or entry[1].id in danger_ids,
        entry[1].partner_id in partners and entry[1].kind in ("promise_created", "promise_broken"),
        entry[1].kind in ("heard_claim", "claim_opinion", "leverage"),
        entry[1].partner_id == partner_id, entry[1].partner_id == goal_target,
        abs(entry[1].weight), entry[0]), reverse=True)
    seen, selected = set(), []
    for _, memory in ranked:
        if memory.id not in seen:
            selected.append(memory)
            seen.add(memory.id)
        if len(selected) == 5:
            break
    return selected


def shared_conversation_history(conversation):
    """Describe this pair's actual earlier dialogue, excluding the current request."""
    world = conversation.initiator.world
    records = {index for index, record in enumerate(world.history)
               if record.id != conversation.id and record.between(conversation.initiator, conversation.recipient)
               and len(record.turns) >= 2}
    dated = {event.history for event in world.events
             if event.history in records and event.kind in ("conversation_completed", "food_requested")}
    days = sorted({event.day for event in world.events
                   if event.history in dated and event.kind in ("conversation_completed", "food_requested")})
    # Older saves can contain replayed conversations. Feed each transcript only
    # once so duplicate records do not reinforce the model's repeated wording.
    recent_dialogue, seen = [], set()
    for index in sorted(records, reverse=True):
        excerpts = [(turn.speaker, turn.text[:400]) for turn in world.history[index].turns]
        signature = tuple((speaker, " ".join(text.casefold().split())) for speaker, text in excerpts)
        if signature in seen:
            continue
        recent_dialogue.append({"excerpts": excerpts})
        seen.add(signature)
        if len(recent_dialogue) == 2:
            break
    recent_dialogue.reverse()
    pair_ids = {conversation.initiator.id, conversation.recipient.id}
    # These committed interactions are known to both participants. Private goals,
    # beliefs, deception, and opinion changes must stay in their owner's context.
    shared_events = [event for event in world.events if set(event.participants) == pair_ids
                     and event.kind in ("fruit_shared", "food_requested", "food_refused", "promise_created",
                                        "promise_fulfilled", "promise_broken")]
    return {"first_conversation": not records, "previous_conversation_count": len(records),
            "recorded_conversation_days": days, "has_undated_conversations": bool(records - dated),
            "recent_dialogue": recent_dialogue,
            "confirmed_events": [{"kind": event.kind, "day": event.day, "phase": event.phase, "text": event.text}
                                 for event in shared_events[-6:]]}


def describe_opinion_change(person, target, previous, opinion, reason):
    before = "Not recorded" if previous is None else f"{previous:+d}"
    change = "" if previous is None else f" ({opinion - previous:+d})"
    return f"{person}'s opinion of {target}: {before} -> {opinion:+d}{change}\nReason: {reason}"
