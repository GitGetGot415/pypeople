"""Session records of dialogue and its outcomes."""

from dataclasses import dataclass, field
from uuid import uuid4
from dialogue_text import contains_serialized_dialogue


@dataclass
class ConversationRecord:
    participants: tuple[str, str]
    topic: str
    status: str
    turns: list[tuple[str, str]] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    error: str = ""
    origin: str = "player"
    purpose: str = "chat"
    id: str = field(default_factory=lambda: uuid4().hex)
    dialogue_parts: dict[int, list[dict]] = field(default_factory=dict)
    rewritten_parts: dict[int, list[str]] = field(default_factory=dict)
    failed_turns: list[dict] = field(default_factory=list)

    def add_turn(self, speaker, text, parts, rewritten=()):
        if rewritten:
            self.rewritten_parts[len(self.turns)] = list(rewritten)
        self.dialogue_parts[len(self.turns)] = parts
        self.turns.append((speaker, text))


def usable_rewrite(parts, rewritten):
    """Old or malformed presentation data falls back without editing saved words."""
    return bool(rewritten) and len(rewritten) == len(parts) and all(
        isinstance(text, str) and text.strip() and not contains_serialized_dialogue(text)
        for text in rewritten)


def display_parts(parts, rewritten=(), *, original=False):
    """Select presentation text while keeping the original part's metadata."""
    use_original = uses_original_dialogue(parts, rewritten, original=original)
    return [(part[0] if use_original else rewritten[index], part[1])
            for index, part in enumerate(parts)]


def uses_original_dialogue(parts, rewritten=(), *, original=False):
    """Rendering and text selection agree when a missing rewrite falls back."""
    return original or not usable_rewrite(parts, rewritten)


@dataclass
class Memory:
    id: str
    day: int
    phase: str
    partner_id: str
    kind: str
    summary: str
    weight: float


@dataclass
class WorldEvent:
    day: int
    phase: str
    kind: str
    participants: list[str]
    text: str
    history: int | None = None
    id: str = field(default_factory=lambda: uuid4().hex)


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
    pair = {conversation.initiator.name, conversation.recipient.name}
    records = {index for index, record in enumerate(world.history)
               if record.id != conversation.id and set(record.participants) == pair and len(record.turns) >= 2}
    dated = {event.history for event in world.events
             if event.history in records and event.kind in ("conversation_completed", "food_requested")}
    days = sorted({event.day for event in world.events
                   if event.history in dated and event.kind in ("conversation_completed", "food_requested")})
    # Older saves can contain replayed conversations. Feed each transcript only
    # once so duplicate records do not reinforce the model's repeated wording.
    recent_dialogue, seen = [], set()
    for index in sorted(records, reverse=True):
        excerpts = [(speaker, text[:400]) for speaker, text in world.history[index].turns]
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
