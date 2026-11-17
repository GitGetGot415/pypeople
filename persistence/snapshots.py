from simulation.conversation import ResidentProgress, Conversation
from simulation.npc import NPC, ResidentValues
from simulation.tree import Tree
from simulation.transfers import HandoffProgress, Handoff
from simulation.snake import Snake, SnakeValues
from dataclasses import asdict
from persistence.records import restore_record
from simulation.interactions import DialogueProgress
from constants import PLAYER_CALL_COOLDOWN, CONVERSATION_REPLY_PENDING_WHEN_SAVED
from simulation.apartments import Invitation, HostedVisit
from simulation.locations import Journey, RouteStep
from simulation.maps import clamp_point
from simulation.player_records import (
    PlayerOpinion, PlayerCall, PlayerReplyCall, PlayerReport, PlayerAcknowledgment,
)
from dialogue.history import Memory, WorldEvent
from dialogue.models import EmotionReaction
from simulation.activity import ActivitySnapshot
from simulation.actions import PendingAction
from simulation.world import Simulation
from simulation.errands import Errand
from simulation.player import Player, PlayerInteraction
from simulation.social.records import (
    Belief, Claim, ConversationConcern, Disclosure, Fact, Goal, OpinionEffect, Promise, Proposal,
)
from persistence.models import (
    ConversationState, HandoffState, HistoryState, NPCState, PlayerInteractionState, PlayerState,
    SimulationState, SnakeState, TreeState,
)


def _parse_player_interaction(value):
    if value is None:
        return None
    try:
        physical = PlayerInteraction(**value)
    except TypeError:
        raise ValueError("Invalid player interaction fields.") from None
    if not physical.capture().keys() <= value.keys():
        raise ValueError("Missing required physical interaction fields.")
    validated = PlayerInteractionState.model_validate(asdict(physical))
    return _restore_saved_activity(restore_record(PlayerInteraction, validated))


def _restore_journey(value):
    if value is None:
        return None
    return restore_record(Journey, value, steps=[restore_record(RouteStep, step) for step in value.steps])


def _restore_saved_activity(owner):
    """Convert the same nested activity value after its owner has been attached."""
    if owner.saved_state is not None:
        owner.saved_state = restore_record(ActivitySnapshot, owner.saved_state)
    return owner


def _capture_fields(model, source, **overrides):
    """Use the save model's fields; spell out only reference/type conversions."""
    return model(**{name: overrides[name] if name in overrides else getattr(source.dialogue if name in DialogueProgress.__dataclass_fields__ else source, name)
                    for name in model.model_fields})


def capture_resident(npc):
    from simulation.player import call_valid
    stale_call = npc.player_call and not call_valid(npc.world, npc, npc.player_call)
    return _capture_fields(
        NPCState, npc, emotion=EmotionReaction(name=npc.emotion, intensity=npc.emotion_intensity),
        player_call=None if stale_call else npc.player_call,
        player_call_cooldown=PLAYER_CALL_COOLDOWN if stale_call else npc.player_call_cooldown,
        resolved_player_calls=([*npc.resolved_player_calls, npc.player_call.key]
            if stale_call and not npc.unread_player_reply else npc.resolved_player_calls.copy()),
    )


def capture_simulation(world, *, camera_offset, camera_zoom):
    """Capture main-thread state, including accepted actions awaiting a tick."""
    for actor in (*world.npcs, world.snake):
        if actor.destination is not None:
            # A destination outside the walkable map must never block saving; the actor resumes at the nearest valid point.
            actor.destination = clamp_point(actor.location, actor.destination)
    records = {id(record): record.id for record in world.history}
    history = [HistoryState.capture(record) for record in world.history]
    unread_records = {npc.unread_player_reply.reference for npc in world.npcs if npc.unread_player_reply}
    for record in history:
        for offer in record.player_promise_offers:
            if offer.status == "offered" and record.id not in unread_records:
                offer.status = "closed"
    interaction = world.player_interaction
    if interaction and interaction.pending_request:
        record = next((row for row in history if row.id == interaction.record_id), None)
        if record:
            record.status = CONVERSATION_REPLY_PENDING_WHEN_SAVED
            record.outcomes.insert(0, "The reply was interrupted by saving.")
    npcs, conversations, handoffs = [], [], []
    captured_handoffs = set()
    for npc in [*world.npcs, world.snake]:
        if npc is not world.snake:
            npcs.append(capture_resident(npc))
        gift = npc.gift
        if gift is not None and id(gift) not in captured_handoffs:
            captured_handoffs.add(id(gift))
            handoffs.append(_capture_fields(HandoffState, gift,
                initiator=gift.initiator.id, recipient=gift.recipient.id,
                record=records[id(gift.record)],
            ))
        conversation = npc.conversation
        if conversation is None or conversation.initiator is not npc:
            continue
        conversations.append(_capture_fields(ConversationState, conversation,
            initiator=npc.id, recipient=conversation.recipient.id,
            record=records[id(conversation.record)],
        ))
    return SimulationState(
        day=world.day, npc_trait_mode=world.npc_trait_mode,
        mode=world.mode, player_interaction=world.player_interaction.capture() if world.player_interaction else None,
        player_invitations=world.player_invitations, player_claims=world.player_claims,
        visit=world.visit,
        player=PlayerState.model_validate(world.player), player_call_check=max(0.0, world.player_call_check),
        player_acknowledgments=world.player_acknowledgments,
        creations=world.creations,
        events=world.events,
        promises=world.promises,
        facts=world.facts, claims=world.claims,
        disclosures=world.disclosures,
        social_rotation=world.social_rotation,
        snake=_capture_fields(SnakeState, world.snake),
        archived=[capture_resident(npc) for npc in world.archived],
        arrival_error=world.arrival_error,
        pending_arrival_days=world.pending_arrival_days.copy(), processed_arrival_days=world.processed_arrival_days.copy(),
        introductions=world.introductions.copy(), roster_revision=world.roster_revision,
        npcs=npcs, history=history, conversations=conversations, handoffs=handoffs,
        tree=_capture_fields(TreeState, world.tree),
        pending_actions=world.pending_actions,
        errands=world.errands,
        camera_offset=tuple(camera_offset),
        camera_zoom=camera_zoom,
    )


def restore_simulation(state):
    """Build and validate a complete world before replacing the current simulation."""
    # Revalidate the external boundary and isolate all mutable values before actor attachment.
    state = SimulationState.model_validate(state.model_dump())
    all_residents = []
    for saved in [*state.npcs, *state.archived]:
        values = restore_record(ResidentValues, saved,
            emotion=saved.emotion.name, emotion_intensity=saved.emotion.intensity,
            journey=_restore_journey(saved.journey), preferences=saved.preferences.model_dump(),
            departure=saved.departure.model_dump() if saved.departure else None,
            memories=[restore_record(Memory, memory) for memory in saved.memories],
            goal=restore_record(Goal, saved.goal) if saved.goal else None,
            pending_social_opinions=[restore_record(OpinionEffect, effect) for effect in saved.pending_social_opinions],
            beliefs=[restore_record(Belief, belief) for belief in saved.beliefs],
            current_concern=restore_record(ConversationConcern, saved.current_concern) if saved.current_concern else None,
            player_opinion=restore_record(PlayerOpinion, saved.player_opinion),
            player_call=(restore_record(PlayerReplyCall if saved.player_call.kind == "reply" else PlayerCall,
                                        saved.player_call) if saved.player_call else None))
        npc = NPC(**vars(values))
        all_residents.append(npc)
    npcs = all_residents[:len(state.npcs)]
    by_id = {npc.id: npc for npc in npcs}
    tree = Tree(id=state.tree.id, x=state.tree.x, y=state.tree.y, fruit=state.tree.fruit)
    world = Simulation(npcs, tree, assign_goals=False, npc_trait_mode=state.npc_trait_mode, mode=state.mode, initialize_opinions=False)
    world.archived = all_residents[len(state.npcs):]
    world.player = restore_record(Player, state.player)
    world.player.journey = _restore_journey(state.player.journey)
    world.player.world = world
    world.player_invitations = [restore_record(Invitation, o) for o in state.player_invitations]
    world.player_claims = [restore_record(PlayerReport, c) for c in state.player_claims]
    world.visit = restore_record(HostedVisit, state.visit) if state.visit else None
    if world.visit:
        _restore_saved_activity(world.visit)
    world.player_interaction = _parse_player_interaction(state.player_interaction)
    world.player_call_check = state.player_call_check
    world.player_acknowledgments = [restore_record(PlayerAcknowledgment, a) for a in state.player_acknowledgments]
    world.creations = [work.model_copy(deep=True) for work in state.creations]
    for npc in world.archived:
        npc.world = world
    world.snake = Snake(**vars(restore_record(SnakeValues, state.snake,
        beliefs=[restore_record(Belief, b) for b in state.snake.beliefs],
        memories=[restore_record(Memory, m) for m in state.snake.memories],
        pending_social_opinions=[restore_record(OpinionEffect, e) for e in state.snake.pending_social_opinions],
        current_concern=restore_record(ConversationConcern, state.snake.current_concern) if state.snake.current_concern else None,
        player_opinion=restore_record(PlayerOpinion, state.snake.player_opinion))))
    world.snake.world = world
    by_id[world.snake.id] = world.snake
    world.arrival_error = state.arrival_error
    world.pending_arrival_days = state.pending_arrival_days.copy()
    world.processed_arrival_days = state.processed_arrival_days.copy()
    world.introductions = state.introductions.copy()
    world.roster_revision = state.roster_revision
    world.day = state.day
    world.pending_actions = [restore_record(PendingAction, action) for action in state.pending_actions]
    world.errands = [restore_record(Errand, errand) for errand in state.errands]
    history = [saved.restore() for saved in state.history]
    world.history = history
    history = {record.id: record for record in history}
    world.events = [restore_record(WorldEvent, event) for event in state.events]
    world.promises = [restore_record(Promise, promise) for promise in state.promises]
    world.facts = [restore_record(Fact, fact) for fact in state.facts]
    world.claims = [restore_record(Claim, claim) for claim in state.claims]
    world.disclosures = [restore_record(Disclosure, disclosure) for disclosure in state.disclosures]
    world.social_rotation = state.social_rotation
    for saved in state.conversations:
        Conversation.restore(by_id[saved.initiator], by_id[saved.recipient], history[saved.record],
            progress=restore_record(ResidentProgress, saved),
            dialogue=restore_record(DialogueProgress, saved),
            proposals=[restore_record(Proposal, proposal) for proposal in saved.proposals],
            information_context=saved.information_context.restore() if saved.information_context else None,
            concerns={role: restore_record(ConversationConcern, c) if c else None for role, c in saved.concerns.items()},
            saved_states=[restore_record(ActivitySnapshot, activity) for activity in saved.saved_states])
    actor_lookup = {actor.id: actor for actor in [*world.actors, world.player]}
    for saved in state.handoffs:
        Handoff.restore(actor_lookup[saved.initiator], actor_lookup[saved.recipient], history[saved.record],
            progress=restore_record(HandoffProgress, saved),
            saved_states={actor_id: restore_record(ActivitySnapshot, activity) for actor_id, activity in saved.saved_states.items()})
    if world.player_interaction is not None:
        world.player_interaction = world.player_interaction
    from simulation.player import call_valid, clear_call
    for npc in npcs:
        if npc.player_call and not call_valid(world, npc, npc.player_call):
            clear_call(npc)
    return world
