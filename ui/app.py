import os
from dotenv import load_dotenv

import math
from pathlib import Path
from uuid import uuid4

import pygame

from app_settings import load_settings
from character_data import load_character, random_character
from conversation_history import ConversationRecord, describe_opinion_change
from npc import NPC, assign_npc_names
from ollama_manager import load_model
from simulation_data import InvalidSaveName, SaveStore, capture_simulation, restore_simulation
from simulation import PendingAction, Simulation
from snake import SnakeEncounter
from name_knowledge import actor_catalog, name_forms, orientation_speech, validate_name_use, permitted_forms

from .game import GameScreen
from .character import CharacterScreen
from .menu import MenuScreen
from .load_sim import LoadSimScreen
from .save_sim import SaveSimScreen
from .requests import RequestManager
from .settings import SettingsScreen
from .setup import SetupScreen
from .stats import StatsScreen
from .ollama_setup import OllamaSetupScreen
from constants import (FPS, INTERACTION_GAP, STARTING_CHARACTER_FILE, STARTING_NPC_POSITION,
                       STARTING_SPAWN_ORIGIN, STARTING_SPAWN_SIDE_OFFSET, STARTING_SPAWN_SPREAD, WINDOW_SIZE)
from .world import CHARACTER_SCALE


STARTING_CHARACTER_PATH = Path(__file__).resolve().parent.parent / STARTING_CHARACTER_FILE


class Application:
    def __init__(self):
        self.window = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Pypeople")
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.Font(None, 78)
        self.heading_font = pygame.font.Font(None, 36)
        self.font = pygame.font.Font(None, 26)
        self.small_font = pygame.font.Font(None, 22)
        self.running = True
        self.screen = "menu"
        self.original_dialogue = False

        # Load the Gemini API key from the project's .env
        load_dotenv(
            Path(__file__).resolve().parent.parent / ".env"
        )

        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.provider = "gemini"
        self.ollama_model = load_model()
        settings = load_settings()
        self.npc_trait_mode = settings["npc_trait_mode"]
        self.voice_rewrites = settings["voice_rewrites"]
        self.dialogue_validation = settings["dialogue_validation"]
        self.session_id = 0
        self.request_npc = None
        self.conversation_requests = {}
        self.request_message = ""
        self.request_history = None
        self.simulation = None
        self.request_revision = None
        self.request_id = None
        self.request_world_revision = None
        self.save_store = SaveStore()
        self.save_id = None
        self.save_identity = None
        self.save_name_refresh_remaining = 0
        self.requests = RequestManager()
        self.menu = MenuScreen()
        self.load_sim = LoadSimScreen()
        self.save_sim = SaveSimScreen()
        self.settings = SettingsScreen()
        self.setup = SetupScreen()
        self.character = CharacterScreen()
        self.game = GameScreen(self.requests.max_requests)
        self.stats = StatsScreen()
        self.ollama_setup = OllamaSetupScreen()
        self.screens = {
            "menu": self.menu,
            "load_sim": self.load_sim,
            "save_sim": self.save_sim,
            "settings": self.settings,
            "setup": self.setup,
            "character": self.character,
            "game": self.game,
            "stats": self.stats,
            "ollama_setup": self.ollama_setup,
        }

    @property
    def npcs(self):
        return self.simulation.npcs if self.simulation is not None else []

    @property
    def tree(self):
        return self.simulation.tree if self.simulation is not None else None

    @property
    def conversation_history(self):
        return self.simulation.history if self.simulation is not None else []

    @property
    def simulation_name(self):
        return self.save_id if self.save_id is not None else "Unsaved simulation"

    def _activate_simulation(self, simulation, *, save_id=None, save_identity=None):
        """Switch sessions only after the new world is fully prepared."""
        self.requests.cancel_all()
        self.session_id += 1
        self.game.reset()
        self.stats.open()
        self.conversation_requests.clear()
        self.request_npc = self.request_history = None
        self.request_revision = self.request_id = self.request_world_revision = None
        self.request_message = ""
        self.simulation = simulation
        self.save_id, self.save_identity = save_id, save_identity
        self.screen = "game"

    def start_game(self):
        try:
            characters = [load_character(STARTING_CHARACTER_PATH)]
            characters[0]["Opinions"] = {}
            # Each simulation owns its profiles and opinions inside its save file.
            for index, source in enumerate(self.setup.sources, start=1):
                try:
                    characters.append(random_character(trait_mode=self.npc_trait_mode)
                                      if source is None else load_character(source))
                except (OSError, ValueError) as error:
                    raise ValueError(f"NPC {index + 1}: {error}") from error
                if source is not None and Path(source).resolve() == STARTING_CHARACTER_PATH.resolve():
                    characters[-1]["Opinions"] = {}
        except (OSError, ValueError) as error:
            self.setup.message = f"Could not start game: {error}"
            return
        self.setup.message = ""
        self.setup.count_input.active = False
        npcs = []
        columns = max(1, math.ceil(math.sqrt((len(characters) - 1) / 2)))
        rows = max(1, math.ceil((len(characters) - 1) / (columns * 2)))
        for index, character in enumerate(characters):
            if index == 0:
                position = STARTING_NPC_POSITION
            else:
                # Spawn on either side, leaving the central tree and its approach clear.
                side, slot = (index - 1) % 2, (index - 1) // 2
                column, row = slot % columns, slot // columns
                position = (STARTING_SPAWN_ORIGIN[0] + side * STARTING_SPAWN_SIDE_OFFSET
                            + column * STARTING_SPAWN_SPREAD[0] / max(1, columns - 1),
                            STARTING_SPAWN_ORIGIN[1] + row * STARTING_SPAWN_SPREAD[1] / max(1, rows - 1))
            npc = NPC(character=character, x=position[0], y=position[1])
            sprite = self.character.renderer.render(character, npc.facing, CHARACTER_SCALE)
            npc.sprite_size = sprite.get_size()
            npcs.append(npc)
        assign_npc_names(npcs)
        self._activate_simulation(Simulation(npcs, npc_trait_mode=self.npc_trait_mode))
        self.menu.message = ""

    def sync_save_name(self):
        if self.save_id is not None and self.save_identity is not None:
            name = self.save_store.current_name(self.save_id, self.save_identity)
            if name != self.save_id:
                self.game.save_message_remaining = 0
            self.save_id = name

    def request_save(self):
        try:
            name = self.save_store.next_name()
        except OSError:
            self.game.show_save_message("Could not read saves. Check folder permissions and try again.")
            return
        self.game.prompt_input.active = False
        self.game.prompt_input.backspace_held = False
        self.save_sim.open(name)
        self.screen = "save_sim"

    def save_current_simulation(self, *, name):
        try:
            state = capture_simulation(self)
            self.save_id = self.save_store.save(state, name=name)
            self.save_identity = self.save_store.file_identity(self.save_id)
        except (InvalidSaveName, FileExistsError) as error:
            self.save_sim.message = str(error)
            self.game.show_save_message(str(error))
            return False
        except (OSError, ValueError):
            message = "Could not save. Check the file and folder permissions, then try again."
            self.save_sim.message = message
            self.game.show_save_message(message)
            return False
        self.game.show_save_message(f"Saved {self.simulation_name}.")
        return True

    def open_load_simulations(self):
        self.load_sim.open(self.save_store)
        self.screen = "load_sim"

    def load_simulation(self, save_id):
        try:
            saved = self.save_store.load(save_id)
            identity = self.save_store.file_identity(save_id)
            simulation = restore_simulation(saved.state)
            for npc in simulation.npcs:
                sprite = self.character.renderer.render(npc.character, npc.facing, CHARACTER_SCALE)
                npc.sprite_size = sprite.get_size()
        except (OSError, ValueError):
            self.load_sim.message = "Could not load this simulation. The save may be incompatible, damaged, or unavailable."
            return False
        self._activate_simulation(simulation, save_id=save_id, save_identity=identity)
        self.game.camera_offset[:] = saved.state.camera_offset
        self.game.camera_zoom = saved.state.camera_zoom
        self.game.show_save_message(f"Loaded {saved.name}.")
        return True

    def leave_game(self):
        self.requests.cancel_all()
        for npc in self.npcs:
            npc.request_pending = any(action.npc_id == npc.id for action in self.simulation.pending_actions)
        if self.request_history is not None:
            self.request_history.status = "Ended before the reply arrived"
            self.request_history.outcomes.insert(0, "Left the game before a reply was received.")
            self.request_history = None
        self.session_id += 1
        self.conversation_requests.clear()
        self.screen = "menu"
        self.menu.message = ""
        self.game.close_dialogue()
        for npc in self.npcs:
            if npc.conversation is not None:
                npc.conversation.finish(self.tree)

    def open_settings(self):
        self.screen = "settings"
        self.settings.open(self.provider, self.api_key, self.ollama_model, self.npc_trait_mode,
                           voice_rewrites=self.voice_rewrites, dialogue_validation=self.dialogue_validation)

    def open_stats(self):
        self.game.prompt_input.active = False
        self.game.prompt_input.backspace_held = False
        self.stats.open()
        self.screen = "stats"

    def submit(self):
        if self.simulation.paused:
            return
        message = self.game.prompt_input.text.strip()[:300]
        npc = self.game.selected_npc
        if npc is self.simulation.snake:
            self.submit_snake(message)
            return
        if self.screen != "game" or npc is None or npc.busy or npc.end_conversation or npc.request_pending or not message or self.requests.busy or self.request_history is not None:
            return
        if self.provider == "gemini" and not self.api_key:
            self.game.set_reply("Add an API key in Settings to generate a reply.")
            return
        snapshot = npc.snapshot(self.tree)
        snapshot["player_dialogue"] = [
            {"speaker": speaker, "text": text}
            for speaker, text in self.game.player_record.turns[-12:]
        ] if self.game.player_record is not None else []
        snapshot["other_npcs"] = [
            {**other.snapshot(self.tree, private=False),
             "available_to_talk": not other.busy and not other.request_pending}
            for other in self.npcs if other is not npc
        ]
        self._submit_player_message(npc, message, snapshot)

    def _submit_player_message(self, npc, message, snapshot, *, purpose="chat"):
        request_id = uuid4().hex
        if not self.requests.submit(
            self.session_id, self.provider, self.api_key, message, snapshot,
            request_id=request_id, ollama_model=self.ollama_model,
            voice_rewrites=self.voice_rewrites, dialogue_validation=self.dialogue_validation,
        ):
            self.game.set_reply("Could not start the request. Please try again.")
            return
        self.request_npc = npc
        self.request_id = request_id
        self.request_revision = npc.command_revision
        self.request_world_revision = self.simulation.roster_revision
        npc.request_pending = True
        self.request_message = message
        if self.game.player_record is None:
            self.game.player_record = ConversationRecord(("You", npc.name), message, "Waiting for reply", purpose=purpose)
            self.conversation_history.append(self.game.player_record)
        self.request_history = self.game.player_record
        self.request_history.status = "Waiting for reply"
        self.request_history.error = ""
        self.request_history.turns.append(("You", message))
        self.game.prompt_input.set_value("")

    def remember_response_opinions(self, npc, response):
        updates = {}
        for entry in response.opinion_updates:
            matches = [other for other in self.npcs if other is not npc and other.name.casefold() == entry.person.strip().casefold()]
            if len(matches) != 1:
                return "I couldn't identify that person to remember my opinion. Please try again."
            updates[matches[0].name] = {"opinion": entry.opinion, "reason": entry.reason}
        if updates:
            try:
                npc.remember_opinions(updates)
            except (OSError, ValueError):
                return "I couldn't save my opinion to my character file. Please try again."
        return None

    def start_conversation(self, npc, target_name, topic):
        matches = [other for other in self.npcs if other.name.casefold() == target_name.strip().casefold()]
        if len(matches) != 1:
            return "Please name one other NPC in this world to talk to."
        target = matches[0]
        if target.request_pending:
            return "That person has a pending player request. Try again once it finishes."
        try:
            # Leave enough space for both sprites and their name labels.
            spacing = self._interaction_spacing(npc, target)
            conversation = self.simulation.start_conversation(npc, target, topic, spacing=spacing)
        except ValueError as error:
            return str(error)
        if self.game.selected_npc is npc or self.game.selected_npc is target:
            self.game.close_dialogue()
        return None

    def apply_pending_action(self, action):
        npc = self.simulation.npc_by_id(action.npc_id)
        if npc is None:
            return
        npc.request_pending = False
        error = ""
        if npc.busy or npc.command_revision != action.revision:
            error = "That request is no longer available because the NPC's activity changed."
        elif action.action == "TALK_TO_NPC":
            target = self.simulation.npc_by_id(action.target_id)
            error = self.start_conversation(npc, target.name, action.topic) if target else "That person is unavailable."
        elif action.action == "GIVE_ITEM":
            target = self.simulation.npc_by_id(action.target_id)
            try:
                record = self.conversation_history[action.record] if action.record is not None else None
                spacing = self._interaction_spacing(npc, target) if target else 0
                gift = npc.give_items(target, action.item, action.quantity, record, spacing)
                if record is None:
                    self.conversation_history.append(gift.record)
            except ValueError as exc:
                error = str(exc)
        else:
            try:
                npc.apply_action(action.action, self.tree)
            except ValueError as exc:
                error = str(exc)
        if action.record is not None:
            record = self.conversation_history[action.record]
            record.status = "Failed" if error else ("Walking to give" if action.action == "GIVE_ITEM" else "Completed")
            if error:
                record.error = error
                record.outcomes[0:1] = [error]
            else:
                outcomes = {
                    "WANDER": "Returned to autonomous activities.",
                    "STOP": "Holding position until another player task.",
                    "GO_TO_TREE": "Started walking to the tree.",
                    "SHAKE_TREE": "Started the task of shaking the tree.",
                    "TALK_TO_NPC": f"Started a conversation with {target.name}." if action.action == "TALK_TO_NPC" else "",
                    "GIVE_ITEM": f"Walking to give {action.quantity} {action.item} to {target.name}." if action.action == "GIVE_ITEM" else "",
                    "EAT": "Started eating an apple.",
                }
                record.outcomes[0:1] = [outcomes[action.action]]
        if error:
            npc.end_conversation = False
            npc.reply = error
            npc.rewritten_reply = ""
            npc.set_emotion("neutral", 0)
            if self.game.selected_npc is npc:
                self.game.set_reply(error)

    def _interaction_spacing(self, first, second):
        return sum(self.small_font.size(actor.name)[0] for actor in (first, second)) / 2 + INTERACTION_GAP

    def request_conversation_dialogue(self, conversation, *, start=True):
        if not conversation.accepts_response or not conversation.arrived:
            return
        if conversation.dialogue_state != "waiting":
            return
        if self.provider == "gemini" and not self.api_key:
            conversation.fail_dialogue("Add an API key in Settings to generate this conversation.")
            return
        if (self.session_id, conversation.id) not in self.requests.logs:
            self.requests.prepare(self.session_id, self.provider, conversation.snapshot(self.tree, self.npcs),
                                  request_id=conversation.id, ollama_model=self.ollama_model)
        if self.requests.busy or not start:
            return
        snapshot = conversation.snapshot(self.tree, self.npcs)
        if not self.requests.submit(
            self.session_id, self.provider, self.api_key,
            f"Write what {conversation.initiator.name} and {conversation.recipient.name} say directly to each other.", snapshot,
            request_id=conversation.id, ollama_model=self.ollama_model,
            voice_rewrites=self.voice_rewrites, dialogue_validation=self.dialogue_validation,
        ):
            conversation.fail_dialogue("Could not start dialogue generation.")
            return
        conversation.error = ""
        conversation.dialogue_state = "pending"
        self.conversation_requests[conversation.id] = conversation

    def retry_conversation_dialogue(self, conversation):
        if isinstance(conversation, SnakeEncounter):
            if self.requests.busy or conversation.active or self.simulation.npc_by_id(conversation.recipient.id) is None:
                return
            try:
                conversation.recipient.set_selected(False)
                replacement = SnakeEncounter(self.simulation, conversation.recipient, conversation.purpose,
                                             conversation.accusation, origin=conversation.origin)
            except ValueError as exc:
                conversation.record.error = str(exc)
                return
            self.game.viewed_conversation = replacement
            self.game._conversation_page_turns = None
            return
        if conversation.active or conversation.turns or self.requests.busy:
            return
        try:
            replacement = self.simulation.start_conversation(
                conversation.initiator, conversation.recipient, conversation.topic,
                origin=conversation.origin, purpose=conversation.purpose)
        except ValueError as exc:
            conversation.error = str(exc)
            return
        self.game.viewed_conversation = replacement
        self.game._conversation_page_turns = None

    def retry_arrival(self):
        self.simulation.arrival_error = ""

    def submit_snake(self, message):
        snake = self.simulation.snake
        if self.screen != "game" or not message or self.requests.busy or snake.encounter or snake.request_pending or self.request_history is not None:
            return
        snapshot = {"request_kind": "snake_player", "snake": snake.snapshot(),
                    "_name_actors": actor_catalog(self.simulation),
                    "player_dialogue": self.game.player_record.turns[-12:] if self.game.player_record else []}
        self._submit_player_message(snake, message, snapshot, purpose="snake_player")

    def cancel_model_request(self, session_id, request_id):
        if not self.requests.cancel(session_id, request_id):
            return
        if session_id != self.session_id:
            return
        reason = "Model request cancelled by player. Retry when ready."
        conversation = self.conversation_requests.pop(request_id, None)
        if conversation is not None:
            conversation.fail_dialogue(reason)
        elif request_id == self.request_id:
            npc, record = self.request_npc, self.request_history
            if npc is not None:
                npc.request_pending = False
                npc.reply = reason
                npc.rewritten_reply = ""
                if self.game.selected_npc is npc:
                    self.game.set_reply(reason)
            if record is not None:
                record.status, record.error = "Failed", reason
            self.request_history = self.request_npc = self.request_id = None

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        self.screens[self.screen].handle_event(self, event)

    def update(self, dt):
        paused = self.simulation is not None and self.simulation.paused
        simulation_active = self.screen in ("game", "stats") and not paused
        session_active = self.screen in ("game", "stats", "save_sim")
        active_session = self.session_id if session_active else None
        # Stats owns the version toggle; visiting it keeps the live reading hold.
        viewed = self.game.viewed_conversation if self.screen in ("game", "stats") and self.game.dialogue_open else None
        # Leave completed replies queued so they cannot change a paused world.
        replies = () if session_active and paused else self.requests.poll(active_session)
        if session_active and not paused:
            # A progressing exchange gets a fresh wait for its next provider call.
            for (session_id, request_id), worker in self.requests.workers.items():
                pending = self.conversation_requests.get(request_id)
                if (session_id == active_session and pending is not None
                        and getattr(pending, "request_stage", None) != worker.stage):
                    pending.request_stage = worker.stage
                    pending.wait_elapsed = 0.0
        for request_id, result, error, failed_turns, rewritten in replies:
            conversation = self.conversation_requests.pop(request_id, None)
            if conversation is not None:
                if conversation.id == request_id and conversation.accepts_response:
                    if error:
                        conversation.fail_dialogue(error, failed_turns=failed_turns)
                    else:
                        try:
                            conversation.set_response(result, reading=conversation is viewed, rewritten_parts=rewritten,
                                                      dialogue_validation=self.dialogue_validation)
                        except ValueError as exc:
                            error = str(exc)
                            conversation.fail_dialogue(str(exc), failed_turns=result.turns)
                    self.requests.finish(active_session, request_id, error)
                else:
                    self.requests.finish(active_session, request_id, "The conversation ended before the reply arrived.")
                continue
            if request_id != self.request_id:
                self.requests.finish(active_session, request_id, "The player request is no longer active.")
                continue
            npc = self.request_npc
            if npc is self.simulation.snake:
                npc.request_pending = False
                record = self.request_history
                if self.request_world_revision != self.simulation.roster_revision or npc.encounter:
                    error = "The world changed before the reply arrived. Send the message again."
                if record:
                    record.status, record.error = ("Failed", error) if error else ("Completed", "")
                    if not error:
                        record.dialogue_errors = list(dict.fromkeys([*record.dialogue_errors, *result._dialogue_errors]))
                        record.name_forms = name_forms(actor_catalog(self.simulation))
                        record.add_turn(npc.name, result.reply, [{"text": result.reply,
                            "emotion": {"name": "neutral", "intensity": 0}}], rewritten[0] if rewritten else ())
                npc.reply = f"Request failed: {error}" if error else result.reply
                npc.rewritten_reply = rewritten[0][0] if not error and rewritten else ""
                if self.game.selected_npc is npc:
                    self.game.set_reply(npc.reply)
                self.requests.finish(active_session, request_id, error)
                self.request_history = None
                continue
            if npc is None or self.simulation.npc_by_id(npc.id) is not npc:
                self.requests.finish(active_session, request_id, "The NPC is no longer available.")
                self.request_history = None
                continue
            npc.request_pending = False
            record = self.request_history
            if not error and (npc.busy or npc.command_revision != self.request_revision or self.request_world_revision != self.simulation.roster_revision):
                error = "That reply arrived after the NPC became busy. Please send the request again."
            if not error and self.dialogue_validation:
                try:
                    person = npc.snapshot(self.tree)
                    validate_name_use(result.reply, permitted_forms(person, person["_name_actors"], npc.known_names), person["_name_actors"])
                except ValueError as exc:
                    error = str(exc)
            if record is not None and not error:
                record.dialogue_errors = list(dict.fromkeys([*record.dialogue_errors, *result._dialogue_errors]))
                record.outcomes.extend(
                    describe_opinion_change(
                        npc.name, entry.person, npc.opinions.get(entry.person, {}).get("opinion"), entry.opinion, entry.reason
                    )
                    for entry in result.opinion_updates
                )
            if not error:
                error = self.remember_response_opinions(npc, result)
            if not error:
                if result.action != "SAY":
                    target = next((other for other in self.npcs if other.name == result.target), None)
                    if result.action == "EAT" and npc.inventory.get("fruit", 0) < 1:
                        error = "I don't have an apple to eat."
                    elif result.action in ("TALK_TO_NPC", "GIVE_ITEM") and (target is None or target.busy or target.request_pending):
                        error = "That person is busy now. Please try again later."
                    else:
                        if result.action == "GIVE_ITEM":
                            try:
                                npc.check_gift(target, result.item, result.quantity)
                            except ValueError as exc:
                                error = str(exc)
                        if not error:
                            self.simulation.pending_actions.append(PendingAction(
                                npc.id, result.action, npc.command_revision,
                                target.id if result.action in ("TALK_TO_NPC", "GIVE_ITEM") else None,
                                self.request_message,
                                self.conversation_history.index(record) if record is not None else None,
                                result.item, result.quantity,
                            ))
                            npc.request_pending = True
            npc.reply = f"Request failed: {error}" if error else result.reply
            npc.rewritten_reply = rewritten[0][0] if not error and rewritten else ""
            # A reply arriving after the player leaves must not wait for a reader.
            npc.end_conversation = bool(not error and result.end_conversation and npc.selected)
            if error:
                npc.set_emotion("neutral", 0)
            else:
                npc.set_emotion(result.emotion.name, result.emotion.intensity)
            if record is not None:
                if not error:
                    record.name_forms = name_forms(actor_catalog(self.simulation))
                    record.orientation_discussed = record.orientation_discussed or orientation_speech([npc.reply])
                    record.add_turn(npc.name, npc.reply, [{
                        "text": npc.reply,
                        "emotion": {"name": npc.emotion, "intensity": npc.emotion_intensity},
                    }], rewritten[0] if rewritten else ())
                record.error = error or ""
                waiting_for_done = npc.request_pending and npc.end_conversation
                if error:
                    record.status = "Failed"
                    record.outcomes.insert(0, f"The request could not be completed: {error}")
                else:
                    if waiting_for_done:
                        record.status = "Waiting for Done"
                        outcome = "Accepted; starts after the player closes the conversation."
                    elif npc.request_pending:
                        record.status = "Applying player request"
                        outcome = "Accepted; applies on the next simulation tick."
                    else:
                        record.status = "Completed"
                        outcome = "Replied without changing their activity."
                    record.outcomes.insert(0, outcome)
                self.request_history = None
            if self.game.selected_npc is npc and self.game.viewed_conversation is None:
                self.game.set_reply(npc.reply)
            self.requests.finish(active_session, request_id, error)
        self.game.prompt_input.update(dt)
        self.settings.key_input.update(dt)
        self.settings.update()
        if self.screen == "load_sim":
            self.load_sim.name_input.update(dt)
        if self.screen == "save_sim":
            self.save_sim.name_input.update(dt)
        if self.screen in ("game", "stats", "save_sim"):
            self.save_name_refresh_remaining -= dt
            if self.save_name_refresh_remaining <= 0:
                self.save_name_refresh_remaining = 1
                try:
                    self.sync_save_name()
                except (OSError, ValueError):
                    pass
        self.game.save_message_remaining = max(0, self.game.save_message_remaining - dt)
        self.character.update(dt)
        if self.screen == "setup":
            self.setup.update(dt)
        if self.simulation is not None:
            self.simulation.advance(dt, active=simulation_active, action_handler=self.apply_pending_action,
                                    viewed_conversation=viewed)
        if simulation_active:
            if self.game.selected_npc is not None and self.game.selected_npc is not self.simulation.snake and self.simulation.npc_by_id(self.game.selected_npc.id) is not self.game.selected_npc:
                self.game.close_dialogue()
            # Initialize arriving sprites before the next render.
            for npc in self.npcs:
                if npc.sprite_size is None:
                    npc.sprite_size = self.character.renderer.render(npc.character, npc.facing, CHARACTER_SCALE).get_size()
            conversations = [npc.conversation for npc in self.npcs
                             if npc.conversation is not None and npc.conversation.initiator is npc]
            if self.simulation.snake.encounter:
                conversations.append(self.simulation.snake.encounter)
            # Player conversations get the first free slot. While composing,
            # leave one slot free and let other conversations use the rest.
            composing = self.screen == "game" and self.game.prompt_input.active
            for conversation in sorted(conversations, key=lambda interaction: interaction.origin != "player"):
                self.request_conversation_dialogue(conversation,
                    start=not composing or self.requests.available_slots > 1)
            self.requests.expire_queued(self.session_id, {
                conversation.id for conversation in conversations
                if conversation.accepts_response and conversation.arrived and conversation.dialogue_state == "waiting"
            })

    def draw(self):
        self.screens[self.screen].draw(self)
        pygame.display.flip()

    def run(self):
        try:
            while self.running:
                dt = self.clock.tick(FPS) / 1000.0
                for event in pygame.event.get():
                    self.handle_event(event)
                    if not self.running:
                        break
                if self.running:
                    self.update(dt)
                    self.draw()
        finally:
            self.requests.cancel_all()


def run():
    """Initialize Pygame, run the interface, and always release its resources."""
    try:
        pygame.display.init()
        pygame.font.init()
        pygame.key.start_text_input()
        Application().run()
    finally:
        pygame.quit()
    return 0
