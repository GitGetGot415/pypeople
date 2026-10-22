import os
from dotenv import load_dotenv

import math
from pathlib import Path
from uuid import uuid4
from time import monotonic

import pygame

from app_settings import load_settings
from character_data import load_character, random_character
from conversation_history import ConversationRecord
from conversation_results import (accept_generated_dialogue, accept_resident_reply, accept_snake_reply,
                                  apply_pending_action)
from npc import Behavior, NPC, assign_npc_names
from player import active_player
from ollama_manager import load_model
from simulation_data import InvalidSaveName, SaveStore, capture_simulation, restore_simulation
from simulation import Simulation
from snake import SnakeEncounter
from social import concern_description
from dialogue_context import actor_catalog, name_forms

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
                       STARTING_SPAWN_ORIGIN, STARTING_SPAWN_SIDE_OFFSET, STARTING_SPAWN_SPREAD, WINDOW_SIZE,
                       PLAYER_CONTEXT_TRANSFER_LIMIT, PLAYER_PROMISE_DEADLINE_STEPS)
from .world import CHARACTER_SCALE
from .fonts import UnicodeFont
from constants import UI_TITLE_FONT_SIZE, UI_HEADING_FONT_SIZE, UI_TEXT_FONT_SIZE, UI_SMALL_FONT_SIZE, NAME_TAG_FONT_SIZE


STARTING_CHARACTER_PATH = Path(__file__).resolve().parent.parent / STARTING_CHARACTER_FILE


class Application:
    def __init__(self):
        self.window = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Pypeople")
        self.clock = pygame.time.Clock()
        self.title_font = UnicodeFont(UI_TITLE_FONT_SIZE)
        self.heading_font = UnicodeFont(UI_HEADING_FONT_SIZE)
        self.font = UnicodeFont(UI_TEXT_FONT_SIZE)
        self.small_font = UnicodeFont(UI_SMALL_FONT_SIZE)
        self.name_tag_font = UnicodeFont(NAME_TAG_FONT_SIZE)
        self.running = True
        self.screen = "menu"
        self.original_dialogue = False
        self.condensed_dialogue = False

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
        self.player_requests = {}
        self.conversation_requests = {}
        self.creative_requests = {}
        self.simulation = None
        self.save_store = SaveStore()
        self.save_id = None
        self.save_identity = None
        self.save_name_refresh_remaining = 0
        self.requests = RequestManager(settings["max_llm_requests"])
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
        if self.simulation:
            self.end_player_conversation("The session changed.")
            self.simulation.cancel_gifts("The session changed before the handoff finished.")
        from apartments import discard_work
        for npc in self.npcs:
            if npc.creative_job:
                discard_work(npc)
        self.session_id += 1
        self.game.reset()
        self.stats.open()
        self.conversation_requests.clear()
        self.creative_requests.clear()
        self.player_requests.clear()
        self.simulation = simulation
        self.game.app = self
        self.game.location = simulation.player.location if simulation.mode == "player" else "garden"
        self.save_id, self.save_identity = save_id, save_identity
        self.screen = "game"

    def start_game(self):
        try:
            sources = [STARTING_CHARACTER_PATH]
            if self.setup.mode == "spectator":
                sources.extend(self.setup.sources[1:])
            characters = []
            # Both modes require Judy first; Spectator can add other residents.
            for index, source in enumerate(sources, start=1):
                try:
                    characters.append(random_character(trait_mode=self.npc_trait_mode)
                                      if source is None else load_character(source))
                except (OSError, ValueError) as error:
                    raise ValueError(f"NPC {index}: {error}") from error
                if source is not None and Path(source).resolve() == STARTING_CHARACTER_PATH.resolve():
                    characters[-1]["Opinions"] = {}
        except (OSError, ValueError) as error:
            self.setup.message = f"Could not start game: {error}"
            return
        self.setup.message = ""
        self.setup.count_input.active = False
        npcs = []
        other_count = max(0, len(characters) - 1)
        columns = max(1, math.ceil(math.sqrt(other_count / 2)))
        rows = max(1, math.ceil(other_count / (columns * 2)))
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
        simulation = Simulation(npcs, npc_trait_mode=self.npc_trait_mode, mode=self.setup.mode,
                                player_start=self.setup.mode == "player")
        self._activate_simulation(simulation)
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
        self.end_player_conversation("Left the game.")
        self.game.inventory.close()
        self.simulation.cancel_gifts("Left the game before the handoff finished.")
        self.requests.cancel_all()
        self.stats.gallery.close()
        self.creative_requests.clear()
        from apartments import relax
        for npc in self.npcs:
            if npc.creative_job:
                relax(npc)
        for npc in self.npcs:
            npc.request_pending = any(action.npc_id == npc.id for action in self.simulation.pending_actions)
        for request in self.player_requests.values():
            if request["pending"]:
                request["record"].status = "Ended before the reply arrived"
                request["record"].outcomes.insert(0, "Left the game before a reply was received.")
        self.player_requests.clear()
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
                           voice_rewrites=self.voice_rewrites, dialogue_validation=self.dialogue_validation,
                           max_requests=self.requests.max_requests)

    def open_stats(self):
        self.game.prompt_input.active = False
        self.game.prompt_input.backspace_held = False
        self.stats.open()
        self.screen = "stats"

    def submit(self):
        from player import active_player
        if self.simulation.paused or not active_player(self.simulation):
            return
        message = self.game.prompt_input.text.strip()[:300]
        npc = self.game.selected_npc
        if npc is self.simulation.snake:
            self.submit_snake(message)
            return
        if self.screen != "game" or npc is None or not self.player_owns(npc) or npc.busy or npc.end_conversation or npc.request_pending or not message:
            return
        if self.provider == "gemini" and not self.api_key:
            self.game.set_reply("Add an API key in Settings to generate a reply.")
            return
        snapshot = self.player_snapshot(npc, message)
        snapshot["player_dialogue"] = [
            {"speaker": turn.speaker, "text": turn.text}
            for turn in self.game.player_record.turns[-12:]
        ] if self.game.player_record is not None else []
        snapshot["other_npcs"] = [
            {**other.snapshot(self.tree, private=False),
             "available_to_talk": not other.busy and not other.request_pending and not other.selected and not other.player_call}
            for other in self.simulation.npcs if other is not npc and other.location == npc.location
        ]
        snapshot["other_npcs"].append(self.player_target())
        self._submit_player_message(npc, message, snapshot)

    def player_target(self):
        player = self.simulation.player
        return dict(npc_id=player.id, npc_name="You", kind="player", character={"Name": "You"},
                    inventory=player.inventory.copy(),
                    available_to_talk=True)

    def player_owns(self, npc, *, stages=("conversation",)):
        from player import owns_conversation
        return owns_conversation(self.simulation, npc, stages=stages)

    def end_player_conversation(self, reason="Conversation ended."):
        if not self.simulation or not self.simulation.player_interaction:
            return
        from player import end_interaction
        from snake import PlayerSnakeEncounter
        interaction = self.simulation.player_interaction
        for request_id, request in tuple(self.player_requests.items()):
            if request["npc"].id == interaction.get("npc_id"):
                if request["pending"]:
                    self.requests.cancel(self.session_id, request_id)
                self.player_requests.pop(request_id, None)
        npc = self.simulation.npc_by_id(interaction.get("npc_id"))
        if reason == "Conversation ended." and npc and (npc.gift or any(a.npc_id == npc.id and a.action == "GIVE_ITEM"
                and a.target_id == self.simulation.player.id for a in self.simulation.pending_actions)):
            interaction["stage"] = "handoff"
            self.game.pending_offer = None
            return
        encounter = self.simulation.snake.encounter
        if isinstance(encounter, PlayerSnakeEncounter):
            if encounter.outcome == "eat":
                self.simulation.player_interaction = None
            else:
                encounter.finish(error=reason if reason != "Conversation ended." else "")
        else:
            end_interaction(self.simulation, reason)
        self.game.pending_offer = None
        self.game.call_choice = None

    def open_player_dialogue(self, npc):
        if not self.player_owns(npc):
            return
        self.game.dialogue_open, self.game.selected_npc = True, npc
        self.game.prompt_input.set_value("")
        npc.set_selected(True)
        from snake import PlayerSnakeEncounter
        if npc is self.simulation.snake:
            self.game.player_record = npc.encounter.record
            npc.end_conversation = npc.encounter.outcome == "eat"
        else:
            interaction = self.simulation.player_interaction
            record = next((r for r in self.simulation.history if r.id == interaction.get("record_id")), None)
            if record is None:
                from player_story import scripted_player_opening
                record = scripted_player_opening(self.simulation, npc)
            self.game.player_record = record
            gift = interaction.get("gift")
            if gift:
                snapshot = self.player_snapshot(npc)
                work = next((w for w in self.simulation.creations if w.id == gift["item"]), None)
                incoming = dict(**gift, giver_id=self.simulation.player.id, title=work.title if work else "apples",
                                creator_id=work.creator_id if work else None)
                snapshot.update(request_kind="player_gift", player_interaction={"incoming_gift": incoming})
                self._submit_player_message(npc, "", snapshot)
                interaction["gift"] = None
        record = self.game.player_record
        pending = self.simulation.player_interaction.get("pending_request")
        if record and pending and record.status == "Reply pending when saved":
            record.error = "Generation was interrupted by saving. Retry this exchange."
            request_id = uuid4().hex
            snapshot = self.player_snapshot(npc, pending["message"]) if npc is not self.simulation.snake else {}
            if pending["kind"]:
                snapshot.update(request_kind=pending["kind"], player_interaction=pending["context"])
            self.player_requests[request_id] = dict(id=request_id, npc=npc, record=record, pending=False,
                message=pending["message"], snapshot=snapshot, revision=npc.command_revision,
                world_revision=self.simulation.roster_revision)
        self.game.set_reply(npc.reply if self.game.player_record else "")
        self.game.prompt_input.focus()

    def accept_player_invitation(self, offer_id):
        from apartments import accept_invitation
        for request_id, request in tuple(self.player_requests.items()):
            if request["pending"]:
                self.requests.cancel(self.session_id, request_id)
            self.player_requests.pop(request_id, None)
        try:
            accept_invitation(self.simulation, offer_id)
            self.game.close_dialogue(terminate=False)
        except ValueError as exc:
            self.game.show_save_message(str(exc))

    def use_door(self, target):
        from locations import start_journey, allocated_floors, hallway_location, door_point, valid_location
        from constants import ROOM_ARRIVAL, ELEVATOR_ARRIVAL, HALLWAY_EXIT, GARDEN_BUILDING_ARRIVAL
        world, player = self.simulation, self.simulation.player
        if target["kind"] == "elevator":
            self.game.open_location_menu("Elevator", [(f"Floor {f}", hallway_location(f)) for f in allocated_floors(world)])
            return
        if not active_player(world):
            if valid_location(world, target["destination"]):
                self.game.observe_location(target["destination"])
            else:
                self.game.show_save_message("This room is vacant.")
            return
        if target["kind"] == "room" and target["room"] != 1:
            room = target["room"]
            host = next((n for n in world.npcs if n.room_number == room), None)
            permitted = bool(world.visit and world.visit["host_id"] == (host.id if host else None))
            if not permitted:
                if not host or host.location != target["destination"]:
                    self.game.show_save_message("This room is vacant." if not host else "The resident is absent.")
                    return
                from player import start_approach
                try:
                    start_approach(world, host, knock_room=room)
                    self.open_player_dialogue(host)
                    self.game.show_save_message("Knocking. Ask the resident for an invitation.")
                except ValueError as exc:
                    self.game.show_save_message(str(exc))
                return
        destination = target["destination"]
        point = (ROOM_ARRIVAL if destination.startswith("room:") else
                 GARDEN_BUILDING_ARRIVAL if destination == "garden" else
                 door_point(target["room"]) if target["kind"] == "exit_room" else HALLWAY_EXIT)
        start_journey(player, destination, point, purpose="doorway", fast=True)

    def player_snapshot(self, npc, message=""):
        from player import work_choices, requested_work_id
        snapshot = npc.snapshot(self.tree, self.simulation.player.id)
        from apartments import invitation_choices
        snapshot["invitation_choices"] = invitation_choices(self.simulation, npc)
        snapshot["pending_invitation"] = self.game.pending_invitation(self.simulation)
        interaction = self.simulation.player_interaction
        snapshot["closed_door_knock"] = interaction.get("knock_room") if interaction else None
        from player import relevant_claims
        snapshot["player_claims"] = relevant_claims(self.simulation, npc.id, message)
        snapshot.update(player_id=self.simulation.player.id, player_inventory=self.player_target(),
                        owned_works=work_choices(self.simulation, npc.id, message))
        from social import phase_label
        snapshot.update(_player_message=message, phase_index=self.simulation.phase_index,
            apple_promise_choices=[dict(deadline=self.simulation.phase_index + step,
                label=phase_label(self.simulation.phase_index + step)) for step in PLAYER_PROMISE_DEADLINE_STEPS])
        snapshot["resolved_work_id"] = requested_work_id(self.simulation, npc.id, message)
        record = self.game.player_record
        offers = record.player_promise_offers if record and npc.id in record.participant_ids else []
        snapshot["player_apple_offer_count"] = len(offers)
        snapshot["pending_apple_promise"] = next((o.copy() for o in offers if o["status"] == "offered"), None)
        snapshot["known_player_transfers"] = sorted([t for r in self.conversation_history for t in r.transfers
            if {t["giver_id"], t["recipient_id"]} == {self.simulation.player.id, npc.id}], key=lambda t: t["sequence"])[-PLAYER_CONTEXT_TRANSFER_LIMIT:]
        return snapshot

    def talk_to_call(self, npc):
        from player import call_valid, clear_call
        if npc.unread_player_reply:
            self.game.open_call(self, npc)
            return ""
        if (self.simulation.paused
                or not npc.player_call or not call_valid(self.simulation, npc, npc.player_call)
                or not self.player_owns(npc, stages=("prompt",))):
            return "This call is unavailable while you or the resident are occupied."
        call = npc.player_call.copy()
        self.game.close_dialogue(terminate=False)
        self.simulation.player_interaction["stage"] = "conversation"
        self.game.dialogue_open, self.game.selected_npc = True, npc
        npc.set_selected(True)
        from player_story import scripted_player_opening
        record = scripted_player_opening(self.simulation, npc)
        if record is not None:
            self.game.player_record = record
            self.game.set_reply(npc.reply)
            self.game.prompt_input.focus()
            return ""
        snapshot = self.player_snapshot(npc)
        from player import work_choices
        snapshot["owned_works"] = work_choices(self.simulation, npc.id, call["reference"])
        snapshot.update(request_kind="player_opening", player_interaction={"call": call})
        snapshot["other_npcs"] = [self.player_target()]
        self._submit_player_message(npc, "", snapshot)
        if npc.request_pending:
            self.game.set_reply("")
            clear_call(npc)
            return ""
        return "Could not start this call."

    def offer_player_item(self, recipient, item, quantity):
        from player import start_approach
        if self.simulation.paused:
            raise ValueError("Resume time before giving.")
        if self.player_owns(recipient) and not recipient.request_pending and not recipient.busy:
            self.simulation.player_interaction["gift"] = dict(item=item, quantity=quantity)
            self.open_player_dialogue(recipient)
            return
        if self.game.location_menu or self.game.call_choice or self.screen != "game":
            raise ValueError("Close the current blocking menu before giving.")
        blocked = self.simulation.player_menu_blocked
        self.simulation.player_menu_blocked = False
        try:
            start_approach(self.simulation, recipient, gift=dict(item=item, quantity=quantity))
        finally:
            self.simulation.player_menu_blocked = blocked

    def accept_gift_offer(self):
        from player import PlayerHandoff, validate_transfer
        offer = self.game.pending_offer
        npc = self.game.selected_npc
        if not offer or not npc or self.simulation.paused:
            return
        try:
            if (offer["npc_id"] != npc.id or offer["revision"] != npc.command_revision
                    or self.game.player_record is None or offer["record_id"] != self.game.player_record.id):
                raise ValueError("The conversation changed. A fresh offer is needed.")
            validate_transfer(self.simulation, npc, self.simulation.player, offer["item"], offer["quantity"])
            PlayerHandoff(self.simulation, npc, self.simulation.player, offer["item"],
                          offer["quantity"], self.game.player_record)
            self.game.pending_offer = None
            self.game.show_save_message("Gift accepted; the handoff is in progress.")
        except ValueError as exc:
            self.game.pending_offer = None
            self.game.show_save_message(f"Offer unavailable: {exc}")

    def decline_gift_offer(self):
        npc, offer = self.game.selected_npc, self.game.pending_offer
        self.game.pending_offer = None
        self.game.show_save_message("Gift declined; no possessions changed hands.")
        if npc and offer and not self.pending_player_request(npc):
            snapshot = self.player_snapshot(npc)
            snapshot.update(request_kind="player_acknowledgment", player_interaction={"declined_offer": offer})
            self._submit_player_message(npc, "", snapshot)

    def retry_player_dialogue(self):
        npc = self.game.selected_npc
        request = self.player_request(npc) if npc else None
        if npc is self.simulation.snake and request and not request["pending"]:
            self.submit_snake(request["message"])
            return
        if (not request or request["pending"] or npc.request_pending or npc.busy
                or self.simulation.paused or self.game.player_record is not request["record"]):
            return
        message = request["message"]
        snapshot = self.player_snapshot(npc, message)
        old = request["snapshot"]
        if old.get("request_kind") in ("player_opening", "player_gift", "player_acknowledgment"):
            snapshot.update(request_kind=old["request_kind"], player_interaction=old["player_interaction"])
            if old["player_interaction"].get("call"):
                from player import work_choices
                snapshot["owned_works"] = work_choices(self.simulation, npc.id, old["player_interaction"]["call"]["reference"])
        else:
            snapshot["other_npcs"] = [{**other.snapshot(self.tree, private=False),
                "available_to_talk": not other.busy and not other.request_pending and not other.selected and not other.player_call}
                for other in self.simulation.npcs if other is not npc and other.location == npc.location] + [self.player_target()]
        if self.game.player_record:
            snapshot["player_dialogue"] = [{"speaker": t.speaker, "text": t.text} for t in self.game.player_record.turns[-12:]]
        self._submit_player_message(npc, message, snapshot)

    def player_request(self, npc):
        """Latest direct request for this actor, retained for retry after completion."""
        return next((request for request in self.player_requests.values() if request["npc"] is npc), None)

    def pending_player_request(self, npc):
        request = self.player_request(npc)
        return request if request and request["pending"] else None

    def _submit_player_message(self, npc, message, snapshot, *, purpose="chat"):
        if not self.player_owns(npc):
            return
        previous = self.player_request(npc)
        if previous and previous["pending"]:
            return
        kind = snapshot.get("request_kind")
        if kind in ("player_opening", "player_gift", "player_acknowledgment"):
            snapshot.pop("last_reply", None)
        request_id = uuid4().hex
        if not self.requests.submit(
            self.session_id, self.provider, self.api_key, message, snapshot,
            request_id=request_id, ollama_model=self.ollama_model,
            voice_rewrites=self.voice_rewrites, dialogue_validation=self.dialogue_validation,
            queue_if_busy=True,
        ):
            self.game.set_reply("Could not start the request. Please try again.")
            return
        npc.request_pending = True
        if self.game.player_record is None:
            topic = message or ("Player gift offer" if kind == "player_gift" else
                                "Gift acknowledgment" if kind == "player_acknowledgment" else "Resident initiated conversation")
            self.game.player_record = ConversationRecord(("You", npc.name), topic, "Waiting for reply", purpose=purpose,
                origin="autonomous" if kind == "player_opening" else "player", participant_ids=[self.simulation.player.id, npc.id])
            self.game.player_record.intended_subjects = {
                "You": message or "Listening to a resident's generated opening.",
                npc.name: (snapshot["player_interaction"]["call"]["reason"] if kind == "player_opening" else
                           concern_description(npc, npc.current_concern) if isinstance(npc, NPC)
                           else "Responding to the player's message.")}
            if kind == "player_gift":
                incoming = snapshot["player_interaction"]["incoming_gift"]
                self.game.player_record.intended_subjects["You"] = f'Offering {incoming["quantity"]} × {incoming["title"]}.'
            self.conversation_history.append(self.game.player_record)
        record = self.game.player_record
        self.simulation.player_interaction["record_id"] = record.id
        self.simulation.player_interaction["pending_request"] = dict(message=message, kind=snapshot.get("request_kind", ""),
            context=snapshot.get("player_interaction", {}))
        snapshot["_ownership_id"] = self.simulation.player_interaction["id"]
        if previous:
            self.player_requests.pop(previous["id"])
        self.player_requests[request_id] = dict(id=request_id, npc=npc, snapshot=snapshot, message=message,
            record=record, revision=npc.command_revision, world_revision=self.simulation.roster_revision, pending=True)
        record.status = "Waiting for reply"
        record.error = ""
        if message and not (record.turns and record.turns[-1].speech == ("You", message)
                and len(record.turns) - 1 not in record.accepted_player_turns):
            record.add_turn("You", message)
        if snapshot.get("request_kind") != "player_acknowledgment":
            self.game.prompt_input.set_value("")

    def apply_pending_action(self, action):
        npc, error, meeting = apply_pending_action(self.simulation, action,
            lambda first, second: self._interaction_spacing(first, second))
        if meeting and (self.game.selected_npc is meeting.initiator or self.game.selected_npc is meeting.recipient):
            self.game.close_dialogue()
        if error and self.game.selected_npc is npc:
            self.game.set_reply(error)

    def _interaction_spacing(self, first, second):
        return sum(max(self.name_tag_font.size(word)[0] for word in actor.name.split())
                   for actor in (first, second)) / 2 + INTERACTION_GAP

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
                if not self.simulation.join_tree(conversation.recipient, conversation.purpose, conversation.accusation):
                    raise ValueError("Tree service is unavailable. Join the line when ready.")
            except ValueError as exc:
                conversation.record.error = str(exc)
                return
            self.game.close_dialogue(terminate=False)
            return
        if conversation.active or conversation.turns or self.requests.busy:
            return
        try:
            replacement = self.simulation.start_conversation(
                conversation.initiator, conversation.recipient, conversation.topic,
                origin=conversation.origin, purpose=conversation.purpose, opening_kind=conversation.opening_kind,
                focus_claim_id=conversation.focus_claim_id, goal_id=conversation.goal_id)
        except ValueError as exc:
            conversation.error = str(exc)
            return
        self.game.viewed_conversation = replacement
        self.game._conversation_page_turns = None

    def retry_arrival(self):
        self.simulation.arrival_error = ""

    def submit_snake(self, message):
        from snake import PlayerSnakeEncounter
        encounter = self.simulation.snake.encounter
        if (not isinstance(encounter, PlayerSnakeEncounter) or not self.player_owns(self.simulation.snake)
                or self.simulation.snake.request_pending or not message or encounter.outcome == "eat"):
            return
        record = encounter.record
        self.game.player_record = record
        turn = len(record.turns) - 1 if record.turns and record.turns[-1].speech == ("You", message) and len(record.turns) - 1 not in record.accepted_player_turns else len(record.turns)
        snapshot = encounter.player_snapshot(message, turn)
        self._submit_player_message(self.simulation.snake, message, snapshot, purpose="snake_player")

    def cancel_model_request(self, session_id, request_id):
        if not self.requests.cancel(session_id, request_id):
            return
        if session_id != self.session_id:
            return
        creative = self.creative_requests.pop(request_id, None)
        if creative is not None:
            npc = self.simulation.npc_by_id(creative[0])
            if npc and npc.creative_job and npc.creative_job.id == request_id:
                from apartments import relax
                relax(npc)
            return
        for npc in self.npcs:
            if npc.creative_job and npc.creative_job.id == request_id:
                from apartments import relax
                relax(npc)
                return
        reason = "Model request cancelled by player. Retry when ready."
        conversation = self.conversation_requests.pop(request_id, None)
        if conversation is not None:
            conversation.fail_dialogue(reason)
        elif request_id in self.player_requests:
            request = self.player_requests[request_id]
            npc, record = request["npc"], request["record"]
            if npc is not None:
                npc.request_pending = False
                npc.reply = reason
                npc.rewritten_reply = ""
                if self.game.selected_npc is npc:
                    self.game.set_reply(reason)
            if record is not None:
                record.status, record.error = "Failed", reason
            request["pending"] = False
            self.end_player_conversation(reason)

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        if self.screen == "game" and self.game.confirm_menu:
            self.game.handle_event(self, event)
            return
        if self.game.inventory.visible and self.screen in ("game", "stats"):
            if active_player(self.simulation) or self.game.inventory.owner is not self.simulation.player:
                self.game.inventory.handle_event(self, event)
                return
            self.game.inventory.close()
        self.screens[self.screen].handle_event(self, event)

    def schedule_creations(self, dt, *, composing=False):
        from apartments import discard_work, relax
        from constants import CREATIVE_REQUEST_SECONDS
        for job_id, (creator_id, revision, started) in tuple(self.creative_requests.items()):
            npc = self.simulation.npc_by_id(creator_id)
            job = npc.creative_job if npc else None
            worker = self.requests.workers.get((self.session_id, job_id))
            queued_seconds = worker.control.queued_seconds if worker and worker.control else 0.0
            if (not job or job.id != job_id or not self.simulation.is_creative_job(npc, job)
                    or monotonic() - started - queued_seconds > CREATIVE_REQUEST_SECONDS):
                self.requests.cancel(self.session_id, job_id)
                self.requests.finish(self.session_id, job_id, "Creative work ended or exceeded its time budget.")
                self.creative_requests.pop(job_id, None)
                if job and job.id == job_id:
                    discard_work(npc)
        queued = set()
        for npc in self.npcs:
            job = npc.creative_job
            if not job or job.accepted is not None or job.id in self.creative_requests:
                continue
            if not self.simulation.is_creative_job(npc, job):
                discard_work(npc)
                continue
            job.queue_remaining = max(0.0, job.queue_remaining - dt)
            if job.queue_remaining == 0:
                self.requests.finish(self.session_id, job.id, "No creative worker became available in time.")
                relax(npc)
                continue
            # Only this resident's authored profile enters a creative prompt.
            snapshot = {"request_kind": "create_" + job.kind,
                        "author": {key: value for key, value in npc.character.items()
                                   if key in ("Name", "Personality", "Background")}}
            self.requests.prepare(self.session_id, self.provider, snapshot, request_id=job.id, ollama_model=self.ollama_model)
            queued.add(job.id)
            reserve = 1 if self.requests.max_requests > 1 or composing else 0
            if self.creative_requests or self.requests.available_slots <= reserve:
                continue
            if self.provider == "gemini" and not self.api_key:
                self.requests.finish(self.session_id, job.id, "No API key is configured for creation generation.")
                relax(npc)
                continue
            if self.requests.submit(self.session_id, self.provider, self.api_key, "Create a private work.", snapshot,
                    request_id=job.id, ollama_model=self.ollama_model, voice_rewrites=False):
                self.creative_requests[job.id] = (npc.id, job.revision, monotonic())
            else:
                self.requests.finish(self.session_id, job.id, "Could not start creative generation.")
                relax(npc)
        return queued

    def update(self, dt):
        if self.screen != "stats" or self.stats.tab != "Creations":
            self.stats.gallery.stop()
        self.stats.gallery.update()
        if self.simulation:
            self.game.update_physical(self, min(dt, 0.1) if not self.simulation.paused else 0.0)
            for request_id, request in tuple(self.player_requests.items()):
                if request["pending"] and not self.player_owns(request["npc"]):
                    self.requests.cancel(self.session_id, request_id)
                    request["pending"] = False
        paused = self.simulation is not None and self.simulation.paused
        simulation_active = self.screen in ("game", "stats") and not paused and not self.game.confirm_menu
        session_active = self.screen in ("game", "stats", "save_sim")
        active_session = self.session_id if session_active else None
        # Live dialogue and Stats share the version toggle and reading hold.
        viewed = self.game.viewed_conversation if self.screen in ("game", "stats") and self.game.dialogue_open else None
        # Leave completed replies queued so they cannot change a paused world.
        replies = () if session_active and paused else self.requests.poll(active_session)
        if session_active and not paused:
            # A progressing exchange gets a fresh wait for its next provider call.
            for (session_id, request_id), worker in self.requests.workers.items():
                pending = self.conversation_requests.get(request_id)
                if pending is not None and worker.control and worker.control.stage == "Waiting for provider admission":
                    pending.wait_elapsed = 0.0
                if (session_id == active_session and pending is not None
                        and getattr(pending, "request_stage", None) != worker.stage):
                    pending.request_stage = worker.stage
                    pending.wait_elapsed = 0.0
        for request_id, result, error, failed_turns, rewritten in replies:
            creative = self.creative_requests.pop(request_id, None)
            if creative is not None:
                from creations import accept_creation_result
                error = accept_creation_result(self.simulation, request_id, creative, result, error)
                self.requests.finish(active_session, request_id, error)
                continue
            conversation = self.conversation_requests.pop(request_id, None)
            if conversation is not None:
                error = (accept_generated_dialogue(conversation, result, error, failed_turns, rewritten,
                            reading=conversation is viewed, dialogue_validation=self.dialogue_validation)
                         if conversation.id == request_id else "The conversation ended before the reply arrived.")
                self.requests.finish(active_session, request_id, error)
                continue
            request = self.player_requests.get(request_id)
            if request is None or not request["pending"]:
                self.requests.finish(active_session, request_id, "The player request is no longer active.")
                continue
            request["pending"] = False
            npc, record = request["npc"], request["record"]
            if self.simulation.player_interaction:
                self.simulation.player_interaction.pop("pending_request", None)
            if npc is self.simulation.snake:
                accepted = accept_snake_reply(self.simulation, request, result, error, failed_turns, rewritten)
                if self.game.selected_npc is npc:
                    self.game.set_reply(npc.reply)
                self.requests.finish(active_session, request_id, accepted.error)
                if accepted.error:
                    self.end_player_conversation(accepted.error)
                continue
            if npc is None or self.simulation.npc_by_id(npc.id) is not npc:
                self.requests.finish(active_session, request_id, "The NPC is no longer available.")
                continue
            visible = (self.game.dialogue_open and self.game.selected_npc is npc
                       and self.game.viewed_conversation is None)
            accepted = accept_resident_reply(self.simulation, request, result, error, rewritten,
                dialogue_validation=self.dialogue_validation, visible=visible,
                record_open=self.game.player_record is record)
            if not accepted.error and record is not None and visible:
                self.game.player_record = record
                self.game.pending_offer = accepted.pending_offer
            if self.game.selected_npc is npc and self.game.viewed_conversation is None:
                self.game.set_reply(npc.reply)
            self.requests.finish(active_session, request_id, accepted.error)
            if accepted.invitation:
                self.accept_player_invitation(accepted.invitation)
            elif accepted.error:
                self.end_player_conversation(accepted.error)
        for acknowledgment in list(self.simulation.player_acknowledgments) if self.simulation else []:
            if self.requests.busy or paused or not simulation_active:
                break
            npc = self.simulation.npc_by_id(acknowledgment["npc_id"])
            if npc and (npc.busy or npc.request_pending or npc.unread_player_reply
                        or self.game.dialogue_open and self.game.selected_npc is not npc):
                continue
            record = next(r for r in self.conversation_history if r.id == acknowledgment["record_id"])
            if npc and self.player_owns(npc):
                snapshot = self.player_snapshot(npc)
                snapshot.update(request_kind="player_acknowledgment", player_interaction={
                    "committed_transfer": next(t for t in record.transfers if t["id"] == acknowledgment["transfer_id"])})
                previous = self.game.player_record
                self.game.player_record = record
                self._submit_player_message(npc, "", snapshot)
                if not self.game.dialogue_open or self.game.selected_npc is not npc:
                    self.game.player_record = previous
                if not npc.request_pending:
                    continue
            self.simulation.player_acknowledgments.remove(acknowledgment)
        self.game.prompt_input.update(dt)
        self.settings.key_input.update(dt)
        self.settings.update(self)
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
            from snake import PlayerSnakeEncounter
            if self.simulation.snake.encounter and not isinstance(self.simulation.snake.encounter, PlayerSnakeEncounter):
                conversations.append(self.simulation.snake.encounter)
            # Player conversations get the first free slot. While composing,
            # leave one slot free and let other conversations use the rest.
            composing = self.screen == "game" and self.game.prompt_input.active and not self.simulation.phase_pending
            for conversation in sorted(conversations, key=lambda interaction: interaction.origin != "player"):
                self.request_conversation_dialogue(conversation,
                    start=not composing or self.requests.available_slots > 1)
            from constants import MAX_SIMULATION_STEP_SECONDS
            queued_creations = self.schedule_creations(min(MAX_SIMULATION_STEP_SECONDS, max(0.0, dt)), composing=composing)
            self.requests.expire_queued(self.session_id, queued_creations | {
                conversation.id for conversation in conversations
                if conversation.accepts_response and conversation.arrived and conversation.dialogue_state == "waiting"
            })

    def draw(self):
        self.screens[self.screen].draw(self)
        if self.screen == "game":
            self.game.draw_call_choice(self)
        if (self.screen in ("game", "stats") and self.game.inventory.visible
                and (active_player(self.simulation) or self.game.inventory.owner is not self.simulation.player)):
            self.game.inventory.draw(self)
        if self.screen == "game":
            self.game.draw_menu_confirmation(self)
            self.game.draw_bed_popup(self)
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
            self.stats.gallery.close()
            self.game.inventory.close()


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
