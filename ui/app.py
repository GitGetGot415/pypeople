import os
import math
import pygame
import traceback
from pydantic import ValidationError
from dotenv import load_dotenv
from pathlib import Path
from time import monotonic
from app_settings import load_settings
from character_data import load_character, random_character
from conversation_session import ConversationSession
from conversation_results import apply_pending_action
from npc import NPC, assign_npc_names
from player import active_player
from ollama_manager import load_model
from simulation_data import InvalidSaveName, SaveStore, capture_simulation, restore_simulation
from simulation import Simulation
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
                      UI_TITLE_FONT_SIZE, UI_HEADING_FONT_SIZE, UI_TEXT_FONT_SIZE, UI_SMALL_FONT_SIZE,
                      NAME_TAG_FONT_SIZE, MAX_RESIDENTS)
from .world import CHARACTER_SCALE
from .fonts import UnicodeFont


STARTING_CHARACTER_PATH = Path(__file__).resolve().parent.parent / STARTING_CHARACTER_FILE
SAVE_ERROR_LOG = Path(__file__).resolve().parent.parent / "logs" / "save_error.log"


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
        self.conversations = None
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
            self._sync_conversation_view()
            self.conversations.end_player_conversation("The session changed.")
            self._present_conversation_events()
            self.simulation.cancel_gifts("The session changed before the handoff finished.")
            self.conversations.shutdown()
        from apartments import discard_work
        for npc in self.npcs:
            if npc.creative_job:
                discard_work(npc)
        self.session_id += 1
        self.game.reset()
        self.stats.open()
        self.creative_requests.clear()
        self.simulation = simulation
        self.conversations = ConversationSession(simulation, self.requests, session_id=self.session_id,
            provider=self.provider, api_key=self.api_key, ollama_model=self.ollama_model,
            voice_rewrites=self.voice_rewrites, dialogue_validation=self.dialogue_validation)
        self.game.app = self
        self.game.location = simulation.player.location if simulation.mode == "player" else "garden"
        self.save_id, self.save_identity = save_id, save_identity
        self.screen = "game"

    @property
    def player_requests(self):
        return self.conversations.player_requests if self.conversations else {}

    @property
    def conversation_requests(self):
        return self.conversations.conversation_requests if self.conversations else {}

    def _sync_conversation_view(self):
        self.conversations.configure(provider=self.provider, api_key=self.api_key, ollama_model=self.ollama_model,
            voice_rewrites=self.voice_rewrites, dialogue_validation=self.dialogue_validation)
        self.conversations.set_view(actor=self.game.selected_npc,
            reply_open=self.game.dialogue_open, viewed_conversation=self.game.viewed_conversation,
            menu_open=bool(self.game.location_menu or self.screen != "game"))

    def _present_conversation_events(self):
        events, self.conversations.events = self.conversations.events, []
        for kind, value in events:
            if kind == "reply": self.game.set_reply(value)
            elif kind == "message": self.game.show_save_message(value)
            elif kind == "clear_input": self.game.prompt_input.set_value("")
            elif kind == "focus": self.game.prompt_input.focus()
            elif kind == "close": self.game.close_dialogue(terminate=False, notify=False)
            elif kind == "view_conversation":
                self.game.viewed_conversation = value
                self.game._conversation_page_turns = None
            elif kind == "open":
                self.game.dialogue_open, self.game.selected_npc = True, value
                self.game.viewed_conversation = None

    def start_game(self):
        try:
            sources = [STARTING_CHARACTER_PATH]
            sources.extend(self.setup.sources[1:])
            if len(sources) > MAX_RESIDENTS - int(self.setup.mode == "player"):
                raise ValueError("Too many starting NPCs for this mode.")
            characters = []
            # Both modes require Judy first and permit additional residents.
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
            state = capture_simulation(self.simulation, camera_offset=self.game.camera_offset, camera_zoom=self.game.camera_zoom)
            self.save_id = self.save_store.save(state, name=name)
            self.save_identity = self.save_store.file_identity(self.save_id)
        except (InvalidSaveName, FileExistsError) as error:
            self.save_sim.message = str(error)
            self.game.show_save_message(str(error))
            return False
        except (OSError, ValueError) as error:
            details = traceback.format_exc()
            if isinstance(error, ValidationError):
                for issue in error.errors(include_url=False):
                    value = issue.get("input")
                    if isinstance(value, dict):
                        actor = {key: value[key] for key in ("id", "display_name", "location", "x", "y", "behavior", "destination")
                                 if key in value}
                        if actor:
                            details += f"\nInvalid state at {issue['loc']}: {actor!r}\n"
            try:
                SAVE_ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
                SAVE_ERROR_LOG.write_text(details, encoding="utf-8")
            except OSError:
                print(details)
            if isinstance(error, ValidationError):
                issue = error.errors(include_input=False, include_url=False)[0]
                location = ".".join(str(part) for part in issue["loc"])
                reason = issue["msg"].removeprefix("Value error, ")
                message = f"Could not save: {location + ': ' if location else ''}{reason}"
            elif isinstance(error, OSError):
                message = f"Could not write the save file: {error.strerror or error}"
            else:
                message = f"Could not save: {error}"
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
        self._sync_conversation_view()
        self.conversations.end_player_conversation("Left the game.")
        self._present_conversation_events()
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
            npc.player_reply.complete()
        self.conversations.shutdown()
        self.session_id += 1
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


    def use_door(self, target):
        from locations import start_journey, allocated_floors, hallway_location, doorway_arrival, valid_location
        from constants import PLAYER_ROOM_NUMBER
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
        if target["kind"] == "room" and target["room"] != PLAYER_ROOM_NUMBER:
            room = target["room"]
            host = next((n for n in world.npcs if n.room_number == room), None)
            permitted = bool(world.visit and world.visit.host_id == (host.id if host else None))
            if not permitted:
                if not host or host.location != target["destination"]:
                    self.game.show_save_message("This room is vacant." if not host else "The resident is absent.")
                    return
                from player import start_approach
                try:
                    start_approach(world, host, knock_room=room)
                    self._sync_conversation_view()
                    self.conversations.open_player_dialogue(host)
                    self._present_conversation_events()
                    self.game.show_save_message("Knocking. Ask the resident for an invitation.")
                except ValueError as exc:
                    self.game.show_save_message(str(exc))
                return
        destination = target["destination"]
        start_journey(player, destination, doorway_arrival(target), purpose="doorway", fast=True)


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


    def retry_arrival(self):
        self.simulation.arrival_error = ""


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
        self._sync_conversation_view()
        self.conversations.cancel_request(request_id, "Model request cancelled by player. Retry when ready.")
        self._present_conversation_events()

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
            self._sync_conversation_view()
            self.conversations.cancel_unowned()
        paused = self.simulation is not None and self.simulation.paused
        simulation_active = self.screen in ("game", "stats") and not paused and not self.game.confirm_menu
        session_active = self.screen in ("game", "stats", "save_sim")
        active_session = self.session_id if session_active else None
        # Live dialogue and Stats share the version toggle and reading hold.
        viewed = self.game.viewed_conversation if self.screen in ("game", "stats") and self.game.dialogue_open else None
        # Leave completed replies queued so they cannot change a paused world.
        replies = () if session_active and paused else self.requests.poll(active_session)
        if session_active and not paused:
            self.conversations.update_progress(active_session)
        for request_id, result, error, failed_turns, rewritten in replies:
            creative = self.creative_requests.pop(request_id, None)
            if creative is not None:
                from creations import accept_creation_result
                error = accept_creation_result(self.simulation, request_id, creative, result, error)
                self.requests.finish(active_session, request_id, error)
                continue
            self.conversations.accept_result(request_id, result, error, failed_turns, rewritten,
                session_id=active_session, viewed=viewed)
            self._present_conversation_events()
        if self.conversations:
            if simulation_active:
                self.conversations.prepare_player_opening()
            self.conversations.schedule_acknowledgments(paused=paused, simulation_active=simulation_active)
            self._present_conversation_events()
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
                self._sync_conversation_view()
                self.conversations.request_conversation_dialogue(conversation,
                    start=not composing or self.requests.available_slots > 1)
                self._present_conversation_events()
            from constants import MAX_SIMULATION_STEP_SECONDS
            queued_creations = self.schedule_creations(min(MAX_SIMULATION_STEP_SECONDS, max(0.0, dt)), composing=composing)
            self.requests.expire_queued(self.session_id, queued_creations | {
                conversation.id for conversation in conversations
                if conversation.accepts_response and conversation.arrived and conversation.dialogue.dialogue_state == "waiting"
            })

    def draw(self):
        self.screens[self.screen].draw(self)
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
