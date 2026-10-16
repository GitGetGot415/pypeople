import os
from dotenv import load_dotenv

import math
from pathlib import Path

import pygame

from character_data import load_character, random_character
from conversation_history import ConversationRecord, describe_opinion_change
from npc import Conversation, NPC, Tree, assign_npc_names
from ollama_manager import load_model
from simulation_data import InvalidSaveName, SaveStore, capture_simulation, restore_simulation
from simulation import PendingAction, Simulation

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
from .theme import FPS, WINDOW_SIZE
from .world import CHARACTER_SCALE


JUDY_CHARACTER_PATH = Path(__file__).resolve().parent.parent / "characters" / "judy.json"


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

        # Load the Gemini API key from the project's .env
        load_dotenv(
            Path(__file__).resolve().parent.parent / ".env"
        )

        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.provider = "gemini"
        self.ollama_model = load_model()
        self.session_id = 0
        self.npcs = []
        self.request_npc = None
        self.request_conversation = None
        self.request_message = ""
        self.request_history = None
        self.conversation_history = []
        self.tree = None
        self.simulation = None
        self.request_revision = None
        self.save_store = SaveStore()
        self.save_id = None
        self.save_identity = None
        self.save_name_refresh_remaining = 0
        self.simulation_name = "Unsaved simulation"
        self.requests = RequestManager()
        self.menu = MenuScreen()
        self.load_sim = LoadSimScreen()
        self.save_sim = SaveSimScreen()
        self.settings = SettingsScreen()
        self.setup = SetupScreen()
        self.character = CharacterScreen()
        self.game = GameScreen()
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

    def start_game(self):
        try:
            characters = [load_character(JUDY_CHARACTER_PATH)]
            characters[0]["Opinions"] = {}
            # Each simulation owns its profiles and opinions inside its save file.
            for index, source in enumerate(self.setup.sources, start=1):
                try:
                    characters.append(random_character() if source is None else load_character(source))
                except (OSError, ValueError) as error:
                    raise ValueError(f"NPC {index + 1}: {error}") from error
                if source is not None and Path(source).resolve() == JUDY_CHARACTER_PATH.resolve():
                    characters[-1]["Opinions"] = {}
        except (OSError, ValueError) as error:
            self.setup.message = f"Could not start game: {error}"
            return
        self.setup.message = ""
        self.setup.count_input.active = False
        self.session_id += 1
        self.game.reset()
        self.request_npc = None
        self.request_conversation = None
        self.request_history = None
        self.conversation_history = []
        self.npcs = []
        columns = max(1, math.ceil(math.sqrt((len(characters) - 1) / 2)))
        rows = max(1, math.ceil((len(characters) - 1) / (columns * 2)))
        for index, character in enumerate(characters):
            if index == 0:
                position = (240, 230)
            else:
                # Spawn on either side, leaving the central tree and its approach clear.
                side, slot = (index - 1) % 2, (index - 1) // 2
                column, row = slot % columns, slot // columns
                position = (120 + side * 500 + column * 200 / max(1, columns - 1),
                            150 + row * 260 / max(1, rows - 1))
            npc = NPC(character=character, x=position[0], y=position[1])
            sprite = self.character.renderer.render(character, npc.facing, CHARACTER_SCALE)
            npc.sprite_size = sprite.get_size()
            self.npcs.append(npc)
        assign_npc_names(self.npcs)
        self.tree = Tree()
        self.simulation = Simulation(self.npcs, self.tree)
        self.save_id = None
        self.save_identity = None
        self.simulation_name = "Unsaved simulation"
        self.screen = "game"
        self.menu.message = ""

    def sync_save_name(self):
        if self.save_id is not None and self.save_identity is not None:
            name = self.save_store.current_name(self.save_id, self.save_identity)
            if name != self.save_id:
                self.game.save_message_remaining = 0
            self.save_id = name
            self.simulation_name = self.save_id

    def request_save(self):
        if self.save_id is not None:
            self.save_current_simulation()
            return
        try:
            name = self.save_store.next_name()
        except OSError:
            self.game.show_save_message("Could not read saves. Check folder permissions and try again.")
            return
        self.game.prompt_input.active = False
        self.game.prompt_input.backspace_held = False
        self.save_sim.open(name)
        self.screen = "save_sim"

    def save_current_simulation(self, *, name=None):
        try:
            self.sync_save_name()
            if self.save_id is not None and self.save_identity is not None:
                try:
                    identity = self.save_store.file_identity(self.save_id)
                except FileNotFoundError:
                    identity = self.save_identity
                if identity != self.save_identity:
                    raise ValueError("The save file was replaced outside the game.")
            state = capture_simulation(self)
            self.save_id = self.save_store.save(state, save_id=self.save_id, name=name)
            self.simulation_name = self.save_id
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
            simulation, history = restore_simulation(saved.state)
            for npc in simulation.npcs:
                sprite = self.character.renderer.render(npc.character, npc.facing, CHARACTER_SCALE)
                npc.sprite_size = sprite.get_size()
        except (OSError, ValueError):
            self.load_sim.message = "Could not load this simulation. The save may be damaged or unavailable."
            return False
        self.session_id += 1
        self.game.reset()
        self.stats.open()
        self.request_npc = None
        self.request_conversation = None
        self.request_history = None
        self.request_message = ""
        self.simulation = simulation
        self.npcs, self.tree, self.conversation_history = simulation.npcs, simulation.tree, history
        self.save_id, self.simulation_name = save_id, saved.name
        self.save_identity = identity
        self.game.camera_offset[:] = saved.state.camera_offset
        self.game.camera_zoom = saved.state.camera_zoom
        self.screen = "game"
        self.game.show_save_message(f"Loaded {saved.name}.")
        return True

    def leave_game(self):
        for npc in self.npcs:
            npc.request_pending = any(action.npc_id == npc.id for action in self.simulation.pending_actions)
        if self.request_history is not None:
            self.request_history.status = "Ended before the reply arrived"
            self.request_history.outcomes = ["Left the game before a reply was received."]
            self.request_history = None
        self.session_id += 1
        self.screen = "menu"
        self.menu.message = ""
        self.game.close_dialogue()
        for npc in self.npcs:
            if npc.conversation is not None:
                npc.conversation.finish(self.tree)

    def open_settings(self):
        self.screen = "settings"
        self.settings.open(self.provider, self.api_key, self.ollama_model)

    def open_stats(self):
        self.game.prompt_input.active = False
        self.game.prompt_input.backspace_held = False
        self.stats.open()
        self.screen = "stats"

    def submit(self):
        message = self.game.prompt_input.text.strip()[:300]
        npc = self.game.selected_npc
        if self.screen != "game" or npc is None or npc.conversation is not None or npc.gift is not None or npc.eating or npc.request_pending or not message or self.requests.busy:
            return
        if self.provider == "gemini" and not self.api_key:
            self.game.set_reply("I think you forgot to add your API key in Settings...")
            return
        snapshot = npc.snapshot(self.tree)
        snapshot["other_npcs"] = [
            {**other.snapshot(self.tree), "available_to_talk": other.conversation is None and other.gift is None and not other.eating and not other.request_pending}
            for other in self.npcs if other is not npc
        ]
        if not self.requests.submit(
            self.session_id, self.provider, self.api_key, message, snapshot,
            ollama_model=self.ollama_model,
        ):
            self.game.set_reply("Could not start the request. Please try again.")
            return
        self.request_npc = npc
        self.request_revision = npc.command_revision
        npc.request_pending = True
        self.request_conversation = None
        self.request_message = message
        self.request_history = ConversationRecord(
            ("You", npc.name), message, "Waiting for reply", turns=[("You", message)]
        )
        self.conversation_history.append(self.request_history)
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
            spacing = (self.small_font.size(npc.name)[0] + self.small_font.size(target.name)[0]) / 2 + 12
            conversation = Conversation(npc, target, topic, self.tree, spacing=spacing)
        except ValueError as error:
            return str(error)
        self.conversation_history.append(conversation.record)
        if self.game.selected_npc is npc or self.game.selected_npc is target:
            self.game.close_dialogue()
        return None

    def apply_pending_action(self, action):
        npc = self.simulation.npc_by_id(action.npc_id)
        if npc is None:
            return
        npc.request_pending = False
        error = ""
        if npc.conversation is not None or npc.gift is not None or npc.eating or npc.command_revision != action.revision:
            error = "That request is no longer available because the NPC's activity changed."
        elif action.action == "TALK_TO_NPC":
            target = self.simulation.npc_by_id(action.target_id)
            error = self.start_conversation(npc, target.name, action.topic) if target else "That person is unavailable."
        elif action.action == "GIVE_ITEM":
            target = self.simulation.npc_by_id(action.target_id)
            try:
                record = self.conversation_history[action.record] if action.record is not None else None
                spacing = (self.small_font.size(npc.name)[0] + self.small_font.size(target.name)[0]) / 2 + 12 if target else 0
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
                record.outcomes = [error]
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
            npc.reply = error
            npc.set_emotion("neutral", 0)
            if self.game.selected_npc is npc:
                self.game.set_reply(error)

    def request_conversation_dialogue(self, conversation):
        if not conversation.active or not conversation.arrived or self.requests.busy:
            return
        if conversation.turns:
            if conversation.pending_opinions:
                conversation.remember_opinions()
            return
        if self.provider == "gemini" and not self.api_key:
            conversation.error = "Add your API key in Settings to hear their conversation."
            return
        snapshot = conversation.snapshot(self.tree, self.npcs)
        if not self.requests.submit(
            self.session_id, self.provider, self.api_key,
            f"Write what {conversation.initiator.name} and {conversation.recipient.name} say directly to each other.", snapshot,
            ollama_model=self.ollama_model,
        ):
            conversation.error = "Could not start the request. Please try again."
            return
        conversation.error = ""
        self.request_conversation = conversation
        self.request_history = None
        self.request_npc = conversation.initiator

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        self.screens[self.screen].handle_event(self, event)

    def update(self, dt):
        simulation_active = self.screen in ("game", "stats")
        active_session = self.session_id if simulation_active or self.screen == "save_sim" else None
        for result, error in self.requests.poll(active_session):
            conversation = self.request_conversation
            if conversation is not None:
                if conversation.active:
                    if error:
                        conversation.error = error
                    else:
                        conversation.set_response(result)
                continue
            npc = self.request_npc
            if npc is None:
                continue
            npc.request_pending = False
            record = self.request_history
            if not error and (npc.conversation is not None or npc.gift is not None or npc.eating or npc.command_revision != self.request_revision):
                error = "That reply arrived after the NPC became busy. Please send the request again."
            if record is not None and not error:
                record.outcomes = [
                    describe_opinion_change(
                        npc.name, entry.person, npc.opinions.get(entry.person, {}).get("opinion"), entry.opinion, entry.reason
                    )
                    for entry in result.opinion_updates
                ]
            if not error:
                error = self.remember_response_opinions(npc, result)
            if not error:
                if result.action != "SAY":
                    target = next((other for other in self.npcs if other.name == result.target), None)
                    if result.action == "EAT" and npc.inventory.get("fruit", 0) < 1:
                        error = "I don't have an apple to eat."
                    elif result.action in ("TALK_TO_NPC", "GIVE_ITEM") and (target is None or target.conversation is not None or target.gift is not None or target.eating or target.request_pending):
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
            npc.reply = error or result.reply
            if error:
                npc.set_emotion("neutral", 0)
            else:
                npc.set_emotion(result.emotion.name, result.emotion.intensity)
            if record is not None:
                record.turns.append((npc.name, npc.reply))
                record.error = error or ""
                record.status = "Failed" if error else ("Applying player request" if npc.request_pending else "Completed")
                if error:
                    record.outcomes = [f"The request could not be completed: {error}"]
                else:
                    record.outcomes.insert(0, "Accepted; applies on the next simulation tick." if npc.request_pending else "Replied without changing their activity.")
                self.request_history = None
            if self.game.selected_npc is npc and self.game.viewed_conversation is None:
                self.game.set_reply(npc.reply)
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
            self.simulation.advance(dt, active=simulation_active, action_handler=self.apply_pending_action)
        if simulation_active:
            for npc in self.npcs:
                conversation = npc.conversation
                if conversation is not None and conversation.initiator is npc:
                    if conversation.active and not conversation.turns and not conversation.error:
                        self.request_conversation_dialogue(conversation)

    def draw(self):
        self.screens[self.screen].draw(self)
        pygame.display.flip()

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                self.handle_event(event)
                if not self.running:
                    break
            if self.running:
                self.update(dt)
                self.draw()


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
