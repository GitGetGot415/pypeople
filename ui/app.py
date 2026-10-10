import json
import math
from pathlib import Path
from uuid import uuid4

import pygame

from character_data import load_character, random_character
from llm import is_opinion_question
from npc import Conversation, NPC, Tree, assign_npc_names

from .game import GameScreen
from .character import CharacterScreen
from .menu import MenuScreen
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
        self.api_key = ""
        self.provider = "gemini"
        self.session_id = 0
        self.npcs = []
        self.request_npc = None
        self.request_conversation = None
        self.request_message = ""
        self.tree = None
        self.requests = RequestManager()
        self.menu = MenuScreen()
        self.settings = SettingsScreen()
        self.setup = SetupScreen()
        self.character = CharacterScreen()
        self.game = GameScreen()
        self.stats = StatsScreen()
        self.ollama_setup = OllamaSetupScreen()
        self.screens = {
            "menu": self.menu,
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
            paths = [JUDY_CHARACTER_PATH]
            generated_dir = JUDY_CHARACTER_PATH.parent / "generated" / uuid4().hex
            for index, source in enumerate(self.setup.sources, start=1):
                try:
                    characters.append(random_character() if source is None else load_character(source))
                except (OSError, ValueError) as error:
                    raise ValueError(f"NPC {index + 1}: {error}") from error
                paths.append(generated_dir / f"random_{index}.json" if source is None else source)
            # Validate all profiles before saving any generated characters.
            for source, path, character in zip(self.setup.sources, paths[1:], characters[1:]):
                if source is not None:
                    continue
                generated_dir.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as character_file:
                    json.dump(character, character_file, indent=2, ensure_ascii=False)
        except (OSError, ValueError) as error:
            self.setup.message = f"Could not start game: {error}"
            return
        self.setup.message = ""
        self.setup.count_input.active = False
        self.session_id += 1
        self.game.reset()
        self.request_npc = None
        self.request_conversation = None
        self.npcs = []
        columns = max(1, math.ceil(math.sqrt(len(characters) - 1)))
        rows = max(1, math.ceil((len(characters) - 1) / columns))
        for index, (character, path) in enumerate(zip(characters, paths)):
            if index == 0:
                position = (240, 230)
            else:
                # Arrange additional NPCs beside Judy, clear of the tree.
                column, row = (index - 1) % columns, (index - 1) // columns
                position = (350 + column * 220 / max(1, columns - 1), 150 + row * 260 / max(1, rows - 1))
            npc = NPC(character=character, x=position[0], y=position[1], character_path=path)
            sprite = self.character.renderer.render(character, npc.facing, CHARACTER_SCALE)
            npc.sprite_size = sprite.get_size()
            self.npcs.append(npc)
        assign_npc_names(self.npcs)
        self.tree = Tree()
        self.screen = "game"

    def leave_game(self):
        self.session_id += 1
        self.screen = "menu"
        self.game.close_dialogue()
        for npc in self.npcs:
            if npc.conversation is not None:
                npc.conversation.finish(self.tree)

    def open_settings(self):
        self.screen = "settings"
        self.settings.open(self.provider, self.api_key)

    def open_stats(self):
        self.game.prompt_input.active = False
        self.game.prompt_input.backspace_held = False
        self.stats.open()
        self.screen = "stats"

    def submit(self):
        message = self.game.prompt_input.text.strip()[:300]
        npc = self.game.selected_npc
        if self.screen != "game" or npc is None or npc.conversation is not None or not message or self.requests.busy:
            return
        if self.provider == "gemini" and not self.api_key:
            self.game.set_reply("I think you forgot to add your API key in Settings...")
            return
        snapshot = npc.snapshot(self.tree)
        snapshot["reply_only"] = is_opinion_question(message)
        snapshot["other_npcs"] = [
            {**other.snapshot(self.tree), "available_to_talk": other.conversation is None}
            for other in self.npcs if other is not npc
        ]
        if not self.requests.submit(self.session_id, self.provider, self.api_key, message, snapshot):
            self.game.set_reply("Could not start the request. Please try again.")
            return
        self.request_npc = npc
        self.request_conversation = None
        self.request_message = message
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
        try:
            # Leave enough space for both sprites and their name labels.
            spacing = (self.small_font.size(npc.name)[0] + self.small_font.size(target.name)[0]) / 2 + 12
            Conversation(npc, target, topic, self.tree, spacing=spacing)
        except ValueError as error:
            return str(error)
        if self.game.selected_npc is npc or self.game.selected_npc is target:
            self.game.close_dialogue()
        return None

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
        if not self.requests.submit(self.session_id, self.provider, self.api_key, "Let me hear their conversation.", snapshot):
            conversation.error = "Could not start the request. Please try again."
            return
        conversation.error = ""
        self.request_conversation = conversation
        self.request_npc = conversation.initiator

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        self.screens[self.screen].handle_event(self, event)

    def update(self, dt):
        simulation_active = self.screen in ("game", "stats")
        active_session = self.session_id if simulation_active else None
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
            if not error:
                error = self.remember_response_opinions(npc, result)
            if not error:
                if result.action == "TALK_TO_NPC":
                    error = self.start_conversation(npc, result.target, self.request_message)
                else:
                    npc.apply_action(result.action, self.tree)
            npc.reply = error or result.reply
            if self.game.selected_npc is npc and self.game.viewed_conversation is None:
                self.game.set_reply(npc.reply)
        self.game.prompt_input.update(dt)
        self.settings.key_input.update(dt)
        self.character.update(dt)
        if self.screen == "setup":
            self.setup.update(dt)
        if simulation_active:
            self.tree.update(dt)
            for npc in self.npcs:
                npc.update(dt, self.tree)
                if npc.conversation is not None and npc.conversation.initiator is npc:
                    npc.conversation.update(dt)
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
