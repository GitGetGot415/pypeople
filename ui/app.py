import json
from pathlib import Path

import pygame

from character_data import load_character, random_character
from npc import NPC, Tree

from .game import GameScreen
from .character import CharacterScreen
from .menu import MenuScreen
from .requests import RequestManager
from .settings import SettingsScreen
from .theme import FPS, WINDOW_SIZE
from .world import CHARACTER_SCALE


JUDY_CHARACTER_PATH = Path(__file__).resolve().parent.parent / "characters" / "judy.json"
RANDOM_CHARACTER_PATHS = tuple(JUDY_CHARACTER_PATH.parent / f"random_{index}.json" for index in (1, 2))


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
        self.tree = None
        self.requests = RequestManager()
        self.menu = MenuScreen()
        self.settings = SettingsScreen()
        self.character = CharacterScreen()
        self.game = GameScreen()
        self.screens = {
            "menu": self.menu,
            "settings": self.settings,
            "character": self.character,
            "game": self.game,
        }

    def start_game(self):
        try:
            characters = [load_character(JUDY_CHARACTER_PATH)]
            generated = [random_character() for _ in RANDOM_CHARACTER_PATHS]
            for path, character in zip(RANDOM_CHARACTER_PATHS, generated):
                with path.open("w", encoding="utf-8") as character_file:
                    json.dump(character, character_file, indent=2, ensure_ascii=False)
                characters.append(load_character(path))
        except (OSError, ValueError) as error:
            self.menu.message = f"Could not start game: {error}"
            return
        self.menu.message = ""
        self.session_id += 1
        self.game.reset()
        self.request_npc = None
        self.npcs = []
        for character, position in zip(characters, ((240, 230), (430, 230), (530, 330))):
            npc = NPC(character=character, x=position[0], y=position[1])
            sprite = self.character.renderer.render(character, npc.facing, CHARACTER_SCALE)
            npc.sprite_size = sprite.get_size()
            self.npcs.append(npc)
        self.tree = Tree()
        self.screen = "game"

    def leave_game(self):
        self.session_id += 1
        self.screen = "menu"
        self.game.close_dialogue()

    def open_settings(self):
        self.screen = "settings"
        self.settings.open(self.provider, self.api_key)

    def submit(self):
        message = self.game.prompt_input.text.strip()[:300]
        npc = self.game.selected_npc
        if self.screen != "game" or npc is None or not message or self.requests.busy:
            return
        if self.provider == "gemini" and not self.api_key:
            self.game.set_reply("I think you forgot to add your API key in Settings...")
            return
        if not self.requests.submit(self.session_id, self.provider, self.api_key, message, npc.snapshot(self.tree)):
            self.game.set_reply("Could not start the request. Please try again.")
            return
        self.request_npc = npc
        self.game.prompt_input.set_value("")

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        self.screens[self.screen].handle_event(self, event)

    def update(self, dt):
        active_session = self.session_id if self.screen == "game" else None
        for result, error in self.requests.poll(active_session):
            npc = self.request_npc
            if npc is None:
                continue
            if not error:
                npc.apply_action(result.action, self.tree)
            npc.reply = error or result.reply
            if self.game.selected_npc is npc:
                self.game.set_reply(npc.reply)
        self.game.prompt_input.update(dt)
        self.settings.key_input.update(dt)
        self.character.update(dt)
        if self.screen == "game":
            self.tree.update(dt)
            for npc in self.npcs:
                npc.update(dt, self.tree)

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
