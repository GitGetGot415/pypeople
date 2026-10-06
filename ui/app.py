import pygame

from npc import NPC, Tree

from .game import GameScreen
from .character import CharacterScreen
from .menu import MenuScreen
from .requests import RequestManager
from .settings import SettingsScreen
from .theme import FPS, WINDOW_SIZE


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
        self.npc = None
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
        self.session_id += 1
        self.npc = NPC()
        self.tree = Tree()
        self.game.reset()
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
        if self.screen != "game" or not self.game.dialogue_open or not message or self.requests.busy:
            return
        if self.provider == "gemini" and not self.api_key:
            self.game.set_reply("I think you forgot to add your API key in Settings...")
            return
        if not self.requests.submit(self.session_id, self.provider, self.api_key, message, self.npc.snapshot(self.tree)):
            self.game.set_reply("Could not start the request. Please try again.")
            return
        self.game.prompt_input.set_value("")

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        self.screens[self.screen].handle_event(self, event)

    def update(self, dt):
        active_session = self.session_id if self.screen == "game" else None
        for result, error in self.requests.poll(active_session):
            if error:
                self.game.set_reply(error)
            else:
                self.npc.apply_action(result.action, self.tree)
                self.game.set_reply(result.reply)
        self.game.prompt_input.update(dt)
        self.settings.key_input.update(dt)
        self.character.update(dt)
        if self.screen == "game":
            self.npc.update(dt, self.tree)

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
