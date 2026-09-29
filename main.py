from __future__ import annotations

import queue
import sys
import threading

try:
    import pygame
    from pydantic import ValidationError
    from llm import ActionResponse, RequestError, request_action
    from npc import NPC, Tree, NPC_RADIUS, WORLD_WIDTH, WORLD_HEIGHT
    from ui import Button, TextInput, draw_wrapped_text, INK, MUTED, PAPER
except ImportError:
    DEPENDENCIES_MISSING = True
else:
    DEPENDENCIES_MISSING = False

WINDOW_SIZE = (960, 640)
FPS = 60
GRASS = (172, 199, 137)

BORDER_RADIUS = 0

def _request_worker(session_id, api_key, message, snapshot, results):
    """Only communicate through the queue; never touch Pygame or live state."""
    try:
        result = request_action(api_key, message, snapshot)
        results.put((session_id, result.model_dump_json(), None))
    except RequestError as exc:
        results.put((session_id, None, str(exc)))
    except Exception:
        results.put((session_id, None, "Could not get a reply from Gemini. Please try again."))


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
        self.session_id = 0
        self.worker = None
        self.pending_session = None
        self.results = queue.Queue()
        self.npc = None
        self.tree = None
        self.dialogue_open = False
        self.reply = ""

        self.start_button = Button((340, 300, 280, 54), "Start")
        self.settings_button = Button((340, 372, 280, 54), "Settings")
        self.exit_button = Button((340, 444, 280, 54), "Exit")
        self.save_button = Button((340, 400, 280, 54), "Save & Back")
        self.key_input = TextInput(
            (180, 280, 600, 52),
            max_length=512,
            secret=True,
            strip_paste=True,
            placeholder="Paste your key here",
        )
        self.menu_button = Button((824, 16, 112, 38), "Menu")
        self.send_button = Button((816, 564, 120, 44), "Send")
        self.prompt_input = TextInput(
            (24, 564, 778, 44), placeholder="[insert cool action here]"
        )

    @property
    def busy(self):
        # Keep the slot reserved until the queue is polled AND the thread exits.
        return self.worker is not None

    def start_game(self):
        self.session_id += 1
        self.npc = NPC()
        self.tree = Tree()
        self.dialogue_open = False
        self.prompt_input.set_value("")
        self.prompt_input.active = False
        self.reply = ""
        self.screen = "game"

    def leave_game(self):
        self.session_id += 1
        self.screen = "menu"
        self.dialogue_open = False
        self.prompt_input.active = False

    def open_settings(self):
        self.screen = "settings"
        self.key_input.set_value(self.api_key)
        self.key_input.focus()

    def set_reply(self, text):
        normalized = " ".join(text.split())
        self.reply = "".join(char for char in normalized if char.isprintable())[:160]

    def submit(self):
        message = self.prompt_input.text.strip()[:300]
        if self.screen != "game" or not self.dialogue_open or not message or self.busy:
            return
        if not self.api_key:
            self.set_reply("I think you forgot to add your Gemini API key in Settings...")
            return
        self.pending_session = self.session_id
        self.worker = threading.Thread(
            target=_request_worker,
            args=(self.session_id, self.api_key, message, self.npc.snapshot(self.tree), self.results),
            daemon=True,
            name="Judy-Gemini-request",
        )
        try:
            self.worker.start()
        except RuntimeError:
            self.worker = None
            self.pending_session = None
            self.set_reply("Could not start the request. Please try again.")
            return
        self.prompt_input.set_value("")

    def poll_results(self):
        while True:
            try:
                session_id, raw, error = self.results.get_nowait()
            except queue.Empty:
                break
            if self.screen != "game" or session_id != self.session_id:
                continue
            if error:
                self.set_reply(error)
                continue
            try:
                # Revalidate at the main-thread boundary before touching the world.
                result = ActionResponse.model_validate_json(raw)
                if not result.reply.strip():
                    raise ValueError("Empty reply")
            except (ValidationError, ValueError, TypeError):
                self.set_reply("Gemini returned an invalid action or reply. Try again.")
                continue
            self.npc.apply_action(result.action, self.tree)
            self.set_reply(result.reply)
        if self.worker is not None and not self.worker.is_alive():
            # A result can arrive between draining the queue and checking the
            # thread. Process that result next frame before freeing the slot.
            if self.results.empty():
                self.worker = None
                self.pending_session = None

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        escape = event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        if self.screen == "menu":
            if self.start_button.clicked(event):
                self.start_game()
            elif self.settings_button.clicked(event):
                self.open_settings()
            elif self.exit_button.clicked(event):
                self.running = False
        elif self.screen == "settings":
            if escape:
                self.key_input.set_value(self.api_key)
                self.key_input.active = False
                self.screen = "menu"
            elif self.save_button.clicked(event):
                self.api_key = self.key_input.text.strip()
                self.key_input.set_value(self.api_key)
                self.key_input.active = False
                self.screen = "menu"
            else:
                self.key_input.handle_event(event)
        elif self.screen == "game":
            if self.menu_button.clicked(event):
                self.leave_game()
                return
            if escape:
                if self.dialogue_open:
                    self.dialogue_open = False
                    self.prompt_input.active = False
                else:
                    self.leave_game()
                return
            if (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and event.pos[1] < WORLD_HEIGHT
                and self.npc.contains(event.pos)
            ):
                if not self.dialogue_open:
                    self.dialogue_open = True
                    self.set_reply("Hi! What should I do today?")
                self.prompt_input.focus()
                return
            if self.dialogue_open:
                if self.send_button.clicked(event):
                    self.submit()
                    self.prompt_input.focus()
                elif self.prompt_input.handle_event(event):
                    self.submit()

    def update(self, dt):
        self.poll_results()
        self.prompt_input.update(dt)
        self.key_input.update(dt)
        if self.screen == "game":
            self.npc.update(dt, self.tree)

    def _center_text(self, text, font, y, color=None):
        rendered = font.render(text, True, INK if color is None else color)
        self.window.blit(rendered, rendered.get_rect(midtop=(WINDOW_SIZE[0] // 2, y)))

    def draw_menu(self):
        self.window.fill(GRASS)
        pygame.draw.rect(self.window, PAPER, (240, 110, 480, 470), border_radius=BORDER_RADIUS)
        self._center_text("Pypeople", self.title_font, 155)
        self._center_text("say whatever you want", self.font, 245, MUTED)
        for button in (self.start_button, self.settings_button, self.exit_button):
            button.draw(self.window, self.font)
        self._center_text("(no you can't flirt with the ai yet)", self.small_font, 541, MUTED)

    def draw_settings(self):
        self.window.fill(GRASS)
        pygame.draw.rect(self.window, PAPER, (120, 120, 720, 405), border_radius=BORDER_RADIUS)
        self._center_text("Settings", self.heading_font, 160)
        self.window.blit(self.font.render("Gemini API key", True, INK), (180, 243))
        self.key_input.draw(self.window, self.font)
        self.window.blit(self.small_font.render("Stored only for this session.", True, MUTED), (180, 345))
        self.save_button.draw(self.window, self.font)
        self._center_text("Get a key at Google AI Studio: aistudio.google.com/apikey", self.small_font, 475, MUTED)
        self._center_text(self.key_input.error or "Ctrl+V to paste. Escape discards your edits.", self.small_font, 553, INK)

    def draw_game(self):
        self.window.fill(GRASS)
        tree_x = round(self.tree.x) + self.tree.offset
        tree_y = round(self.tree.y)
        pygame.draw.ellipse(self.window, (145, 171, 111), (tree_x - 43, tree_y + 50, 86, 20))
        pygame.draw.rect(self.window, (134, 89, 55), (tree_x - 13, tree_y - 12, 26, 76))
        pygame.draw.circle(self.window, (68, 124, 66), (tree_x, tree_y - 38), 53)
        pygame.draw.circle(self.window, (83, 141, 72), (tree_x - 11, tree_y - 49), 31)
        judy = (round(self.npc.x), round(self.npc.y))
        pygame.draw.circle(self.window, (88, 156, 214), judy, NPC_RADIUS)
        pygame.draw.circle(self.window, (41, 89, 133), judy, NPC_RADIUS, width=2)
        name = self.small_font.render("Judy", True, INK)
        name_rect = name.get_rect(midtop=(judy[0], judy[1] + NPC_RADIUS + 6))
        name_rect.clamp_ip(pygame.Rect(0, 0, WORLD_WIDTH, WORLD_HEIGHT))
        self.window.blit(name, name_rect)
        self.window.blit(self.small_font.render("Pypeople", True, INK), (24, 26))
        self.menu_button.draw(self.window, self.small_font)

        pygame.draw.rect(self.window, PAPER, (0, WORLD_HEIGHT, 960, 180))
        pygame.draw.line(self.window, (129, 154, 110), (0, WORLD_HEIGHT), (960, WORLD_HEIGHT), 3)
        status = self.npc.status
        if self.busy:
            prefix = "Thinking..." if self.pending_session == self.session_id else "Finishing earlier request..."
            status = f"{prefix} / {status}"
        rendered_status = self.small_font.render(status, True, MUTED)
        self.window.blit(rendered_status, rendered_status.get_rect(topright=(936, 478)))
        if not self.dialogue_open:
            self.window.blit(self.heading_font.render("So far we have one AI", True, INK), (24, 494))
            self.window.blit(self.font.render("She's represented by the blue circle on the screen", True, MUTED), (24, 538))
            self.window.blit(self.small_font.render("if you click her you can talk to her", True, MUTED), (24, 602))
            return
        self.window.blit(self.font.render("Judy", True, INK), (24, 477))
        draw_wrapped_text(self.window, self.reply, self.font, INK, (24, 500, 912, 58))
        self.prompt_input.draw(self.window, self.font)
        self.send_button.draw(self.window, self.font, enabled=not self.busy)
        hint = self.prompt_input.error or "Enter to send. Ctrl+V to paste. Escape closes this conversation."
        self.window.blit(self.small_font.render(hint, True, MUTED), (24, 616))
        count = self.small_font.render(f"{len(self.prompt_input.text)}/300", True, MUTED)
        self.window.blit(count, count.get_rect(topright=(936, 616)))

    def draw(self):
        if self.screen == "menu":
            self.draw_menu()
        elif self.screen == "settings":
            self.draw_settings()
        else:
            self.draw_game()
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


def main():
    if sys.version_info < (3, 11):
        print("Pypeople requires Python 3.11 or newer.")
        return 1
    if DEPENDENCIES_MISSING:
        print("Pypeople needs pygame, google-genai, pydantic 2, and pyperclip.")
        print("In the project folder, run: python -m pip install --user -r requirements.txt")
        return 1
    try:
        pygame.display.init()
        pygame.font.init()
        pygame.key.start_text_input()
        Application().run()
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
