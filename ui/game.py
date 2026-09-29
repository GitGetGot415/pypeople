import pygame

from npc import WORLD_HEIGHT

from .button import Button
from .text import draw_wrapped_text
from .text_input import TextInput
from .theme import INK, MUTED, PAPER
from .world import draw_world


class GameScreen:
    def __init__(self):
        self.menu_button = Button((824, 16, 112, 38), "Menu")
        self.send_button = Button((816, 564, 120, 44), "Send")
        self.prompt_input = TextInput(
            (24, 564, 778, 44), placeholder="[insert cool action here]"
        )
        self.dialogue_open = False
        self.reply = ""

    def reset(self):
        self.close_dialogue()
        self.prompt_input.set_value("")
        self.reply = ""

    def close_dialogue(self):
        self.dialogue_open = False
        self.prompt_input.active = False

    def set_reply(self, text):
        normalized = " ".join(text.split())
        self.reply = "".join(char for char in normalized if char.isprintable())[:160]

    def handle_event(self, app, event):
        if self.menu_button.clicked(event):
            app.leave_game()
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.dialogue_open:
                self.close_dialogue()
            else:
                app.leave_game()
            return
        if (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and event.pos[1] < WORLD_HEIGHT
            and app.npc.contains(event.pos)
        ):
            if not self.dialogue_open:
                self.dialogue_open = True
                self.set_reply("Hi! What should I do today?")
            self.prompt_input.focus()
            return
        if self.dialogue_open:
            if self.send_button.clicked(event):
                app.submit()
                self.prompt_input.focus()
            elif self.prompt_input.handle_event(event):
                app.submit()

    def draw(self, app):
        draw_world(app.window, app.npc, app.tree, app.small_font)
        app.window.blit(app.small_font.render("Pypeople", True, INK), (24, 26))
        self.menu_button.draw(app.window, app.small_font)

        pygame.draw.rect(app.window, PAPER, (0, WORLD_HEIGHT, 960, 180))
        pygame.draw.line(app.window, (129, 154, 110), (0, WORLD_HEIGHT), (960, WORLD_HEIGHT), 3)
        status = app.npc.status
        if app.requests.busy:
            prefix = "Thinking..." if app.requests.pending_session == app.session_id else "Finishing earlier request..."
            status = f"{prefix} / {status}"
        rendered_status = app.small_font.render(status, True, MUTED)
        app.window.blit(rendered_status, rendered_status.get_rect(topright=(936, 478)))
        if not self.dialogue_open:
            app.window.blit(app.heading_font.render("So far we have one AI", True, INK), (24, 494))
            app.window.blit(app.font.render("She's represented by the blue circle on the screen", True, MUTED), (24, 538))
            app.window.blit(app.small_font.render("if you click her you can talk to her", True, MUTED), (24, 602))
            return
        app.window.blit(app.font.render("Judy", True, INK), (24, 477))
        draw_wrapped_text(app.window, self.reply, app.font, INK, (24, 500, 912, 58))
        self.prompt_input.draw(app.window, app.font)
        self.send_button.draw(app.window, app.font, enabled=not app.requests.busy)
        hint = self.prompt_input.error or "Enter to send. Ctrl+V to paste. Escape closes this conversation."
        app.window.blit(app.small_font.render(hint, True, MUTED), (24, 616))
        count = app.small_font.render(f"{len(self.prompt_input.text)}/300", True, MUTED)
        app.window.blit(count, count.get_rect(topright=(936, 616)))
