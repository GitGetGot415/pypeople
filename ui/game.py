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
        self.selected_npc = None
        self.reply = ""

    def reset(self):
        self.close_dialogue()
        self.prompt_input.set_value("")
        self.reply = ""

    def close_dialogue(self):
        if self.selected_npc is not None:
            self.selected_npc.set_selected(False)
            self.selected_npc = None
        self.dialogue_open = False
        self.prompt_input.active = False
        self.prompt_input.backspace_held = False

    def set_reply(self, text):
        normalized = " ".join(text.split())
        self.reply = "".join(char for char in normalized if char.isprintable())[:160]
        if self.selected_npc is not None:
            self.selected_npc.reply = self.reply

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
        ):
            npc = next((npc for npc in sorted(app.npcs, key=lambda npc: npc.y, reverse=True) if npc.contains(event.pos)), None)
            if npc is None or npc is self.selected_npc:
                self.close_dialogue()
            else:
                self.close_dialogue()
                self.dialogue_open = True
                self.selected_npc = npc
                npc.set_selected(True)
                self.prompt_input.set_value("")
                self.set_reply(npc.reply or f"Hi, I'm {npc.name}! What should I do today?")
                self.prompt_input.focus()
            return
        if self.dialogue_open:
            if self.send_button.clicked(event):
                app.submit()
                self.prompt_input.focus()
            elif self.prompt_input.handle_event(event):
                app.submit()

    def draw(self, app):
        draw_world(app.window, app.npcs, app.tree, app.small_font, app.character.renderer)
        app.window.blit(app.small_font.render("Pypeople", True, INK), (24, 26))
        self.menu_button.draw(app.window, app.small_font)

        pygame.draw.rect(app.window, PAPER, (0, WORLD_HEIGHT, 960, 180))
        pygame.draw.line(app.window, (129, 154, 110), (0, WORLD_HEIGHT), (960, WORLD_HEIGHT), 3)
        status = self.selected_npc.status if self.selected_npc is not None else f"{len(app.npcs)} NPCs"
        if app.requests.busy:
            prefix = (
                f"Thinking for {app.request_npc.name}..."
                if app.requests.pending_session == app.session_id and app.request_npc is not None
                else "Finishing earlier request..."
            )
            status = f"{prefix} / {status}"
        rendered_status = app.small_font.render(status, True, MUTED)
        app.window.blit(rendered_status, rendered_status.get_rect(topright=(936, 478)))
        if not self.dialogue_open:
            app.window.blit(app.heading_font.render("Meet the people in your world", True, INK), (24, 494))
            app.window.blit(app.font.render("Click anyone to start a conversation.", True, MUTED), (24, 538))
            app.window.blit(app.small_font.render("Ask them to wander, stop, or shake the tree.", True, MUTED), (24, 602))
            return
        app.window.blit(app.font.render(self.selected_npc.name, True, INK), (24, 477))
        draw_wrapped_text(app.window, self.reply, app.font, INK, (24, 500, 912, 58))
        self.prompt_input.draw(app.window, app.font)
        self.send_button.draw(app.window, app.font, enabled=not app.requests.busy)
        hint = self.prompt_input.error or "Enter to send. Ctrl+V to paste. Escape or click the world to deselect."
        app.window.blit(app.small_font.render(hint, True, MUTED), (24, 616))
        count = app.small_font.render(f"{len(self.prompt_input.text)}/300", True, MUTED)
        app.window.blit(count, count.get_rect(topright=(936, 616)))
