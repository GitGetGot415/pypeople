import pygame

from npc import WORLD_HEIGHT, WORLD_WIDTH

from .button import Button
from .text import draw_wrapped_text, wrap_text
from .text_input import TextInput
from .theme import INK, MUTED, PAPER
from .world import draw_world


class GameScreen:
    def __init__(self):
        self.menu_button = Button((824, 16, 112, 38), "Menu")
        self.stats_button = Button((704, 16, 112, 38), "Stats")
        self.send_button = Button((816, 564, 120, 44), "Send")
        self.next_button = Button((824, 472, 112, 32), "Next")
        self.back_button = Button((self.next_button.rect.left - 116, 472, 112, 32), "Back")
        self.retry_button = Button((584, 472, 112, 32), "Retry")
        self.portrait_rect = pygame.Rect(24, 472, 148, 148)
        dialogue_x = self.portrait_rect.right + 16
        self.prompt_input = TextInput(
            (dialogue_x, 564, 802 - dialogue_x, 44), placeholder="[insert cool action here]"
        )
        self.dialogue_open = False
        self.selected_npc = None
        self.reply = ""
        self.viewed_conversation = None
        self.conversation_tree = None
        self.conversation_turn_index = 0
        self.dialogue_scroll = 0
        self.reply_rect = pygame.Rect(dialogue_x, 500, 936 - dialogue_x, 58)
        self.camera_offset = [0.0, 0.0]
        self.pan_start = None
        self.pan_last = None
        self.panning = False

    def reset(self):
        self.close_dialogue()
        self.prompt_input.set_value("")
        self.reply = ""
        self.camera_offset[:] = (0.0, 0.0)
        self.pan_start = None
        self.pan_last = None
        self.panning = False

    def close_dialogue(self):
        if self.viewed_conversation is not None:
            self.viewed_conversation.finish(self.conversation_tree)
            self.viewed_conversation = None
            self.conversation_tree = None
        if self.selected_npc is not None:
            self.selected_npc.set_selected(False)
            self.selected_npc = None
        self.dialogue_open = False
        self.conversation_turn_index = 0
        self.dialogue_scroll = 0
        self.prompt_input.active = False
        self.prompt_input.backspace_held = False

    def set_reply(self, text):
        normalized = " ".join(text.split())
        self.reply = "".join(char for char in normalized if char.isprintable())
        self.dialogue_scroll = 0
        if self.selected_npc is not None:
            self.selected_npc.reply = self.reply

    def handle_event(self, app, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if event.pos[1] < WORLD_HEIGHT and not any(
                button.rect.collidepoint(event.pos) for button in (self.menu_button, self.stats_button)
            ):
                self.pan_start = event.pos
                self.pan_last = event.pos
                self.panning = False
            else:
                self.pan_start = None
                self.pan_last = None
                self.panning = False
        elif event.type == pygame.MOUSEMOTION and self.pan_start is not None:
            dx = event.pos[0] - self.pan_start[0]
            dy = event.pos[1] - self.pan_start[1]
            if self.panning or dx * dx + dy * dy >= 16:
                self.panning = True
                self.camera_offset[0] += event.pos[0] - self.pan_last[0]
                self.camera_offset[1] += event.pos[1] - self.pan_last[1]
            self.pan_last = event.pos
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.pan_start is not None:
            if not self.panning and 0 <= event.pos[0] < WORLD_WIDTH and event.pos[1] < WORLD_HEIGHT:
                self.select_at(app, event.pos)
            self.pan_start = None
            self.pan_last = None
            self.panning = False
            return

        if self.stats_button.clicked(event):
            app.open_stats()
            return
        if self.menu_button.clicked(event):
            app.leave_game()
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.dialogue_open:
                self.close_dialogue()
            else:
                app.leave_game()
            return
        if self.dialogue_open:
            if self.handle_dialogue_scroll(app, event):
                return
            if self.viewed_conversation is not None:
                if self.next_button.clicked(event, enabled=bool(self.viewed_conversation.turns)):
                    self.advance_conversation()
                elif self.back_button.clicked(event, enabled=self.conversation_turn_index > 0):
                    self.rewind_conversation()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    self.rewind_conversation()
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE, pygame.K_RIGHT):
                    self.advance_conversation()
                elif self.retry_button.clicked(event, enabled=bool(self.viewed_conversation.error) and not app.requests.busy):
                    app.request_conversation_dialogue(self.viewed_conversation)
                return
            if self.send_button.clicked(event):
                app.submit()
                self.prompt_input.focus()
            elif self.prompt_input.handle_event(event):
                app.submit()

    def advance_conversation(self):
        conversation = self.viewed_conversation
        if conversation is None or not conversation.turns:
            return
        if self.conversation_turn_index == len(conversation.turns) - 1:
            self.close_dialogue()
        else:
            self.conversation_turn_index += 1
            self.dialogue_scroll = 0

    def rewind_conversation(self):
        if self.viewed_conversation is not None and self.conversation_turn_index > 0:
            self.conversation_turn_index -= 1
            self.dialogue_scroll = 0

    def handle_dialogue_scroll(self, app, event):
        font = app.font
        if event.type == pygame.MOUSEWHEEL:
            pos = getattr(event, "pos", pygame.mouse.get_pos())
            if self.viewed_conversation is None and not self.reply_rect.collidepoint(pos):
                return False
            self.dialogue_scroll = max(0, self.dialogue_scroll - event.y * font.get_linesize() * 3)
            return True
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_DOWN, pygame.K_PAGEDOWN):
                self.dialogue_scroll += font.get_linesize() if event.key == pygame.K_DOWN else 80
                return True
            if event.key in (pygame.K_UP, pygame.K_PAGEUP):
                self.dialogue_scroll = max(0, self.dialogue_scroll - (font.get_linesize() if event.key == pygame.K_UP else 80))
                return True
        return False

    def select_at(self, app, screen_pos):
        world_pos = (screen_pos[0] - self.camera_offset[0], screen_pos[1] - self.camera_offset[1])
        npc = next((npc for npc in sorted(app.npcs, key=lambda npc: npc.y, reverse=True) if npc.contains(world_pos)), None)
        if self.viewed_conversation is not None and npc is not None and npc.conversation is self.viewed_conversation:
            # Clicking the other participant keeps the same conversation open.
            if npc is not self.selected_npc:
                self.selected_npc.set_selected(False)
                self.selected_npc = npc
                npc.set_selected(True)
            return
        if npc is None or npc is self.selected_npc:
            self.close_dialogue()
        else:
            self.close_dialogue()
            self.dialogue_open = True
            self.selected_npc = npc
            npc.set_selected(True)
            self.prompt_input.set_value("")
            if npc.conversation is not None:
                self.viewed_conversation = npc.conversation
                self.conversation_tree = app.tree
                self.dialogue_scroll = 0
                return
            self.set_reply(npc.reply or f"Hi, I'm {npc.name}! What should I do today?")
            self.prompt_input.focus()

    def draw(self, app):
        draw_world(app.window, app.npcs, app.tree, app.small_font, app.character.renderer, self.camera_offset)
        app.window.blit(app.small_font.render("Pypeople", True, INK), (24, 26))
        self.menu_button.draw(app.window, app.small_font)
        self.stats_button.draw(app.window, app.small_font)

        pygame.draw.rect(app.window, PAPER, (0, WORLD_HEIGHT, 960, 180))
        pygame.draw.line(app.window, (129, 154, 110), (0, WORLD_HEIGHT), (960, WORLD_HEIGHT), 3)
        if self.viewed_conversation is not None:
            self.draw_conversation(app)
            return
        if self.dialogue_open:
            self.draw_portrait(app, self.selected_npc)
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
            app.window.blit(app.small_font.render("Drag to move the camera. Ask people to wander, stop, shake the tree, or talk to another NPC.", True, MUTED), (24, 602))
            return
        app.window.blit(app.font.render(self.selected_npc.name, True, INK), (self.reply_rect.x, 477))
        self.draw_dialogue_lines(app.window, wrap_text(self.reply, app.font, self.reply_rect.width - 12), app.font, self.reply_rect)
        self.prompt_input.draw(app.window, app.font)
        self.send_button.draw(app.window, app.font, enabled=not app.requests.busy)
        if self.prompt_input.error:
            app.window.blit(app.small_font.render(self.prompt_input.error, True, MUTED), (self.reply_rect.x, 616))
        count = app.small_font.render(f"{len(self.prompt_input.text)}/300", True, MUTED)
        app.window.blit(count, count.get_rect(topright=(936, 616)))

    def draw_portrait(self, app, npc):
        pygame.draw.rect(app.window, (255, 254, 248), self.portrait_rect)
        pygame.draw.rect(app.window, INK, self.portrait_rect, width=2)
        portrait = app.character.renderer.render_portrait(
            npc.character, self.portrait_rect.inflate(-12, -12).size
        )
        app.window.blit(portrait, portrait.get_rect(center=self.portrait_rect.center))

    def draw_conversation(self, app):
        conversation = self.viewed_conversation
        turn = conversation.turns[self.conversation_turn_index] if conversation.turns else None
        if turn is not None:
            speaker = conversation.initiator if turn.speaker == "initiator" else conversation.recipient
            title = speaker.name
        else:
            speaker = self.selected_npc
            title = f"{conversation.initiator.name} & {conversation.recipient.name}"
        self.draw_portrait(app, speaker)
        draw_wrapped_text(app.window, title, app.font, INK, (self.reply_rect.x, 477, self.retry_button.rect.x - self.reply_rect.x - 16, 30))
        self.back_button.draw(app.window, app.small_font, enabled=self.conversation_turn_index > 0)
        last_turn = bool(conversation.turns) and self.conversation_turn_index == len(conversation.turns) - 1
        self.next_button.label = "Done" if last_turn else "Next"
        self.next_button.draw(app.window, app.small_font, enabled=bool(conversation.turns))
        if conversation.error:
            self.retry_button.draw(app.window, app.small_font, enabled=not app.requests.busy)
        rect = pygame.Rect(self.reply_rect.x, 514, self.reply_rect.width, 96)
        if not conversation.arrived:
            text = f"{conversation.initiator.name} is walking over to {conversation.recipient.name}..."
        elif conversation.error and not conversation.turns:
            text = conversation.error
        elif not conversation.turns:
            text = "Generating their conversation..." if self.request_is_current(app) else "Waiting to hear their conversation..."
        else:
            text = turn.text
        self.draw_dialogue_lines(app.window, wrap_text(text, app.font, rect.width - 12), app.font, rect)
        if conversation.turns:
            if conversation.error:
                draw_wrapped_text(app.window, conversation.error, app.small_font, MUTED, (rect.x, 616, rect.width - 80, 24))
            progress = app.small_font.render(f"{self.conversation_turn_index + 1}/{len(conversation.turns)}", True, MUTED)
            app.window.blit(progress, progress.get_rect(topright=(936, 616)))

    def draw_dialogue_lines(self, surface, lines, font, rect):
        line_height = font.get_linesize()
        total_height = len(lines) * line_height
        scroll_limit = max(0, total_height - rect.height)
        self.dialogue_scroll = max(0, min(self.dialogue_scroll, scroll_limit))
        previous_clip = surface.get_clip()
        surface.set_clip(rect.clip(previous_clip))
        for index, line in enumerate(lines):
            y = rect.y + index * line_height - self.dialogue_scroll
            if rect.top - line_height < y < rect.bottom:
                surface.blit(font.render(line, True, INK), (rect.x, y))
        if scroll_limit:
            track = pygame.Rect(rect.right - 5, rect.top, 4, rect.height)
            height = max(12, round(rect.height * rect.height / total_height))
            top = rect.top + round((rect.height - height) * self.dialogue_scroll / scroll_limit)
            pygame.draw.rect(surface, (181, 191, 174), track)
            pygame.draw.rect(surface, MUTED, (track.x, top, track.width, height))
        surface.set_clip(previous_clip)

    def request_is_current(self, app):
        return app.requests.busy and app.request_conversation is self.viewed_conversation
