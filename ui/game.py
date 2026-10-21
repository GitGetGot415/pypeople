import pygame
from llm import ConversationTurn
from constants import (DIALOGUE_PANEL_HEIGHT, FAILED_CONVERSATION_SELECTED_COLOR, INK, MUTED, ORIGINAL_DIALOGUE_PAPER, PAPER,
                       MAX_LLM_REQUESTS, PHASE_PENDING_RECT, WINDOW_SIZE, WORLD_HEIGHT, WORLD_WIDTH)
from emotions import portrait_expression
from conversation_history import display_parts, uses_original_dialogue
from name_knowledge import actor_catalog, name_forms
from constants import (APARTMENT_INSPECTOR_CLOSE_RECT, APARTMENT_INSPECTOR_RECT, APARTMENT_INSPECTOR_ROW_HEIGHT,
                       APARTMENT_INSPECTOR_CONTENT_INSET, APARTMENT_INSPECTOR_CONTENT_TOP,
                       APARTMENT_INSPECTOR_TITLE_OFFSET, APARTMENT_RECT, MODAL_OVERLAY_COLOR)

from .button import Button
from .event_feed import EventFeed
from .request_status import RequestStatus
from .text import draw_wrapped_text, paginate_parts, wrap_text
from .text_input import TextInput
from .world import draw_world, snake_artwork
from .inventory import InventoryPanel
from constants import (INVENTORY_BUTTON_RECT, GAME_PAUSE_RECT, GAME_CLOCK_RECT, GAME_PHASE_POSITION,
                       GAME_PAUSED_LABEL_RIGHT, GIFT_ACCEPT_RECT, GIFT_DECLINE_RECT, CALL_CHOICE_RECT,
                       CALL_CHOICE_TITLE_RECT, CALL_CHOICE_STATUS_RECT, CALL_CHOICE_BUTTON_RECTS)


class GameScreen:
    def __init__(self, max_requests=MAX_LLM_REQUESTS):
        self.menu_button = Button((1144, 16, 112, 38), "Menu")
        self.stats_button = Button((1024, 16, 112, 38), "Stats")
        self.save_button = Button((904, 16, 108, 38), "Save")
        self.inventory_button = Button(INVENTORY_BUTTON_RECT, "Inventory")
        self.inventory = InventoryPanel()
        self.pending_offer = None
        self.accept_offer_button = Button(GIFT_ACCEPT_RECT, "Accept gift")
        self.decline_offer_button = Button(GIFT_DECLINE_RECT, "Decline")
        self.call_choice = None
        self.call_status = ""
        self.call_buttons = {label: Button(rect, label)
                             for label, rect in zip(("Talk", "Not now", "Dismiss"), CALL_CHOICE_BUTTON_RECTS)}
        self.pause_button = Button(GAME_PAUSE_RECT, "Pause")
        self.arrival_retry = Button((476, 72, 120, 32), "Retry arrival")
        self.clock_button = Button(GAME_CLOCK_RECT, "")
        self.save_message = ""
        self.save_message_remaining = 0
        self.send_button = Button((1136, 644, 120, 44), "Send")
        self.done_button = Button((1136, 644, 120, 44), "Done")
        self.next_button = Button((1144, 552, 112, 32), "Next")
        self.back_button = Button((self.next_button.rect.left - 116, 552, 112, 32), "Back")
        self.retry_button = Button((904, 552, 112, 32), "Retry")
        self.subjects_button = Button((self.retry_button.rect.left - 116, 552, 112, 32), "Subjects")
        self.portrait_rect = pygame.Rect(24, 552, 148, 148)
        dialogue_x = self.portrait_rect.right + 16
        self.prompt_input = TextInput(
            (dialogue_x, 644, 1122 - dialogue_x, 44), placeholder="[insert cool action here]"
        )
        self.dialogue_open = False
        self.selected_npc = None
        self.reply = ""
        self._reply_version = None
        self.player_record = None
        self.viewed_conversation = None
        self.conversation_tree = None
        self.conversation_page_index = 0
        self.conversation_pages = []
        self._conversation_page_turns = None
        self.dialogue_scroll = 0
        self.reply_rect = pygame.Rect(dialogue_x, 580, 1256 - dialogue_x, 58)
        self.camera_offset = [0.0, 0.0]
        self.camera_zoom = 1.0
        self.pan_start = None
        self.pan_last = None
        self.panning = False
        self.request_status = RequestStatus(max_requests)
        self.feed = EventFeed()
        self.apartment_open = False
        self.apartment_scroll = 0
        self.apartment_close = Button(APARTMENT_INSPECTOR_CLOSE_RECT, "Close")

    def reset(self):
        self.inventory.close()
        self.call_choice = None
        self.apartment_open = False
        self.apartment_scroll = 0
        self.close_dialogue()
        self.prompt_input.set_value("")
        self.reply = ""
        self.camera_offset[:] = (0.0, 0.0)
        self.camera_zoom = 1.0
        self.pan_start = None
        self.pan_last = None
        self.panning = False
        self.save_message = ""
        self.save_message_remaining = 0
        self.feed = EventFeed()

    def show_save_message(self, message):
        self.save_message = message
        self.save_message_remaining = 8

    def close_dialogue(self):
        self.pending_offer = None
        if self.player_record:
            for offer in self.player_record.player_promise_offers:
                if offer["status"] == "offered":
                    offer["status"] = "closed"
        self.player_record = None
        if self.viewed_conversation is not None:
            self.viewed_conversation = None
            self.conversation_tree = None
        if self.selected_npc is not None:
            self.selected_npc.set_selected(False)
            self.selected_npc = None
        self.dialogue_open = False
        self.conversation_page_index = 0
        self.conversation_pages = []
        self._conversation_page_turns = None
        self.dialogue_scroll = 0
        self.prompt_input.active = False
        self.prompt_input.backspace_held = False

    def set_reply(self, text):
        if self.selected_npc is not None and text != self.selected_npc.reply:
            # Interface messages replacing a reply have no emotional reaction.
            self.selected_npc.set_emotion("neutral", 0)
            self.selected_npc.rewritten_reply = ""
        normalized = " ".join(text.split())
        self.reply = "".join(char for char in normalized if char.isprintable())
        self.dialogue_scroll = 0
        if self.selected_npc is not None:
            self.selected_npc.reply = self.reply
            if self.selected_npc.end_conversation:
                self.prompt_input.set_value("")
                self.prompt_input.active = False

    @property
    def world_viewport(self):
        height = WINDOW_SIZE[1] - DIALOGUE_PANEL_HEIGHT if self.dialogue_open else WINDOW_SIZE[1]
        return pygame.Rect(0, 0, WINDOW_SIZE[0], height)

    @property
    def view_offset(self):
        return (self.camera_offset[0] + (WINDOW_SIZE[0] - WORLD_WIDTH) / 2,
                self.camera_offset[1] + (WINDOW_SIZE[1] - WORLD_HEIGHT) / 2)

    def handle_event(self, app, event):
        if self.call_choice:
            from player import clear_call
            npc = app.simulation.npc_by_id(self.call_choice)
            if npc is None or not npc.player_call:
                self.call_choice = None
                return
            for label, button in self.call_buttons.items():
                if button.clicked(event, enabled=label != "Talk" or not app.simulation.paused):
                    if label == "Talk":
                        self.call_status = app.talk_to_call(npc)
                        if not self.call_status:
                            self.call_choice = None
                    else:
                        if label == "Dismiss":
                            clear_call(npc)
                        self.call_choice = None
            return
        if self.inventory_button.clicked(event):
            self.pan_start = self.pan_last = None
            self.prompt_input.active = self.prompt_input.backspace_held = False
            self.inventory.open(app.simulation.player)
            return
        if self.pending_offer and self.dialogue_open:
            if self.accept_offer_button.clicked(event, enabled=not app.simulation.paused):
                app.accept_gift_offer()
                return
            if self.decline_offer_button.clicked(event, enabled=not app.simulation.paused):
                app.decline_gift_offer()
                return
        if self.apartment_open:
            if self.apartment_close.clicked(event) or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.apartment_open = False
            elif event.type == pygame.MOUSEWHEEL:
                _, content, height = self.apartment_rows(app)
                limit = max(0, height - content.height)
                self.apartment_scroll = max(0, min(limit, self.apartment_scroll - event.y * APARTMENT_INSPECTOR_ROW_HEIGHT))
            return
        paused = app.simulation.paused
        if self.arrival_retry.clicked(event, enabled=not paused):
            app.retry_arrival()
            return
        if self.feed.handle_event(app, event):
            return
        if self.request_status.handle_event(app, event):
            return
        if event.type == pygame.MOUSEWHEEL:
            pos = getattr(event, "pos", pygame.mouse.get_pos())
            if self.world_viewport.collidepoint(pos) and not any(
                button.rect.collidepoint(pos)
                for button in (self.menu_button, self.stats_button, self.save_button, self.inventory_button, self.pause_button, self.clock_button)
            ):
                self.zoom_at(pos, event.y)
                return
        if self.handle_clock_event(app, event):
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.world_viewport.collidepoint(event.pos) and not any(
                button.rect.collidepoint(event.pos) for button in (self.menu_button, self.stats_button, self.save_button, self.inventory_button, self.pause_button)
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
            if not self.panning and self.world_viewport.collidepoint(event.pos):
                self.select_at(app, event.pos)
            self.pan_start = None
            self.pan_last = None
            self.panning = False
            return

        if self.save_button.clicked(event) or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_s and event.mod & pygame.KMOD_CTRL
        ):
            app.request_save()
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
            if self.pending_offer:
                return
            if (self.player_record and self.player_record.error and self.retry_button.clicked(event,
                    enabled=not paused and not self.selected_npc.request_pending)):
                app.retry_player_dialogue()
                return
            if self.handle_dialogue_scroll(app, event):
                return
            if self.viewed_conversation is not None:
                if self.subjects_button.clicked(event, enabled=bool(self.viewed_conversation.record.intended_subjects)):
                    record = self.viewed_conversation.record
                    app.open_stats()
                    app.stats.open_record(record)
                elif self.next_button.clicked(event, enabled=bool(self.conversation_pages)):
                    self.advance_conversation(app)
                elif self.back_button.clicked(event, enabled=self.conversation_page_index > 0):
                    self.rewind_conversation()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    self.rewind_conversation()
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE, pygame.K_RIGHT):
                    self.advance_conversation(app)
                elif self.retry_button.clicked(event, enabled=not paused and (
                        bool(self.viewed_conversation.pending_opinions)
                        or bool(self.viewed_conversation.error and not self.viewed_conversation.turns))):
                    if self.viewed_conversation.pending_opinions:
                        self.viewed_conversation.remember_opinions()
                    else:
                        app.retry_conversation_dialogue(self.viewed_conversation)
                return
            if self.selected_npc.end_conversation:
                if self.done_button.clicked(event) or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE)
                ):
                    self.close_dialogue()
                return
            if paused:
                return
            if self.send_button.clicked(event):
                app.submit()
                self.prompt_input.focus()
            elif self.prompt_input.handle_event(event):
                app.submit()

    def handle_clock_event(self, app, event):
        if self.pause_button.clicked(event):
            app.simulation.paused = not app.simulation.paused
            self.pan_start = self.pan_last = None
            self.panning = False
            return True
        if self.clock_button.clicked(event):
            if not app.simulation.paused and not app.simulation.phase_pending and not app.simulation.player_attention_pending:
                app.simulation.next_phase()
            return True
        return False

    def draw_clock(self, app):
        paused = app.simulation.paused
        self.pause_button.label = "Resume" if paused else "Pause"
        self.pause_button.draw(app.window, app.small_font)
        self.clock_button.draw(app.window, app.small_font, enabled=not paused and not app.simulation.phase_pending
                               and not app.simulation.player_attention_pending)
        x, y = self.clock_button.rect.center
        pygame.draw.circle(app.window, PAPER, (x, y), 12, width=2)
        pygame.draw.line(app.window, PAPER, (x, y), (x, y - 8), 2)
        pygame.draw.line(app.window, PAPER, (x, y), (x + 6, y + 3), 2)
        color = PAPER if app.screen == "game" and app.simulation.phase == "night" else INK
        app.window.blit(app.small_font.render(app.simulation.phase_label, True, color), GAME_PHASE_POSITION)
        if app.simulation.player_attention_pending:
            draw_wrapped_text(app.window, "Address the waiting NPCs before changing the time.",
                              app.small_font, color, PHASE_PENDING_RECT)
        elif app.simulation.phase_pending:
            draw_wrapped_text(app.window, f"{app.simulation.next_phase_label} pending: finishing conversations.",
                              app.small_font, color, PHASE_PENDING_RECT)
        if paused:
            label = app.small_font.render("Paused", True, color)
            app.window.blit(label, label.get_rect(midright=(GAME_PAUSED_LABEL_RIGHT, self.pause_button.rect.centery)))

    def advance_conversation(self, app):
        conversation = self.viewed_conversation
        if conversation is None or not self.conversation_pages:
            return
        if self.conversation_page_index == len(self.conversation_pages) - 1:
            if conversation.active:
                if app.simulation.paused:
                    return
                conversation.talking_remaining = 0.0
                conversation.update(0.0)
            self.close_dialogue()
        else:
            self.conversation_page_index += 1
            self.dialogue_scroll = 0

    def rewind_conversation(self):
        if self.viewed_conversation is not None and self.conversation_page_index > 0:
            self.conversation_page_index -= 1
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

    def screen_to_world(self, screen_pos):
        offset = self.view_offset
        return ((screen_pos[0] - offset[0]) / self.camera_zoom,
                (screen_pos[1] - offset[1]) / self.camera_zoom)

    def zoom_at(self, screen_pos, steps):
        world_pos = self.screen_to_world(screen_pos)
        old_zoom = self.camera_zoom
        self.camera_zoom = max(0.5, min(2.5, self.camera_zoom * 1.1 ** steps))
        self.camera_offset[0] += world_pos[0] * (old_zoom - self.camera_zoom)
        self.camera_offset[1] += world_pos[1] * (old_zoom - self.camera_zoom)

    def select_at(self, app, screen_pos):
        from .world import player_bubble_rect, player_bubble_visible
        npc = None
        for resident in app.simulation.outdoor_npcs:
            if player_bubble_visible(resident, self.view_offset, self.camera_zoom) and player_bubble_rect(resident, self.view_offset, self.camera_zoom).clip(self.world_viewport).collidepoint(screen_pos):
                if resident.player_call:
                    self.open_call(app, resident)
                    return
                npc = resident
                break
        world_pos = self.screen_to_world(screen_pos)
        if npc is None and pygame.Rect(APARTMENT_RECT).collidepoint(world_pos):
            self.close_dialogue()
            self.pan_start = self.pan_last = None
            self.panning = False
            self.apartment_open = True
            self.apartment_scroll = 0
            return
        npc = npc or next((npc for npc in sorted([*app.npcs, app.simulation.snake], key=lambda npc: npc.y, reverse=True) if npc.contains(world_pos)), None)
        if npc is not None and npc is not app.simulation.snake and npc.player_call:
            self.open_call(app, npc)
            return
        if self.viewed_conversation is not None and npc is not None and (npc.conversation or getattr(npc, "snake_encounter", None)) is self.viewed_conversation:
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
            interaction = npc.conversation or getattr(npc, "snake_encounter", None)
            if interaction is not None:
                self.viewed_conversation = interaction
                self.conversation_tree = app.tree
                self.dialogue_scroll = 0
                return
            self.set_reply(npc.reply)
            if npc is not app.simulation.snake and npc.player_thinking and app.request_npc is npc and app.request_history is not None:
                self.player_record = app.request_history
            self.prompt_input.focus()

    def draw(self, app):
        draw_world(app.window, app.npcs, app.tree, app.small_font, app.character.renderer,
                   self.view_offset, phase=app.simulation.phase, zoom=self.camera_zoom,
                   viewport=self.world_viewport, snake=app.simulation.snake)
        self.menu_button.draw(app.window, app.small_font)
        self.stats_button.draw(app.window, app.small_font)
        self.save_button.draw(app.window, app.small_font)
        self.inventory_button.draw(app.window, app.small_font)
        self.draw_clock(app)
        self.feed.draw(app)
        self.request_status.draw(app)
        pending = app.simulation.pending_arrival_days
        error = app.simulation.arrival_error
        if pending or error:
            color = PAPER if app.simulation.phase == "night" else INK
            label = f"Arrivals waiting: {', '.join(str(d) for d in pending)}" if pending else "Next arrival"
            if error:
                label += f" / {error}"
                self.arrival_retry.draw(app.window, app.small_font, enabled=not app.simulation.paused)
            else:
                label += " / arriving"
            draw_wrapped_text(app.window, label, app.small_font, color, (476, 110, 550, 64))
        if self.save_message_remaining:
            color = PAPER if app.simulation.phase == "night" else INK
            draw_wrapped_text(app.window, self.save_message, app.small_font, color, (24, 480, 1232, 44))

        if not self.dialogue_open:
            if self.apartment_open:
                self.draw_apartments(app)
            return
        if self.viewed_conversation is not None:
            self.draw_conversation(app)
            return
        rewritten = self.selected_npc.rewritten_reply
        names = name_forms(actor_catalog(app.simulation))
        original = uses_original_dialogue([(self.reply, None)], [rewritten] if rewritten else (),
                                          original=app.original_dialogue, speaker=self.selected_npc.name, listener="You", names=names)
        self.draw_dialogue_background(app, original)
        if self.dialogue_open:
            self.draw_portrait(app, self.selected_npc, self.selected_npc.portrait_emotion)
        status = self.selected_npc.status if self.selected_npc is not None else f"{len(app.npcs)} NPCs"
        if app.request_history is not None and app.request_npc is self.selected_npc:
            state = "Queued" if (app.session_id, app.request_id) in app.requests.pending else "Thinking"
            status = f"{state} for {self.selected_npc.name}... / {status}"
        elif self.selected_npc.end_conversation:
            status = "Click Done when you've finished reading."
        elif self.selected_npc.request_pending:
            status = f"Applying player request / {status}"
        if self.pending_offer:
            offer = self.pending_offer
            work = next((w for w in app.simulation.creations if w.id == offer["item"]), None)
            status = f'Gift offer: {offer["quantity"]} × {work.title + " [" + work.id[:8] + "]" if work else "apples"}'
        elif self.player_record and self.player_record.error:
            self.retry_button.draw(app.window, app.small_font, enabled=not app.simulation.paused and not self.selected_npc.request_pending)
            status = "Reply failed."
        rendered_status = app.small_font.render(status, True, MUTED)
        app.window.blit(rendered_status, rendered_status.get_rect(topright=(1256, 558)))
        app.window.blit(app.font.render(self.selected_npc.name, True, INK), (self.reply_rect.x, 557))
        reply = display_parts([(self.reply, None)], [rewritten] if rewritten else (), original=app.original_dialogue,
                              speaker=self.selected_npc.name, listener="You", names=names)[0][0]
        version = (reply, app.original_dialogue)
        if self._reply_version != version:
            self.dialogue_scroll = 0
            self._reply_version = version
        self.draw_dialogue_lines(app.window, wrap_text(reply or "Enter a message to talk to this NPC.", app.font,
                                                     self.reply_rect.width - 12), app.font, self.reply_rect,
                                 color=INK if self.reply else MUTED)
        closing = self.selected_npc.end_conversation
        if self.pending_offer:
            self.accept_offer_button.draw(app.window, app.small_font, enabled=not app.simulation.paused)
            self.decline_offer_button.draw(app.window, app.small_font, enabled=not app.simulation.paused)
        elif closing:
            self.done_button.draw(app.window, app.font)
        else:
            self.prompt_input.draw(app.window, app.font)
            self.send_button.draw(app.window, app.font, enabled=app.request_history is None
                                  and not self.selected_npc.request_pending and not app.simulation.paused)
        if self.player_record and self.player_record.dialogue_errors:
            draw_wrapped_text(app.window, self.player_record.dialogue_errors[-1], app.small_font,
                              FAILED_CONVERSATION_SELECTED_COLOR, (self.reply_rect.x, 696, self.reply_rect.width - 90, 24))
        elif not closing and self.prompt_input.error:
            app.window.blit(app.small_font.render(self.prompt_input.error, True, MUTED), (self.reply_rect.x, 696))
        else:
            reason = self.selected_npc.decision_reason
            while reason and app.small_font.size(reason)[0] > self.reply_rect.width - 90:
                reason = reason[:-4] + "..."
            app.window.blit(app.small_font.render(reason, True, MUTED), (self.reply_rect.x, 696))
        if not closing:
            count = app.small_font.render(f"{len(self.prompt_input.text)}/300", True, MUTED)
            app.window.blit(count, count.get_rect(topright=(1256, 696)))

    def open_call(self, app, npc):
        if npc.unread_player_reply:
            from player import call_valid, clear_call, player_reply_closes
            call = npc.unread_player_reply
            if not call_valid(app.simulation, npc, call):
                clear_call(npc)
                return
            record = next(r for r in app.conversation_history if r.id == call["reference"])
            self.close_dialogue()
            self.call_choice = None
            self.dialogue_open, self.selected_npc, self.player_record = True, npc, record
            npc.set_selected(True)
            npc.end_conversation = call["closing"]
            if not npc.request_pending:
                npc.end_conversation = player_reply_closes(npc.end_conversation, npc.reply,
                                                          (npc.rewritten_reply,) if npc.rewritten_reply else ())
            self.pending_offer = ({**call["gift_offer"], "npc_id": npc.id,
                "revision": npc.command_revision, "record_id": record.id} if call["gift_offer"] else None)
            clear_call(npc)
            record.status = "Waiting for Done" if npc.end_conversation and npc.request_pending else "Completed"
            self.prompt_input.set_value("")
            self.set_reply(npc.reply)
            if not npc.end_conversation:
                self.prompt_input.focus()
            self.pan_start = self.pan_last = None
            self.panning = False
            return
        if self.dialogue_open or app.request_history is not None:
            self.show_save_message("Finish the current exchange before opening another call.")
            return
        if npc.player_call:
            self.call_choice = npc.id
            self.call_status = ""
            self.pan_start = self.pan_last = None

    def draw_call_choice(self, app):
        if not self.call_choice:
            return
        npc = app.simulation.npc_by_id(self.call_choice)
        if not npc or not npc.player_call:
            self.call_choice = None
            return
        overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
        overlay.fill(MODAL_OVERLAY_COLOR)
        app.window.blit(overlay, (0, 0))
        pygame.draw.rect(app.window, PAPER, CALL_CHOICE_RECT)
        draw_wrapped_text(app.window, f"{npc.name} would like to talk to you.", app.heading_font, INK, CALL_CHOICE_TITLE_RECT)
        draw_wrapped_text(app.window, self.call_status or "Talk or dismiss this invitation to continue. Not now keeps the resident waiting and time changes blocked.",
                          app.font, MUTED, CALL_CHOICE_STATUS_RECT)
        for label, button in self.call_buttons.items():
            button.draw(app.window, app.small_font, enabled=label != "Talk" or not app.simulation.paused)

    def apartment_rows(self, app):
        rect = pygame.Rect(APARTMENT_INSPECTOR_RECT)
        content = rect.inflate(*(-value for value in APARTMENT_INSPECTOR_CONTENT_INSET))
        content.top = rect.top + APARTMENT_INSPECTOR_CONTENT_TOP
        rows, top = [], 0
        for npc in sorted((npc for npc in app.npcs if npc.indoors), key=lambda npc: npc.room_number):
            for paragraph in (f"Room {npc.room_number}: {npc.name}", npc.status):
                for line in wrap_text(paragraph, app.small_font, content.width):
                    rows.append((line, top))
                    top += app.small_font.get_linesize()
            top += APARTMENT_INSPECTOR_ROW_HEIGHT - app.small_font.get_linesize()
        return rows, content, top

    def draw_apartments(self, app):
        overlay = pygame.Surface(app.window.get_size(), pygame.SRCALPHA)
        overlay.fill(MODAL_OVERLAY_COLOR)
        app.window.blit(overlay, (0, 0))
        rect = pygame.Rect(APARTMENT_INSPECTOR_RECT)
        pygame.draw.rect(app.window, PAPER, rect)
        app.window.blit(app.heading_font.render("Apartments", True, INK),
                        (rect.x + APARTMENT_INSPECTOR_TITLE_OFFSET[0], rect.y + APARTMENT_INSPECTOR_TITLE_OFFSET[1]))
        self.apartment_close.draw(app.window, app.small_font)
        rows, content, height = self.apartment_rows(app)
        limit = max(0, height - content.height)
        self.apartment_scroll = min(self.apartment_scroll, limit)
        previous_clip = app.window.get_clip()
        app.window.set_clip(content.clip(previous_clip))
        if not rows:
            app.window.blit(app.font.render("Everyone is outside.", True, MUTED), content.topleft)
        for label, top in rows:
            app.window.blit(app.small_font.render(label, True, INK),
                            (content.x, content.y + top - self.apartment_scroll))
        app.window.set_clip(previous_clip)

    def draw_dialogue_background(self, app, original):
        top = WINDOW_SIZE[1] - DIALOGUE_PANEL_HEIGHT
        color = ORIGINAL_DIALOGUE_PAPER if original else PAPER
        pygame.draw.rect(app.window, color, (0, top, WINDOW_SIZE[0], DIALOGUE_PANEL_HEIGHT))
        pygame.draw.line(app.window, (129, 154, 110), (0, top), (WINDOW_SIZE[0], top), 3)

    def draw_portrait(self, app, npc, emotion=None):
        pygame.draw.rect(app.window, (255, 254, 248), self.portrait_rect)
        pygame.draw.rect(app.window, INK, self.portrait_rect, width=2)
        if npc is app.simulation.snake:
            art = snake_artwork(4, 1)
            ratio = min(136 / art.get_width(), 136 / art.get_height())
            portrait = pygame.transform.scale(art, (round(art.get_width() * ratio), round(art.get_height() * ratio)))
        else:
            portrait = app.character.renderer.render_portrait(
                npc.character, self.portrait_rect.inflate(-12, -12).size, emotion=emotion
            )
        app.window.blit(portrait, portrait.get_rect(center=self.portrait_rect.center))

    def draw_conversation(self, app):
        conversation = self.viewed_conversation
        rect = pygame.Rect(self.reply_rect.x, 594, self.reply_rect.width, 96)
        source = conversation.turns or conversation.record.failed_turns
        failed = bool(source) and not conversation.turns
        version = (id(source), app.original_dialogue, getattr(app, "condensed_dialogue", False))
        if self._conversation_page_turns != version:
            anchor = None
            if (self.conversation_pages and self._conversation_page_turns is not None
                    and self._conversation_page_turns[0] == id(source)):
                old_index = max(0, min(self.conversation_page_index, len(self.conversation_pages) - 1))
                previous_turn = self.conversation_pages[old_index][0]
                turn_ids = list(dict.fromkeys(id(turn) for turn, _, _ in self.conversation_pages))
                anchor = turn_ids.index(id(previous_turn))
            turns = conversation.turns or [ConversationTurn.model_validate(turn) for turn in source]
            self.conversation_pages = [
                (turn, part, text)
                for index, turn in enumerate(turns)
                for part, text in paginate_parts(display_parts([(part.text, part) for part in turn.parts],
                                                 () if failed else conversation.record.rewritten_parts.get(index, ()),
                                                 original=app.original_dialogue,
                                                 names=conversation.record.name_forms,
                                                 introductions=conversation.record.name_introductions.get(index, ()),
                                                 speaker=(conversation.initiator if turn.speaker == "initiator" else conversation.recipient).name,
                                                 listener=(conversation.recipient if turn.speaker == "initiator" else conversation.initiator).name),
                                                 app.font, rect.width - 12, condensed=version[2])
            ]
            self._conversation_page_turns = version
            self.conversation_page_index = min(self.conversation_page_index, max(0, len(self.conversation_pages) - 1))
            if anchor is not None and anchor < len(turns):
                self.conversation_page_index = next((index for index, page in enumerate(self.conversation_pages)
                                                     if page[0] is turns[anchor]), self.conversation_page_index)
            self.dialogue_scroll = 0
        page = self.conversation_pages[self.conversation_page_index] if self.conversation_pages else None
        turn, part, text = page if page is not None else (None, None, "")
        original = app.original_dialogue or failed
        if turn is not None and not failed:
            index = next(index for index, source_turn in enumerate(conversation.turns) if source_turn is turn)
            original = uses_original_dialogue(turn.parts, conversation.record.rewritten_parts.get(index, ()),
                                              original=app.original_dialogue,
                                              names=conversation.record.name_forms,
                                              introductions=conversation.record.name_introductions.get(index, ()),
                                              speaker=(conversation.initiator if turn.speaker == "initiator" else conversation.recipient).name,
                                              listener=(conversation.recipient if turn.speaker == "initiator" else conversation.initiator).name)
        self.draw_dialogue_background(app, original)
        if turn is not None:
            speaker = conversation.initiator if turn.speaker == "initiator" else conversation.recipient
            title = speaker.name + (" - failed draft" if failed else "")
        else:
            speaker = self.selected_npc
            title = f"{conversation.initiator.name} & {conversation.recipient.name}"
        emotion = portrait_expression(part.emotion.name, part.emotion.intensity) if part is not None else None
        self.draw_portrait(app, speaker, emotion)
        title_right = self.subjects_button.rect.left if conversation.record.intended_subjects else self.retry_button.rect.left
        draw_wrapped_text(app.window, title, app.font, INK, (self.reply_rect.x, 557, title_right - self.reply_rect.x - 16, 30))
        if conversation.record.intended_subjects:
            self.subjects_button.draw(app.window, app.small_font)
        self.back_button.draw(app.window, app.small_font, enabled=self.conversation_page_index > 0)
        last_page = bool(self.conversation_pages) and self.conversation_page_index == len(self.conversation_pages) - 1
        self.next_button.label = "Done" if last_page else "Next"
        can_finish = not (last_page and conversation.active and app.simulation.paused)
        self.next_button.draw(app.window, app.small_font, enabled=bool(self.conversation_pages) and can_finish)
        if conversation.pending_opinions or (conversation.error and not conversation.turns):
            self.retry_button.draw(app.window, app.small_font, enabled=not app.simulation.paused)
        if not conversation.arrived:
            if conversation.initiator is app.simulation.snake:
                text = f"{conversation.recipient.name} is walking to the guarded tree..."
            else:
                text = f"{conversation.initiator.name} is walking over to {conversation.recipient.name}..."
        elif conversation.error and not source:
            text = conversation.error
        elif not source:
            text = "Generating their conversation..." if self.request_is_current(app) else "Waiting to hear their conversation..."
        self.draw_dialogue_lines(app.window, wrap_text(text, app.font, rect.width - 12), app.font, rect)
        if source:
            status = ("Simulation paused. Resume to continue." if app.simulation.paused and conversation.active
                      else "Conversation paused while you read." if conversation.active
                      else f"{conversation.record.status} - recorded dialogue.")
            status = "Generated draft; no effects applied. " + conversation.error if failed else conversation.error or status
            if conversation.record.dialogue_errors:
                status = "Validation off: " + conversation.record.dialogue_errors[0]
            status_color = FAILED_CONVERSATION_SELECTED_COLOR if conversation.error or conversation.record.dialogue_errors else MUTED
            draw_wrapped_text(app.window, status, app.small_font, status_color,
                              (rect.x, 696, rect.width - 80, 24))
            progress = app.small_font.render(f"{self.conversation_page_index + 1}/{len(self.conversation_pages)}", True, MUTED)
            app.window.blit(progress, progress.get_rect(topright=(1256, 696)))

    def draw_dialogue_lines(self, surface, lines, font, rect, *, color=INK):
        line_height = font.get_linesize()
        total_height = len(lines) * line_height
        scroll_limit = max(0, total_height - rect.height)
        self.dialogue_scroll = max(0, min(self.dialogue_scroll, scroll_limit))
        previous_clip = surface.get_clip()
        surface.set_clip(rect.clip(previous_clip))
        for index, line in enumerate(lines):
            y = rect.y + index * line_height - self.dialogue_scroll
            if rect.top - line_height < y < rect.bottom:
                surface.blit(font.render(line, True, color), (rect.x, y))
        if scroll_limit:
            track = pygame.Rect(rect.right - 5, rect.top, 4, rect.height)
            height = max(12, round(rect.height * rect.height / total_height))
            top = rect.top + round((rect.height - height) * self.dialogue_scroll / scroll_limit)
            pygame.draw.rect(surface, (181, 191, 174), track)
            pygame.draw.rect(surface, MUTED, (track.x, top, track.width, height))
        surface.set_clip(previous_clip)

    def request_is_current(self, app):
        return (self.viewed_conversation is not None
                and app.conversation_requests.get(self.viewed_conversation.id) is self.viewed_conversation)
