import pygame
from player import PlayerInteraction
from constants import (DIALOGUE_PANEL_HEIGHT, DIALOGUE_READING_Y, DIALOGUE_READING_HEIGHT, FAILED_CONVERSATION_SELECTED_COLOR, INK, MUTED,
                      ORIGINAL_DIALOGUE_PAPER, PAPER, MAX_LLM_REQUESTS, PHASE_PENDING_RECT, WINDOW_SIZE,
                      WORLD_HEIGHT, WORLD_WIDTH, APARTMENT_INSPECTOR_CLOSE_RECT, APARTMENT_INSPECTOR_RECT,
                      APARTMENT_INSPECTOR_ROW_HEIGHT, APARTMENT_INSPECTOR_CONTENT_INSET,
                      APARTMENT_INSPECTOR_CONTENT_TOP, APARTMENT_INSPECTOR_TITLE_OFFSET, MODAL_OVERLAY_COLOR,
                      INVENTORY_BUTTON_RECT, GAME_PAUSE_RECT, GAME_CLOCK_RECT, GAME_PHASE_POSITION,
                      GAME_PAUSED_LABEL_RIGHT, GIFT_ACCEPT_RECT, GIFT_DECLINE_RECT,
                      DIALOGUE_ORIGINAL_TOGGLE_RECT, BORDER_RADIUS, GAME_MENU_CONFIRM_RECT,
                      GAME_MENU_CONFIRM_TITLE_Y, GAME_MENU_CONFIRM_MESSAGE_RECT, GAME_MENU_KEEP_PLAYING_RECT,
                      GAME_MENU_LEAVE_RECT, PLAYER_ROOM_LOCATION, ROOM_BED_RECT,
                      BED_POPUP_RECT, BED_POPUP_TITLE_Y, BED_POPUP_MESSAGE_RECT, BED_ADVANCE_RECT,
                      BED_CANCEL_RECT, PLAYER_KILL_BUTTON_RECT)
from emotions import portrait_expression
from conversation_history import uses_original_dialogue
from .dialogue_view import DialogueView
from .button import Button
from .event_feed import EventFeed
from .request_status import RequestStatus
from .text import draw_centered_text, draw_wrapped_text, wrap_text, scrollbar_rects, ScrollbarDrag
from .text_input import TextInput
from .world import CHARACTER_SCALE, draw_world
from .character_sprite import draw_actor_portrait
from constants import GAME_PORTRAIT_BACKGROUND, GAME_PORTRAIT_PADDING, GAME_PORTRAIT_BORDER_WIDTH
from .inventory import InventoryPanel
from constants import (PLAYER_ACTION_BUTTON_RECT, PLAYER_ACTION_PICKER_RECT, PLAYER_ACTION_PICKER_ROWS,
                       PLAYER_ACTION_PICKER_ROW_HEIGHT, PLAYER_ACTION_PICKER_INSET,
                       PLAYER_ACTION_PICKER_TITLE_HEIGHT, PLAYER_ACTION_PICKER_FOOTER_HEIGHT,
                       PLAYER_ACTION_PICKER_BUTTON_WIDTH, PLAYER_ACTION_PICKER_ROW_BUTTON_HEIGHT,
                       PLAYER_ACTION_PICKER_FOOTER_BUTTON_HEIGHT, PLAYER_ACTION_PICKER_BORDER_WIDTH)


class GameScreen:
    def __init__(self, max_requests=MAX_LLM_REQUESTS):
        from constants import END_CONVERSATION_RECT, HARVEST_BUTTON_RECT, HARVEST_AMOUNT_RECT, INVITATION_ACCEPT_RECT
        self.app = None
        self.location = "garden"
        self.location_menu = None
        self.end_button = Button(END_CONVERSATION_RECT, "End conversation")
        self.kill_button = Button(PLAYER_KILL_BUTTON_RECT, "Kill")
        self.harvest_button = Button(HARVEST_BUTTON_RECT, "Harvest apples")
        self.harvest_amount = TextInput(HARVEST_AMOUNT_RECT, max_length=6, placeholder="1")
        self.harvest_amount.set_value("1")
        self.invitation_button = Button(INVITATION_ACCEPT_RECT, "Accept invitation")
        self.actions_button = Button(PLAYER_ACTION_BUTTON_RECT, "Request action")
        self.menu_button = Button((1144, 16, 112, 38), "Menu")
        self.confirm_menu = False
        self.menu_restore_focus = False
        self.keep_playing_button = Button(GAME_MENU_KEEP_PLAYING_RECT, "Keep playing")
        self.leave_game_button = Button(GAME_MENU_LEAVE_RECT, "Return to menu")
        self.stats_button = Button((1024, 16, 112, 38), "Stats")
        self.save_button = Button((904, 16, 108, 38), "Save")
        self.inventory_button = Button(INVENTORY_BUTTON_RECT, "Inventory")
        self.inventory = InventoryPanel()
        self.accept_offer_button = Button(GIFT_ACCEPT_RECT, "Accept gift")
        self.decline_offer_button = Button(GIFT_DECLINE_RECT, "Decline")
        self.pause_button = Button(GAME_PAUSE_RECT, "Pause")
        self.arrival_retry = Button((476, 72, 120, 32), "Retry arrival")
        self.clock_button = Button(GAME_CLOCK_RECT, "")
        self.bed_open = False
        self.bed_advance_button = Button(BED_ADVANCE_RECT, "")
        self.bed_cancel_button = Button(BED_CANCEL_RECT, "Cancel")
        self.save_message = ""
        self.save_message_remaining = 0
        self.send_button = Button((1136, 644, 120, 44), "Send")
        self.done_button = Button((1136, 644, 120, 44), "Done")
        self.next_button = Button((1144, 552, 112, 32), "Next")
        self.back_button = Button((self.next_button.rect.left - 116, 552, 112, 32), "Back")
        self.retry_button = Button((904, 552, 112, 32), "Retry")
        self.subjects_button = Button((self.retry_button.rect.left - 116, 552, 112, 32), "Subjects")
        self.portrait_rect = pygame.Rect(24, 552, 148, 148)
        self.original_toggle = Button(DIALOGUE_ORIGINAL_TOGGLE_RECT, "Original: Off")
        dialogue_x = self.portrait_rect.right + 16
        self.prompt_input = TextInput(
            (dialogue_x, 644, 1122 - dialogue_x, 44), placeholder="[insert cool action here]"
        )
        self.dialogue = DialogueView()
        self.reply_rect = pygame.Rect(dialogue_x, 580, 1256 - dialogue_x, 58)
        self.dialogue_scroll_rect = self.reply_rect.copy()
        self.dialogue.scroll_limit = 0
        self.dialogue_scrollbar = ScrollbarDrag()
        self.camera_offset = [0.0, 0.0]
        self.camera_zoom = 1.0
        self.pan_start = None
        self.pan_last = None
        self.panning = False
        self.request_status = RequestStatus(max_requests)
        self.feed = EventFeed()
        self.apartment_open = False
        self.apartment_scroll = 0
        self.apartment_scrollbar = ScrollbarDrag()
        self.apartment_close = Button(APARTMENT_INSPECTOR_CLOSE_RECT, "Close")

    @property
    def pending_offer(self):
        return self.app.conversations.pending_offer if self.app and self.app.conversations else None

    @property
    def player_record(self):
        if self.dialogue.conversation is not None:
            return None
        record = self.app.conversations.record if self.app and self.app.conversations else None
        return record or self.dialogue.record

    def reset(self):
        self.apartment_scrollbar.cancel()
        self.bed_open = False
        self.confirm_menu = False
        self.menu_restore_focus = False
        self.inventory.close()
        self.apartment_open = False
        self.location_menu = None
        self.apartment_scroll = 0
        self.close_dialogue()
        self.prompt_input.set_value("")
        self.camera_offset[:] = (0.0, 0.0)
        self.camera_zoom = 1.0
        self.pan_start = None
        self.pan_last = None
        self.panning = False
        self.save_message = ""
        self.save_message_remaining = 0
        self.feed = EventFeed()
        self.request_status = RequestStatus(self.request_status.max_requests)

    def show_save_message(self, message):
        self.save_message = message
        self.save_message_remaining = 8

    def close_dialogue(self, *, terminate=True, notify=True):
        self.dialogue_scrollbar.cancel()
        self.dialogue.scroll_limit = 0
        if notify and self.app and self.app.conversations:
            self.app._sync_conversation_view()
            self.app.conversations.view_closed(terminate=terminate)
            self.app._present_conversation_events()
        self.dialogue.reset()
        self.prompt_input.active = False
        self.prompt_input.backspace_held = False

    def set_reply(self, text):
        normalized = " ".join(text.split())
        self.dialogue.status = "".join(char for char in normalized if char.isprintable())
        self.dialogue.scroll = 0
        self.prompt_input.active = self.prompt_input.backspace_held = False
        if self.dialogue.actor is not None:
            if self.dialogue.actor.player_reply.closing:
                self.prompt_input.set_value("")
                self.prompt_input.active = False

    @property
    def world_viewport(self):
        height = WINDOW_SIZE[1] - DIALOGUE_PANEL_HEIGHT if self.dialogue.open else WINDOW_SIZE[1]
        return pygame.Rect(0, 0, WINDOW_SIZE[0], height)

    @property
    def view_offset(self):
        return (self.camera_offset[0] + (WINDOW_SIZE[0] - WORLD_WIDTH) / 2,
                self.camera_offset[1] + (WINDOW_SIZE[1] - WORLD_HEIGHT) / 2)

    def handle_event(self, app, event):
        from player import active_player
        world = app.simulation
        if self.dialogue.open and self.dialogue.conversation is None and self.dialogue.actor is not None:
            self.update_player_pages(app)
        if self.bed_open:
            self.handle_bed_event(app, event)
            return
        if self.confirm_menu:
            if self.keep_playing_button.clicked(event) or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.confirm_menu = False
                self.prompt_input.active = self.menu_restore_focus
            elif self.leave_game_button.clicked(event):
                self.confirm_menu = False
                app.leave_game()
            return
        if self.location_menu:
            self.handle_location_event(app, event)
            return
        if self.dialogue.picker is not None:
            self.handle_action_picker(app, event)
            return
        if self.dialogue.open and active_player(world) and self.kill_button.clicked(event):
            app._sync_conversation_view()
            app.conversations.kill_resident(self.dialogue.actor)
            app._present_conversation_events()
            return
        if self.dialogue.open and self.dialogue.conversation is None and self.actions_button.clicked(event):
            if self.player_response_ready and not self.pending_offer:
                reason = app.conversations.input_unavailable(self.dialogue.actor, active=app.screen == "game")
                if not reason and self.dialogue.actor is not world.snake:
                    self.dialogue.picker = dict(command={}, page=0, focus=self.prompt_input.active)
                    self.prompt_input.active = self.prompt_input.backspace_held = False
            return
        if self.can_harvest(world):
            self.harvest_amount.handle_event(event)
            if self.harvest_button.clicked(event):
                try:
                    world.harvest(world.player, int(self.harvest_amount.text or "1"))
                except ValueError as exc:
                    self.show_save_message(str(exc))
                return
            if self.harvest_amount.active:
                return
        else:
            self.harvest_amount.active = False
        if self.dialogue.open and world.player_interaction and self.end_button.clicked(event):
            self.close_dialogue()
            return
        if self.dialogue.open and self.player_response_ready and self.invitation_button.clicked(event):
            offer = self.pending_invitation(world)
            if offer:
                app._sync_conversation_view()
                app.conversations.accept_player_invitation(offer.id)
                app._present_conversation_events()
                return
        if active_player(world) and self.inventory_button.clicked(event):
            self.pan_start = self.pan_last = None
            self.prompt_input.active = self.prompt_input.backspace_held = False
            self.inventory.open(app.simulation.player)
            return
        if self.pending_offer and self.dialogue.open and self.player_response_ready:
            if self.accept_offer_button.clicked(event, enabled=not app.simulation.paused):
                app._sync_conversation_view()
                app.conversations.accept_gift_offer()
                app._present_conversation_events()
                return
            if self.decline_offer_button.clicked(event, enabled=not app.simulation.paused):
                app._sync_conversation_view()
                app.conversations.decline_gift_offer()
                app._present_conversation_events()
                return
        if self.apartment_open:
            _, content, height = self.apartment_rows(app)
            limit = max(0, height - content.height)
            value = self.apartment_scrollbar.handle_event(event, content, self.apartment_scroll, limit)
            if value is not None:
                self.apartment_scroll = value
                return
            if self.apartment_close.clicked(event) or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.apartment_open = False
                self.apartment_scrollbar.cancel()
            elif event.type == pygame.MOUSEWHEEL:
                self.apartment_scroll = max(0, min(limit, self.apartment_scroll - event.y * APARTMENT_INSPECTOR_ROW_HEIGHT))
            return
        if self.dialogue.open and self.dialogue_scrollbar.active and self.handle_dialogue_scroll(app, event):
            return
        if self.request_status.scrollbar.active and self.request_status.handle_event(app, event):
            return
        paused = app.simulation.paused
        if self.arrival_retry.clicked(event, enabled=not paused):
            app.retry_arrival()
            return
        if self.feed.handle_event(app, event):
            return
        if self.request_status.handle_event(app, event):
            return
        world_buttons = (self.menu_button, self.stats_button, self.save_button, self.pause_button, self.clock_button)
        if active_player(world):
            world_buttons += (self.inventory_button,)
        if event.type == pygame.MOUSEWHEEL:
            pos = getattr(event, "pos", pygame.mouse.get_pos())
            if self.world_viewport.collidepoint(pos) and not any(
                button.rect.collidepoint(pos)
                for button in world_buttons
            ):
                self.zoom_at(pos, event.y)
                return
        if self.handle_clock_event(app, event):
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.world_viewport.collidepoint(event.pos) and not any(
                button.rect.collidepoint(event.pos) for button in world_buttons
            ):
                self.pan_start = event.pos
                self.pan_last = event.pos
                self.panning = False
            else:
                self.pan_start = None
                self.pan_last = None
                self.panning = False
        elif event.type == pygame.MOUSEMOTION and self.pan_start is not None and not active_player(world):
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
            self.open_menu_confirmation()
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.dialogue.open:
                self.close_dialogue()
            else:
                self.open_menu_confirmation()
            return
        if self.dialogue.open:
            if self.original_toggle.clicked(event):
                app.original_dialogue = not app.original_dialogue
                self.dialogue.scroll = 0
                return
            if (self.player_record and self.player_record.error and self.retry_button.clicked(event,
                    enabled=not paused and not self.dialogue.actor.interaction_blocked)):
                app._sync_conversation_view()
                app.conversations.retry_player_dialogue()
                app._present_conversation_events()
                return
            if self.handle_dialogue_scroll(app, event):
                return
            if self.dialogue.conversation is not None:
                if self.subjects_button.clicked(event, enabled=bool(self.dialogue.conversation.record.intended_subjects)):
                    record = self.dialogue.conversation.record
                    app.open_stats()
                    app.stats.open_record(record)
                elif self.next_button.clicked(event, enabled=bool(self.dialogue.pages)):
                    self.advance_conversation(app)
                elif self.back_button.clicked(event, enabled=self.dialogue.index > 0):
                    self.rewind_conversation()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    self.rewind_conversation()
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE, pygame.K_RIGHT):
                    self.advance_conversation(app)
                elif self.retry_button.clicked(event, enabled=not paused and bool(
                        self.dialogue.conversation.error and not self.dialogue.conversation.record.turns)):
                    app._sync_conversation_view()
                    app.conversations.retry_conversation_dialogue(self.dialogue.conversation)
                    app._present_conversation_events()
                return
            if self.back_button.clicked(event, enabled=self.dialogue.index > 0):
                self.dialogue.move(-1)
                self.prompt_input.active = self.prompt_input.backspace_held = False
                return
            if not self.player_response_ready:
                if self.next_button.clicked(event) or (event.type == pygame.KEYDOWN and event.key in (
                        pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE, pygame.K_RIGHT)):
                    self.advance_player_dialogue(app)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT and self.dialogue.index > 0:
                    self.dialogue.move(-1)
                return
            if self.pending_offer:
                return
            if self.dialogue.actor.player_reply.closing:
                if self.done_button.clicked(event) or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE)
                ):
                    self.close_dialogue()
                return
            if paused:
                return
            if self.send_button.clicked(event):
                app._sync_conversation_view()
                app.conversations.submit(app.game.dialogue.actor, app.game.prompt_input.text, active=app.screen == "game")
                app._present_conversation_events()
                self.focus_player_reply(app)
            elif self.prompt_input.handle_event(event):
                app._sync_conversation_view()
                app.conversations.submit(app.game.dialogue.actor, app.game.prompt_input.text, active=app.screen == "game")
                app._present_conversation_events()

    @property
    def player_response_ready(self):
        return self.dialogue.conversation is None and self.dialogue.final_page

    def update_player_pages(self, app):
        changed = self.dialogue.refresh(self.player_record, app.font, self.reply_rect.width - 12,
            original=app.original_dialogue, condensed=getattr(app, "condensed_dialogue", False), player_reply=True)
        if changed:
            self.prompt_input.active = self.prompt_input.backspace_held = False
        if self.player_response_ready and not self.dialogue.actor.player_reply.closing and not self.pending_offer:
            if changed:
                self.prompt_input.focus()

    def focus_player_reply(self, app):
        if not self.dialogue.open or self.dialogue.conversation is not None or self.dialogue.actor is None:
            return
        self.update_player_pages(app)
        if self.player_response_ready and not self.dialogue.actor.player_reply.closing and not self.pending_offer:
            self.prompt_input.focus()

    def advance_player_dialogue(self, app):
        if self.player_response_ready:
            return
        self.dialogue.move(1)
        if self.player_response_ready:
            self.focus_player_reply(app)

    def handle_clock_event(self, app, event):
        if self.pause_button.clicked(event):
            app.simulation.paused = not app.simulation.paused
            self.pan_start = self.pan_last = None
            self.panning = False
            return True
        if self.clock_button.clicked(event):
            from player import active_player
            if active_player(app.simulation):
                self.show_save_message("To pass time, go to your room and click your bed.")
            elif not app.simulation.paused and not app.simulation.phase_pending and not app.simulation.player_attention_pending:
                app.simulation.next_phase()
            return True
        return False

    def can_use_bed(self, world):
        from player import active_player
        return (active_player(world) and self.location == world.player.location == PLAYER_ROOM_LOCATION
                and not world.player_interaction and not world.player.journey
                and not (world.visit and world.visit.stage == "travel"))

    def can_advance_from_bed(self, world):
        return (self.can_use_bed(world) and not world.paused
                and not world.phase_pending and not world.player_attention_pending)

    def handle_bed_event(self, app, event):
        if self.bed_cancel_button.clicked(event) or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.bed_open = False
        elif self.bed_advance_button.clicked(event, enabled=self.can_advance_from_bed(app.simulation)):
            self.bed_open = False
            app.simulation.next_day()

    def draw_bed_popup(self, app):
        if not self.bed_open:
            return
        world = app.simulation
        overlay = pygame.Surface(app.window.get_size(), pygame.SRCALPHA)
        overlay.fill(MODAL_OVERLAY_COLOR)
        app.window.blit(overlay, (0, 0))
        pygame.draw.rect(app.window, PAPER, BED_POPUP_RECT, border_radius=BORDER_RADIUS)
        draw_centered_text(app.window, "Rest in your bed", app.heading_font, BED_POPUP_TITLE_Y)
        message = ("Resume the game before passing time." if world.paused else
                   "Time will change after the current conversations finish." if world.phase_pending else
                   "Sleep until tomorrow morning? Current conversations and handoffs will finish first. A new resident arrives if there is room.")
        draw_wrapped_text(app.window, message, app.font, INK, BED_POPUP_MESSAGE_RECT)
        self.bed_advance_button.label = f"Sleep until Day {world.day + 1}"
        self.bed_advance_button.draw(app.window, app.small_font, enabled=self.can_advance_from_bed(world))
        self.bed_cancel_button.draw(app.window, app.small_font)

    def open_menu_confirmation(self):
        self.confirm_menu = True
        self.menu_restore_focus = self.prompt_input.active
        self.prompt_input.active = self.prompt_input.backspace_held = False
        self.pan_start = self.pan_last = None
        self.panning = False

    def draw_menu_confirmation(self, app):
        if not self.confirm_menu:
            return
        overlay = pygame.Surface(app.window.get_size(), pygame.SRCALPHA)
        overlay.fill(MODAL_OVERLAY_COLOR)
        app.window.blit(overlay, (0, 0))
        pygame.draw.rect(app.window, PAPER, GAME_MENU_CONFIRM_RECT, border_radius=BORDER_RADIUS)
        draw_centered_text(app.window, "Return to main menu?", app.heading_font, GAME_MENU_CONFIRM_TITLE_Y)
        draw_wrapped_text(app.window, "Progress since your last save will be lost.", app.font, INK,
                          GAME_MENU_CONFIRM_MESSAGE_RECT)
        self.keep_playing_button.draw(app.window, app.small_font)
        self.leave_game_button.draw(app.window, app.small_font)

    def draw_clock(self, app):
        from player import active_player
        paused = app.simulation.paused
        self.pause_button.label = "Resume" if paused else "Pause"
        self.pause_button.draw(app.window, app.small_font)
        self.clock_button.draw(app.window, app.small_font, enabled=not paused and not app.simulation.phase_pending
                               and not app.simulation.player_attention_pending and not active_player(app.simulation))
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
        conversation = self.dialogue.conversation
        if conversation is None or not self.dialogue.pages:
            return
        if self.dialogue.index == len(self.dialogue.pages) - 1:
            if conversation.active:
                if app.simulation.paused:
                    return
                conversation.dialogue.talking_remaining = 0.0
                conversation.update(0.0)
            self.close_dialogue()
        else:
            self.dialogue.move(1)

    def rewind_conversation(self):
        if self.dialogue.conversation is not None and self.dialogue.index > 0:
            self.dialogue.move(-1)

    def handle_dialogue_scroll(self, app, event):
        font = app.font
        value = self.dialogue_scrollbar.handle_event(event, self.dialogue_scroll_rect,
                                                    self.dialogue.scroll, self.dialogue.scroll_limit)
        if value is not None:
            self.dialogue.scroll = value
            return True
        if event.type == pygame.MOUSEWHEEL:
            pos = getattr(event, "pos", pygame.mouse.get_pos())
            if self.dialogue.conversation is None and not self.dialogue_scroll_rect.collidepoint(pos):
                return False
            self.dialogue.scroll = max(0, self.dialogue.scroll - event.y * font.get_linesize() * 3)
            return True
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_DOWN, pygame.K_PAGEDOWN):
                self.dialogue.scroll += font.get_linesize() if event.key == pygame.K_DOWN else 80
                return True
            if event.key in (pygame.K_UP, pygame.K_PAGEUP):
                self.dialogue.scroll = max(0, self.dialogue.scroll - (font.get_linesize() if event.key == pygame.K_UP else 80))
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
        from player import active_player, can_interrupt_approach, start_approach
        from locations import targets
        world = app.simulation
        if active_player(world) and (world.player_interaction or world.player.journey
                                     or world.visit and world.visit.stage == "travel"):
            return
        point = self.screen_to_world(screen_pos)
        if self.can_use_bed(world) and pygame.Rect(ROOM_BED_RECT).collidepoint(point):
            self.close_dialogue()
            self.bed_open = True
            self.pan_start = self.pan_last = None
            self.panning = False
            return
        actors = [n for n in world.npcs if n.location == self.location]
        if self.location == "garden":
            actors.append(world.snake)
        hit_actor = next((n for n in sorted(actors, key=lambda n: n.y, reverse=True) if n.contains(point)), None)
        for target in targets(world, self.location):
            if hit_actor is None and pygame.Rect(target["rect"]).collidepoint(point):
                if active_player(world):
                    from locations import start_journey
                    self.close_dialogue()
                    start_journey(world.player, self.location, target["point"], purpose="door", fast=True)
                    world.player_interaction = PlayerInteraction(id="door", kind="door", stage="travel", target=target,
                        location=self.location, npc_id=None, record_id=None)
                else:
                    app.use_door(target)
                return
        if active_player(world) and self.location == "garden" and hit_actor is None and abs(point[0] - world.tree.x) < 70 and abs(point[1] - world.tree.y) < 100:
            from locations import start_journey
            self.close_dialogue()
            start_journey(world.player, "garden", world.tree.interaction_point, purpose="tree", fast=True)
            return
        npc = hit_actor
        if npc is None:
            self.close_dialogue()
            return
        interaction = npc.conversation
        if interaction and (not active_player(world) or not can_interrupt_approach(npc)):
            self.close_dialogue()
            self.dialogue.show(npc, interaction)
            return
        if not active_player(world):
            self.show_save_message(f"{npc.name}: {npc.status}")
            return
        try:
            self.close_dialogue()
            start_approach(world, npc)
        except ValueError as exc:
            self.show_save_message(str(exc))

    def can_harvest(self, world):
        import math
        from constants import HARVEST_DISTANCE
        from player import can_claim
        return (not world.paused and self.location == "garden" and can_claim(world)
                and not self.dialogue.open and math.dist((world.player.x, world.player.y),
                                                        world.tree.interaction_point) <= HARVEST_DISTANCE)

    def draw(self, app):
        from player import active_player
        draw_world(app.window, app.npcs, app.tree, app.small_font, app.character.renderer,
                   self.view_offset, phase=app.simulation.phase, zoom=self.camera_zoom,
                   viewport=self.world_viewport, snake=app.simulation.snake, world=app.simulation, location=self.location)
        if self.can_harvest(app.simulation):
            self.harvest_button.draw(app.window, app.small_font)
            self.harvest_amount.draw(app.window, app.small_font)
        if self.location_menu:
            self.draw_location_menu(app)
        self.menu_button.draw(app.window, app.small_font)
        self.stats_button.draw(app.window, app.small_font)
        self.save_button.draw(app.window, app.small_font)
        if active_player(app.simulation):
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
            color = PAPER if app.simulation.phase == "night" or self.location != "garden" else INK
            draw_wrapped_text(app.window, self.save_message, app.small_font, color, (24, 480, 1232, 44))

        if self.dialogue.open and app.simulation.player_interaction:
            self.end_button.draw(app.window, app.small_font)
        if self.dialogue.open and active_player(app.simulation):
            reason = app.simulation.player_killing_unavailable(self.dialogue.actor)
            self.kill_button.draw(app.window, app.small_font, enabled=not reason)
        if (self.dialogue.open and self.dialogue.conversation is None
                and self.dialogue.actor is not app.simulation.snake and self.player_response_ready):
            reason = app.conversations.input_unavailable(self.dialogue.actor, active=app.screen == "game")
            self.actions_button.draw(app.window, app.small_font, enabled=not reason and not self.pending_offer)
        if not self.dialogue.open:
            if self.apartment_open:
                self.draw_apartments(app)
            from constants import LOCATION_STATUS_POSITION
            app.window.blit(app.small_font.render(f"{app.simulation.mode.title()} / {self.location}", True, PAPER if self.location != "garden" else INK), LOCATION_STATUS_POSITION)
            return
        if self.dialogue.conversation is not None:
            self.draw_conversation(app)
            return
        self.update_player_pages(app)
        responding = self.player_response_ready
        page_index = min(self.dialogue.index, len(self.dialogue.pages) - 1)
        page = self.dialogue.pages[page_index] if self.dialogue.pages else None
        turn, part, text = page if page else (None, None, "Enter a message to talk to this NPC.")
        original = app.original_dialogue or turn is None or uses_original_dialogue(
            turn.parts, turn.rewritten, original=app.original_dialogue)
        self.draw_dialogue_background(app, original)
        speaker = app.simulation.participant_by_id(turn.speaker_id) if turn else self.dialogue.actor
        emotion = portrait_expression(part["name"], part["intensity"]) if part is not None else None
        self.draw_portrait(app, speaker, emotion)
        status = self.dialogue.actor.status if self.dialogue.actor is not None else f"{app.simulation.population} residents"
        request = app.conversations.pending_player_request(self.dialogue.actor)
        if request:
            worker = app.requests.workers.get((app.session_id, request.id))
            state = "Queued" if worker and worker.control and worker.control.stage == "Waiting for provider admission" else "Thinking"
            status = f"{state} for {self.dialogue.actor.name}... / {status}"
        elif self.dialogue.actor.player_reply.closing:
            status = "Click Done when you've finished reading."
        elif self.dialogue.actor.interaction_blocked:
            status = f"Applying player request / {status}"
        if self.pending_offer:
            offer = self.pending_offer
            work = next((w for w in app.simulation.creations if w.id == offer["item"]), None)
            status = f'Gift offer: {offer["quantity"]} × {work.title + " [" + work.id[:8] + "]" if work else "apples"}'
        elif self.player_record and self.player_record.error:
            self.retry_button.draw(app.window, app.small_font, enabled=not app.simulation.paused and not self.dialogue.actor.interaction_blocked)
            status = "Reply failed."
        title = app.font.render(turn.speaker if turn else "Request status" if page else self.dialogue.actor.name, True, INK)
        title_rect = title.get_rect(topleft=(self.reply_rect.x, 557))
        app.window.blit(title, title_rect)
        status_right = self.retry_button.rect.left if self.player_record and self.player_record.error else self.back_button.rect.left
        draw_wrapped_text(app.window, status, app.small_font, MUTED,
                          (title_rect.right + 16, 558, max(0, status_right - title_rect.right - 24), 24))
        self.back_button.draw(app.window, app.small_font, enabled=self.dialogue.index > 0)
        closing = self.dialogue.actor.player_reply.closing
        self.next_button.label = "Next"
        self.next_button.draw(app.window, app.small_font, enabled=not responding)
        rect = self.reply_rect if responding else pygame.Rect(self.reply_rect.x, DIALOGUE_READING_Y,
                                                             self.reply_rect.width, DIALOGUE_READING_HEIGHT)
        self.draw_dialogue_lines(app.window, wrap_text(text, app.font, rect.width - 12), app.font, rect,
                                 color=INK if turn else MUTED)
        if responding and self.pending_invitation(app.simulation):
            self.invitation_button.draw(app.window, app.small_font)
        if responding and self.pending_offer:
            self.accept_offer_button.draw(app.window, app.small_font, enabled=not app.simulation.paused)
            self.decline_offer_button.draw(app.window, app.small_font, enabled=not app.simulation.paused)
        elif responding and closing:
            self.done_button.draw(app.window, app.font)
        elif responding:
            self.prompt_input.draw(app.window, app.font)
            self.send_button.draw(app.window, app.font, enabled=not self.dialogue.actor.interaction_blocked and not app.simulation.paused)
        if self.player_record and self.player_record.dialogue_errors:
            draw_wrapped_text(app.window, self.player_record.dialogue_errors[-1], app.small_font,
                              FAILED_CONVERSATION_SELECTED_COLOR, (self.reply_rect.x, 696, self.reply_rect.width - 90, 24))
        elif responding and not closing and self.prompt_input.error:
            app.window.blit(app.small_font.render(self.prompt_input.error, True, MUTED), (self.reply_rect.x, 696))
        else:
            reason = self.dialogue.actor.decision_reason
            while reason and app.small_font.size(reason)[0] > self.reply_rect.width - 90:
                reason = reason[:-4] + "..."
            app.window.blit(app.small_font.render(reason, True, MUTED), (self.reply_rect.x, 696))
        if responding and not closing:
            count = app.small_font.render(f"{len(self.prompt_input.text)}/300", True, MUTED)
            app.window.blit(count, count.get_rect(topright=(1256, 696)))
        elif not responding:
            progress = app.small_font.render(f"{self.dialogue.index + 1}/{len(self.dialogue.pages)}", True, MUTED)
            app.window.blit(progress, progress.get_rect(topright=(1256, 696)))
        if self.dialogue.picker is not None:
            self.draw_action_picker(app)

    def picker_options(self, app):
        from player_commands import command_options
        command = self.dialogue.picker["command"]
        if command.get("item") == "fruit":
            quantity = command["quantity"]
            stock = self.dialogue.actor.inventory.get("fruit", 0)
            return "Choose apple quantity", [
                ("Fewer apples", "less", "" if quantity > 1 else "Minimum one"),
                ("More apples", "more", "" if quantity < stock else "All carried apples"),
                (f"Request {quantity} apple" + ("s" if quantity != 1 else ""), command.copy(),
                 "Not enough carried apples" if quantity > stock else "")]
        title = "Choose an action" if not command else "Choose a recipient" if not command.get("target") else "Choose an item"
        return title, command_options(app.simulation, self.dialogue.actor, command.get("action"), command.get("target"))

    def picker_footer(self):
        rect = pygame.Rect(PLAYER_ACTION_PICKER_RECT)
        inset, width = PLAYER_ACTION_PICKER_INSET, PLAYER_ACTION_PICKER_BUTTON_WIDTH
        return [Button((rect.x + inset + i * (width + inset), rect.bottom - PLAYER_ACTION_PICKER_FOOTER_HEIGHT,
                        width, PLAYER_ACTION_PICKER_FOOTER_BUTTON_HEIGHT), label)
                for i, label in enumerate(("Back", "Previous", "Next", "Cancel"))]

    def handle_action_picker(self, app, event):
        picker = self.dialogue.picker
        back, previous, following, cancel = self.picker_footer()
        if cancel.clicked(event) or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            focus = picker["focus"]
            self.dialogue.picker = None
            if focus:
                self.focus_player_reply(app)
            return
        if back.clicked(event):
            command = picker["command"]
            picker["command"] = ({"action": command["action"], "target": command["target"]} if command.get("item") else
                                 {"action": command["action"]} if command.get("target") else {})
            picker["page"] = 0
            picker["buttons"] = []
            return
        if previous.clicked(event, enabled=picker["page"] > 0):
            picker["page"] -= 1
            picker["buttons"] = []
            return
        if following.clicked(event, enabled=picker.get("more_pages", False)):
            picker["page"] += 1
            picker["buttons"] = []
            return
        reason = app.conversations.input_unavailable(self.dialogue.actor, active=app.screen == "game")
        for button, value, unavailable in picker.get("buttons", []):
            if not button.clicked(event, enabled=not reason and not unavailable):
                continue
            if value in ("less", "more"):
                picker["command"]["quantity"] += -1 if value == "less" else 1
                picker["buttons"] = []
                return
            action = value["action"]
            complete = (action not in ("TALK_TO_NPC", "GIVE_ITEM") or action == "TALK_TO_NPC" and value.get("target")
                        or action == "GIVE_ITEM" and value.get("item") and
                        (value["item"] != "fruit" or picker["command"].get("item") == "fruit"))
            if complete:
                self.dialogue.picker = None
                app._sync_conversation_view()
                app.conversations.submit(self.dialogue.actor, "", command=value, active=app.screen == "game")
                app._present_conversation_events()
            else:
                picker["command"], picker["page"], picker["buttons"] = value, 0, []
            return

    def draw_action_picker(self, app):
        picker = self.dialogue.picker
        rect = pygame.Rect(PLAYER_ACTION_PICKER_RECT)
        pygame.draw.rect(app.window, PAPER, rect, border_radius=BORDER_RADIUS)
        pygame.draw.rect(app.window, INK, rect, width=PLAYER_ACTION_PICKER_BORDER_WIDTH, border_radius=BORDER_RADIUS)
        title, options = self.picker_options(app)
        inset = PLAYER_ACTION_PICKER_INSET
        draw_wrapped_text(app.window, title, app.font, INK,
                          (rect.x + inset, rect.y + inset, rect.width - inset * 2, PLAYER_ACTION_PICKER_TITLE_HEIGHT - inset))
        limit = max(0, (len(options) - 1) // PLAYER_ACTION_PICKER_ROWS)
        picker["page"] = min(picker["page"], limit)
        picker["more_pages"] = picker["page"] < limit
        picker["buttons"] = []
        reason = app.conversations.input_unavailable(self.dialogue.actor, active=app.screen == "game")
        start = picker["page"] * PLAYER_ACTION_PICKER_ROWS
        for index, (label, value, unavailable) in enumerate(options[start:start + PLAYER_ACTION_PICKER_ROWS]):
            label += f" - {reason or unavailable}" if reason or unavailable else ""
            while app.small_font.size(label)[0] > rect.width - inset * 3:
                label = label[:-4] + "..."
            button = Button((rect.x + inset, rect.y + PLAYER_ACTION_PICKER_TITLE_HEIGHT + index * PLAYER_ACTION_PICKER_ROW_HEIGHT,
                             rect.width - inset * 2, PLAYER_ACTION_PICKER_ROW_BUTTON_HEIGHT), label)
            button.draw(app.window, app.small_font, enabled=not reason and not unavailable)
            picker["buttons"].append((button, value, unavailable))
        if not options:
            draw_wrapped_text(app.window, "No eligible choices are available right now.", app.small_font, MUTED,
                              (rect.x + inset, rect.y + PLAYER_ACTION_PICKER_TITLE_HEIGHT,
                               rect.width - inset * 2, PLAYER_ACTION_PICKER_ROW_HEIGHT * 2))
        back, previous, following, cancel = self.picker_footer()
        back.draw(app.window, app.small_font, enabled=bool(picker["command"]))
        previous.draw(app.window, app.small_font, enabled=picker["page"] > 0)
        following.draw(app.window, app.small_font, enabled=picker["more_pages"])
        cancel.draw(app.window, app.small_font)

    def apartment_rows(self, app):
        rect = pygame.Rect(APARTMENT_INSPECTOR_RECT)
        content = rect.inflate(*(-value for value in APARTMENT_INSPECTOR_CONTENT_INSET))
        content.top = rect.top + APARTMENT_INSPECTOR_CONTENT_TOP
        rows, top = [], 0
        for npc in sorted((npc for npc in app.npcs if npc.indoors), key=lambda npc: npc.room_number):
            for paragraph in (f"Room {npc.room_number}: {npc.name}", npc.status):
                for line in wrap_text(paragraph, app.small_font, content.width - 12):
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
        if limit:
            track, thumb = scrollbar_rects(content, self.apartment_scroll, limit)
            pygame.draw.rect(app.window, MUTED, track)
            pygame.draw.rect(app.window, INK, thumb)

    def draw_dialogue_background(self, app, original):
        top = WINDOW_SIZE[1] - DIALOGUE_PANEL_HEIGHT
        color = ORIGINAL_DIALOGUE_PAPER if original else PAPER
        pygame.draw.rect(app.window, color, (0, top, WINDOW_SIZE[0], DIALOGUE_PANEL_HEIGHT))
        pygame.draw.line(app.window, (129, 154, 110), (0, top), (WINDOW_SIZE[0], top), 3)
        self.original_toggle.label = "Original: " + ("On" if app.original_dialogue else "Off")
        self.original_toggle.draw(app.window, app.small_font)

    def draw_portrait(self, app, npc, emotion=None):
        draw_actor_portrait(app.window, npc, self.portrait_rect, app.character.renderer,
            snake=app.simulation.snake, padding=GAME_PORTRAIT_PADDING, background=GAME_PORTRAIT_BACKGROUND,
            emotion=emotion, border_color=INK, border_width=GAME_PORTRAIT_BORDER_WIDTH)

    def draw_conversation(self, app):
        conversation = self.dialogue.conversation
        rect = pygame.Rect(self.reply_rect.x, DIALOGUE_READING_Y, self.reply_rect.width, DIALOGUE_READING_HEIGHT)
        source = conversation.record.turns or conversation.record.failed_turns
        failed = bool(source) and not conversation.record.turns
        self.dialogue.refresh(conversation.record, app.font, rect.width - 12,
            original=app.original_dialogue, condensed=getattr(app, "condensed_dialogue", False))
        page = self.dialogue.pages[self.dialogue.index] if self.dialogue.pages else None
        turn, part, text = page if page is not None else (None, None, "")
        original = app.original_dialogue or failed
        if turn is not None and not failed:
            original = uses_original_dialogue(turn.parts, turn.rewritten, original=app.original_dialogue)
        self.draw_dialogue_background(app, original)
        if turn is not None:
            speaker = app.simulation.participant_by_id(turn.speaker_id)
            title = turn.speaker + (" - failed draft" if failed else "")
        else:
            speaker = self.dialogue.actor
            title = f"{conversation.initiator.name} & {conversation.recipient.name}"
        emotion = portrait_expression(part["name"], part["intensity"]) if part is not None else None
        self.draw_portrait(app, speaker, emotion)
        title_right = self.subjects_button.rect.left if conversation.record.intended_subjects else self.retry_button.rect.left
        draw_wrapped_text(app.window, title, app.font, INK, (self.reply_rect.x, 557, title_right - self.reply_rect.x - 16, 30))
        if conversation.record.intended_subjects:
            self.subjects_button.draw(app.window, app.small_font)
        self.back_button.draw(app.window, app.small_font, enabled=self.dialogue.index > 0)
        last_page = bool(self.dialogue.pages) and self.dialogue.index == len(self.dialogue.pages) - 1
        self.next_button.label = "Done" if last_page else "Next"
        can_finish = not (last_page and conversation.active and app.simulation.paused)
        self.next_button.draw(app.window, app.small_font, enabled=bool(self.dialogue.pages) and can_finish)
        if conversation.error and not conversation.record.turns:
            self.retry_button.draw(app.window, app.small_font, enabled=not app.simulation.paused)
        if not conversation.arrived:
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
            progress = app.small_font.render(f"{self.dialogue.index + 1}/{len(self.dialogue.pages)}", True, MUTED)
            app.window.blit(progress, progress.get_rect(topright=(1256, 696)))

    def draw_dialogue_lines(self, surface, lines, font, rect, *, color=INK):
        line_height = font.get_linesize()
        total_height = len(lines) * line_height
        scroll_limit = max(0, total_height - rect.height)
        self.dialogue_scroll_rect = rect.copy()
        self.dialogue.scroll_limit = scroll_limit
        self.dialogue.scroll = max(0, min(self.dialogue.scroll, scroll_limit))
        previous_clip = surface.get_clip()
        surface.set_clip(rect.clip(previous_clip))
        for index, line in enumerate(lines):
            y = rect.y + index * line_height - self.dialogue.scroll
            if rect.top - line_height < y < rect.bottom:
                surface.blit(font.render(line, True, color), (rect.x, y))
        if scroll_limit:
            track, thumb = scrollbar_rects(rect, self.dialogue.scroll, scroll_limit)
            pygame.draw.rect(surface, (181, 191, 174), track)
            pygame.draw.rect(surface, MUTED, thumb)
        surface.set_clip(previous_clip)

    def pending_invitation(self, world):
        from apartments import pending_invitation
        return pending_invitation(world)

    def open_location_menu(self, choices):
        self.location_menu = dict(choices=choices)

    def observe_location(self, location):
        from locations import map_size
        self.close_dialogue()
        self.location = location
        width, height = map_size(location)
        self.camera_offset[:] = ((WORLD_WIDTH - width * self.camera_zoom) / 2,
                                 (WORLD_HEIGHT - height * self.camera_zoom) / 2)

    def location_buttons(self):
        from constants import (LOCATION_MENU_RECT, LOCATION_BUTTON_HEIGHT, LOCATION_BUTTON_GAP,
            LOCATION_BUTTON_WIDTH, LOCATION_BUTTON_INSET)
        x, y = LOCATION_MENU_RECT[:2]
        return [(Button((x + LOCATION_BUTTON_INSET[0], y + LOCATION_BUTTON_INSET[1] + row * LOCATION_BUTTON_GAP,
            LOCATION_BUTTON_WIDTH, LOCATION_BUTTON_HEIGHT), label), value)
            for row, (label, value) in enumerate(self.location_menu["choices"])]

    def handle_location_event(self, app, event):
        from player import active_player
        from constants import LOCATION_MENU_RECT
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                or event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and not pygame.Rect(LOCATION_MENU_RECT).collidepoint(event.pos)):
            self.location_menu = None
            return
        for button, value in self.location_buttons():
            if button.clicked(event):
                if not active_player(app.simulation):
                    self.observe_location(value)
                elif value != app.simulation.player.location:
                    from locations import start_journey
                    from constants import ELEVATOR_ARRIVAL
                    start_journey(app.simulation.player, value, ELEVATOR_ARRIVAL, purpose="elevator", fast=True)
                self.location_menu = None
                return

    def draw_location_menu(self, app):
        from constants import LOCATION_MENU_RECT
        overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
        overlay.fill(MODAL_OVERLAY_COLOR)
        app.window.blit(overlay, (0, 0))
        pygame.draw.rect(app.window, PAPER, LOCATION_MENU_RECT, border_radius=BORDER_RADIUS)
        for button, _ in self.location_buttons():
            button.draw(app.window, app.small_font)

    def update_physical(self, app, dt):
        from player import active_player, move_player
        from locations import map_size
        from constants import GARDEN_LOCATION
        world = app.simulation
        self.app = app
        if not active_player(world) and self.inventory.visible and self.inventory.owner is world.player:
            self.inventory.close()
        if self.bed_open and not self.can_use_bed(world):
            self.bed_open = False
        world.player_menu_blocked = bool(self.location_menu or self.inventory.visible or self.apartment_open
            or self.confirm_menu or self.bed_open or app.screen != "game")
        if self.confirm_menu:
            return
        if active_player(world):
            self.location = world.player.location
            interaction = world.player_interaction
            if interaction and interaction.kind in ("resident", "snake"):
                if interaction.stage == "conversation" and not self.dialogue.open:
                    app._sync_conversation_view()
                    app.conversations.open_player_dialogue(world.actor_by_id(interaction.npc_id))
                    app._present_conversation_events()
            elif interaction and interaction.kind == "door" and not world.player.journey:
                target = interaction.target
                world.player_interaction = None
                app.use_door(target)
            blocking = (world.player_menu_blocked or self.dialogue.open or self.prompt_input.active
                        or pygame.display.get_init() and pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_ALT))
            if not blocking and app.screen == "game" and dt > 0 and pygame.display.get_init():
                keys = pygame.key.get_pressed()
                move_player(world, keys[pygame.K_d] - keys[pygame.K_a], keys[pygame.K_s] - keys[pygame.K_w], dt,
                            running=bool(keys[pygame.K_SPACE]))
            width, height = map_size(self.location)
            # Dialogue clips the world view but must not move the camera anchor.
            # Follow freely in the garden, even when it fits inside the window.
            base = ((WINDOW_SIZE[0] - WORLD_WIDTH) / 2, (WINDOW_SIZE[1] - WORLD_HEIGHT) / 2)
            # Positions mark the feet; follow the standing body's center above them.
            body = app.character.renderer.body
            center_y = world.player.y - (body.get_height() - body.get_bounding_rect().centery) * CHARACTER_SCALE
            desired = []
            for extent, view, position in ((width, WINDOW_SIZE[0], world.player.x), (height, WINDOW_SIZE[1], center_y)):
                scaled = extent * self.camera_zoom
                if self.location == GARDEN_LOCATION:
                    desired.append(view / 2 - position * self.camera_zoom)
                else:
                    desired.append((view - scaled) / 2 if scaled <= view else min(0, max(view - scaled, view / 2 - position * self.camera_zoom)))
            self.camera_offset[:] = [desired[i] - base[i] for i in (0, 1)]
        elif self.dialogue.actor and not self.dialogue.conversation:
            self.close_dialogue()

    def request_is_current(self, app):
        return (self.dialogue.conversation is not None
                and app.conversation_requests.get(self.dialogue.conversation.id) is self.dialogue.conversation)
