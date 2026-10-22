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
from .text import draw_centered_text, draw_wrapped_text, paginate_parts, wrap_text
from .text_input import TextInput
from .world import draw_world, snake_artwork
from .inventory import InventoryPanel
from constants import (INVENTORY_BUTTON_RECT, GAME_PAUSE_RECT, GAME_CLOCK_RECT, GAME_PHASE_POSITION,
                       GAME_PAUSED_LABEL_RIGHT, GIFT_ACCEPT_RECT, GIFT_DECLINE_RECT, CALL_CHOICE_RECT,
                       CALL_CHOICE_TITLE_RECT, CALL_CHOICE_STATUS_RECT, CALL_CHOICE_BUTTON_RECTS,
                       DIALOGUE_ORIGINAL_TOGGLE_RECT)
from constants import (BORDER_RADIUS, GAME_MENU_CONFIRM_RECT, GAME_MENU_CONFIRM_TITLE_Y,
                       GAME_MENU_CONFIRM_MESSAGE_RECT, GAME_MENU_KEEP_PLAYING_RECT, GAME_MENU_LEAVE_RECT)


class GameScreen:
    def __init__(self, max_requests=MAX_LLM_REQUESTS):
        from constants import END_CONVERSATION_RECT, LEAVE_QUEUE_RECT, OBSERVE_LOCATION_RECT, INVITATION_ACCEPT_RECT
        self.app = None
        self.location = "garden"
        self.location_menu = None
        self.location_page = 0
        self.end_button = Button(END_CONVERSATION_RECT, "End conversation")
        self.leave_queue_button = Button(LEAVE_QUEUE_RECT, "Leave queue")
        self.observe_button = Button(OBSERVE_LOCATION_RECT, "Observe location")
        self.invitation_button = Button(INVITATION_ACCEPT_RECT, "Accept invitation")
        self.menu_button = Button((1144, 16, 112, 38), "Menu")
        self.confirm_menu = False
        self.menu_restore_focus = False
        self.keep_playing_button = Button(GAME_MENU_KEEP_PLAYING_RECT, "Keep playing")
        self.leave_game_button = Button(GAME_MENU_LEAVE_RECT, "Return to menu")
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
                             for label, rect in zip(("Talk", "Decline"), CALL_CHOICE_BUTTON_RECTS)}
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
        self.original_toggle = Button(DIALOGUE_ORIGINAL_TOGGLE_RECT, "Original: Off")
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
        self.confirm_menu = False
        self.menu_restore_focus = False
        self.inventory.close()
        self.call_choice = None
        self.apartment_open = False
        self.location_menu = None
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

    def close_dialogue(self, *, terminate=True):
        if terminate and self.app and self.app.simulation:
            self.app.end_player_conversation()
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
        from player import active_player
        world = app.simulation
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
        if self.observe_button.clicked(event) and not active_player(world):
            self.close_dialogue()
            from locations import allocated_floors, room_location, hallway_location
            self.open_location_menu("Observe", [("Garden", "garden"),
                *[(f"Hallway floor {n}", hallway_location(n)) for n in allocated_floors(world)],
                ("Room 1 (reserved)", room_location(1)),
                *[(f"Room {n.room_number}: {n.name}", room_location(n.room_number)) for n in world.residents]])
            return
        if self.leave_queue_button.clicked(event) and world.can_leave_tree(world.player.id):
            world.leave_tree(world.player.id)
            return
        if self.dialogue_open and world.player_interaction and self.end_button.clicked(event):
            self.close_dialogue()
            return
        if self.dialogue_open and self.invitation_button.clicked(event):
            offer = self.pending_invitation(world)
            if offer:
                app.accept_player_invitation(offer["id"])
                return
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
                        app.end_player_conversation()
                        self.call_choice = None
            return
        if active_player(world) and self.inventory_button.clicked(event):
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
            if self.dialogue_open:
                self.close_dialogue()
            else:
                self.open_menu_confirmation()
            return
        if self.dialogue_open:
            if self.original_toggle.clicked(event):
                app.original_dialogue = not app.original_dialogue
                self.dialogue_scroll = 0
                return
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
        from player import active_player, start_approach
        from locations import targets
        world = app.simulation
        if world.player_interaction or world.player.journey or world.visit and world.visit["stage"] == "travel":
            return
        point = self.screen_to_world(screen_pos)
        actors = [n for n in world.npcs if n.location == self.location]
        if self.location == "garden":
            actors.append(world.snake)
        hit_actor = next((n for n in sorted(actors, key=lambda n: n.y, reverse=True) if n.contains(point)), None)
        if active_player(world):
            for target in targets(world, self.location):
                if hit_actor is None and pygame.Rect(target["rect"]).collidepoint(point):
                    if world.tree_has_actor(world.player.id):
                        return
                    from locations import start_journey
                    self.close_dialogue()
                    start_journey(world.player, self.location, target["point"], purpose="door", fast=True)
                    world.player_interaction = dict(id="door", kind="door", stage="travel", target=target,
                        location=self.location, npc_id=None, record_id=None)
                    return
            if self.location == "garden" and hit_actor in (None, world.snake) and (world.snake.contains(point) or abs(point[0] - world.tree.x) < 70 and abs(point[1] - world.tree.y) < 100):
                self.close_dialogue()
                if not world.join_tree(world.player, "introduction" if world.snake.contains(point) else "bargain"):
                    self.show_save_message("Tree service is unavailable.")
                return
        npc = hit_actor
        if npc is None:
            self.close_dialogue()
            return
        interaction = npc.conversation or getattr(npc, "snake_encounter", None)
        if interaction:
            self.close_dialogue()
            self.dialogue_open, self.selected_npc, self.viewed_conversation = True, npc, interaction
            self.conversation_tree = world.tree
            return
        if not active_player(world):
            self.show_save_message(f"{npc.name}: {npc.status}")
            return
        try:
            self.close_dialogue()
            start_approach(world, npc)
        except ValueError as exc:
            self.show_save_message(str(exc))

    def draw(self, app):
        from player import active_player
        draw_world(app.window, app.npcs, app.tree, app.small_font, app.character.renderer,
                   self.view_offset, phase=app.simulation.phase, zoom=self.camera_zoom,
                   viewport=self.world_viewport, snake=app.simulation.snake, world=app.simulation, location=self.location)
        if app.simulation.mode == "spectator":
            self.observe_button.draw(app.window, app.small_font)
        if app.simulation.can_leave_tree(app.simulation.player.id):
            self.leave_queue_button.draw(app.window, app.small_font)
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
            color = PAPER if app.simulation.phase == "night" else INK
            draw_wrapped_text(app.window, self.save_message, app.small_font, color, (24, 480, 1232, 44))

        if self.dialogue_open and app.simulation.player_interaction:
            self.end_button.draw(app.window, app.small_font)
        if not self.dialogue_open:
            if self.apartment_open:
                self.draw_apartments(app)
            from constants import LOCATION_STATUS_POSITION
            app.window.blit(app.small_font.render(f"{app.simulation.mode.title()} / {self.location}", True, PAPER if self.location != "garden" else INK), LOCATION_STATUS_POSITION)
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
        status = self.selected_npc.status if self.selected_npc is not None else f"{app.simulation.population} residents"
        request = app.pending_player_request(self.selected_npc)
        if request:
            state = "Queued" if (app.session_id, request["id"]) in app.requests.pending else "Thinking"
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
        if self.pending_invitation(app.simulation):
            self.invitation_button.draw(app.window, app.small_font)
        if self.pending_offer:
            self.accept_offer_button.draw(app.window, app.small_font, enabled=not app.simulation.paused)
            self.decline_offer_button.draw(app.window, app.small_font, enabled=not app.simulation.paused)
        elif closing:
            self.done_button.draw(app.window, app.font)
        else:
            self.prompt_input.draw(app.window, app.font)
            self.send_button.draw(app.window, app.font, enabled=not self.selected_npc.request_pending and not app.simulation.paused)
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
        draw_wrapped_text(app.window, self.call_status or "Accept starts a conversation. Decline releases both of you.",
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
        self.original_toggle.label = "Original: " + ("On" if app.original_dialogue else "Off")
        self.original_toggle.draw(app.window, app.small_font)

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

    def pending_invitation(self, world):
        interaction = world.player_interaction
        return next((o for o in reversed(world.player_invitations) if o["status"] == "offered"
            and interaction and o["interaction_id"] == interaction["id"]), None)

    def open_location_menu(self, purpose, choices):
        self.location_menu = dict(purpose=purpose, choices=choices)
        self.location_page = 0

    def location_buttons(self):
        from constants import (LOCATION_MENU_RECT, LOCATION_BUTTON_HEIGHT, LOCATION_BUTTON_GAP,
            LOCATION_BUTTON_WIDTH, LOCATION_BUTTON_INSET, LOCATION_BUTTONS_PER_PAGE)
        x, y = LOCATION_MENU_RECT[:2]
        start = self.location_page * LOCATION_BUTTONS_PER_PAGE
        return [(Button((x + LOCATION_BUTTON_INSET[0], y + LOCATION_BUTTON_INSET[1] + row * LOCATION_BUTTON_GAP,
            LOCATION_BUTTON_WIDTH, LOCATION_BUTTON_HEIGHT), label), value)
            for row, (label, value) in enumerate(self.location_menu["choices"][start:start + LOCATION_BUTTONS_PER_PAGE])]

    def handle_location_event(self, app, event):
        from constants import LOCATION_BUTTONS_RECT, LOCATION_NEXT_RECT, LOCATION_CANCEL_RECT, LOCATION_BUTTONS_PER_PAGE
        if Button(LOCATION_CANCEL_RECT, "Cancel").clicked(event) or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.location_menu = None
            return
        if Button(LOCATION_BUTTONS_RECT, "Previous").clicked(event):
            self.location_page = max(0, self.location_page - 1)
        if Button(LOCATION_NEXT_RECT, "Next").clicked(event):
            self.location_page = min((len(self.location_menu["choices"]) - 1) // LOCATION_BUTTONS_PER_PAGE, self.location_page + 1)
        for button, value in self.location_buttons():
            if button.clicked(event):
                if self.location_menu["purpose"] == "Observe":
                    self.close_dialogue()
                    self.location = value
                    from locations import map_size
                    width, height = map_size(value)
                    self.camera_offset[:] = ((WORLD_WIDTH - width * self.camera_zoom) / 2,
                                             (WORLD_HEIGHT - height * self.camera_zoom) / 2)
                elif value != app.simulation.player.location:
                    from locations import start_journey
                    from constants import ELEVATOR_ARRIVAL
                    start_journey(app.simulation.player, value, ELEVATOR_ARRIVAL, purpose="elevator", fast=True)
                self.location_menu = None
                return

    def draw_location_menu(self, app):
        from constants import LOCATION_MENU_RECT, LOCATION_BUTTONS_RECT, LOCATION_NEXT_RECT, LOCATION_CANCEL_RECT, LOCATION_TITLE_INSET
        overlay = pygame.Surface(WINDOW_SIZE, pygame.SRCALPHA)
        overlay.fill(MODAL_OVERLAY_COLOR)
        app.window.blit(overlay, (0, 0))
        pygame.draw.rect(app.window, PAPER, LOCATION_MENU_RECT)
        app.window.blit(app.heading_font.render(self.location_menu["purpose"], True, INK),
                        (LOCATION_MENU_RECT[0] + LOCATION_TITLE_INSET[0], LOCATION_MENU_RECT[1] + LOCATION_TITLE_INSET[1]))
        for button, _ in self.location_buttons():
            button.draw(app.window, app.small_font)
        for rect, label in ((LOCATION_BUTTONS_RECT, "Previous"), (LOCATION_NEXT_RECT, "Next"), (LOCATION_CANCEL_RECT, "Cancel")):
            Button(rect, label).draw(app.window, app.small_font)

    def update_physical(self, app, dt):
        from player import active_player, move_player
        from locations import map_size
        from constants import GARDEN_LOCATION
        world = app.simulation
        self.app = app
        if not active_player(world) and self.inventory.visible and self.inventory.owner is world.player:
            self.inventory.close()
        world.player_menu_blocked = bool(self.location_menu or self.inventory.visible or self.apartment_open
            or self.call_choice or self.confirm_menu or app.screen != "game")
        if self.confirm_menu:
            return
        if active_player(world):
            self.location = world.player.location
            interaction = world.player_interaction
            if interaction and interaction["kind"] == "resident":
                if interaction["stage"] == "prompt" and not self.call_choice:
                    self.open_call(app, world.npc_by_id(interaction["npc_id"]))
                elif interaction["stage"] == "conversation" and not self.dialogue_open:
                    app.open_player_dialogue(world.npc_by_id(interaction["npc_id"]))
            elif interaction and interaction["kind"] == "snake" and not self.dialogue_open:
                app.open_player_dialogue(world.snake)
            elif interaction and interaction["kind"] == "door" and not world.player.journey:
                target = interaction["target"]
                world.player_interaction = None
                app.use_door(target)
            blocking = (world.player_menu_blocked or self.dialogue_open or self.prompt_input.active
                        or pygame.display.get_init() and pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_ALT))
            if not blocking and app.screen == "game" and dt > 0 and pygame.display.get_init():
                keys = pygame.key.get_pressed()
                move_player(world, keys[pygame.K_d] - keys[pygame.K_a], keys[pygame.K_s] - keys[pygame.K_w], dt,
                            running=bool(keys[pygame.K_SPACE]))
            width, height = map_size(self.location)
            # Dialogue clips the world view but must not move the camera anchor.
            # Follow freely in the garden, even when it fits inside the window.
            base = ((WINDOW_SIZE[0] - WORLD_WIDTH) / 2, (WINDOW_SIZE[1] - WORLD_HEIGHT) / 2)
            desired = []
            for extent, view, position in ((width, WINDOW_SIZE[0], world.player.x), (height, WINDOW_SIZE[1], world.player.y)):
                scaled = extent * self.camera_zoom
                if self.location == GARDEN_LOCATION:
                    desired.append(view / 2 - position * self.camera_zoom)
                else:
                    desired.append((view - scaled) / 2 if scaled <= view else min(0, max(view - scaled, view / 2 - position * self.camera_zoom)))
            self.camera_offset[:] = [desired[i] - base[i] for i in (0, 1)]
        elif self.selected_npc and not self.viewed_conversation:
            self.close_dialogue()

    def request_is_current(self, app):
        return (self.viewed_conversation is not None
                and app.conversation_requests.get(self.viewed_conversation.id) is self.viewed_conversation)
