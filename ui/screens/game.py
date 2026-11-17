from ui.conversation_actions import ConversationActions
import pygame
from simulation.player import PlayerInteraction
from simulation.maps import active_player
from constants import (
    DIALOGUE_PANEL_HEIGHT, DIALOGUE_READING_Y, DIALOGUE_READING_HEIGHT,
    FAILED_CONVERSATION_SELECTED_COLOR, INK, MUTED, ORIGINAL_DIALOGUE_PAPER, PAPER, MAX_LLM_REQUESTS,
    WINDOW_SIZE, WORLD_HEIGHT, WORLD_WIDTH, APARTMENT_INSPECTOR_CLOSE_RECT, APARTMENT_INSPECTOR_RECT,
    APARTMENT_INSPECTOR_ROW_HEIGHT, APARTMENT_INSPECTOR_CONTENT_INSET, APARTMENT_INSPECTOR_CONTENT_TOP,
    APARTMENT_INSPECTOR_TITLE_OFFSET, INVENTORY_BUTTON_RECT, GAME_PAUSE_RECT,
    GAME_CLOCK_RECT, GAME_SAVE_RECT, GAME_STATS_RECT, GAME_DAY_POSITION, GAME_MENU_RECT,
    GAME_TOOLBAR_PADDING, GAME_DAY_BACKGROUND, GAME_DAY_TEXT_COLOR, GAME_DAY_PADDING,
    GAME_TOOLBAR_BORDER_COLOR, GAME_TOOLBAR_BORDER_WIDTH, GIFT_ACCEPT_RECT, GIFT_DECLINE_RECT,
    DIALOGUE_ORIGINAL_TOGGLE_RECT, BORDER_RADIUS, POPUP_RECT, POPUP_TITLE_Y,
    GAME_MENU_CONFIRM_MESSAGE_RECT, GAME_MENU_KEEP_PLAYING_RECT, GAME_MENU_LEAVE_RECT,
    PLAYER_ROOM_LOCATION, ROOM_BED_RECT, BED_POPUP_MESSAGE_RECT,
    BED_ADVANCE_RECT, BED_CANCEL_RECT, GAME_PORTRAIT_BACKGROUND, GAME_PORTRAIT_PADDING,
    GAME_PORTRAIT_BORDER_WIDTH,
)
from dialogue.emotions import portrait_expression
from dialogue.history import uses_original_dialogue
from ui.dialogue_view import DialogueView
from ui.widgets.button import Button
from ui.request_status import RequestStatus
from ui.text import draw_modal, draw_wrapped_text, wrap_text, scrollbar_rects, ScrollbarDrag
from ui.widgets.text_input import TextInput
from ui.rendering.world import CHARACTER_SCALE, draw_world, generation_bubbles
from ui.rendering.character_sprite import draw_actor_portrait
from ui.inventory import InventoryPanel


class GameScreen:
    def __init__(self, max_requests=MAX_LLM_REQUESTS):
        from constants import HARVEST_BUTTON_RECT, HARVEST_AMOUNT_RECT
        self.actions = ConversationActions(self)
        self.app = None
        self.location = "garden"
        self.end_button = Button((1192, 504, 64, 32), "End")
        self.harvest_button = Button(HARVEST_BUTTON_RECT, "Harvest apples")
        self.harvest_amount = TextInput(HARVEST_AMOUNT_RECT, max_length=6, placeholder="1")
        self.harvest_amount.set_value("1")
        self.actions_button = Button((184, 694, 112, 24), "Actions v")
        self.menu_button = Button(GAME_MENU_RECT, "Menu", icon="menu", show_label=True)
        self.confirm_menu = False
        self.menu_restore_focus = False
        self.keep_playing_button = Button(GAME_MENU_KEEP_PLAYING_RECT, "Keep playing")
        self.leave_game_button = Button(GAME_MENU_LEAVE_RECT, "Return to menu")
        self.stats_button = Button(GAME_STATS_RECT, "Stats", icon="stats", show_label=True)
        self.save_button = Button(GAME_SAVE_RECT, "Save", icon="save", show_label=True)
        self.inventory_button = Button(INVENTORY_BUTTON_RECT, "Inventory", icon="inventory")
        self.inventory = InventoryPanel()
        self.accept_offer_button = Button(GIFT_ACCEPT_RECT, "Accept gift")
        self.decline_offer_button = Button(GIFT_DECLINE_RECT, "Decline")
        self.pause_button = Button(GAME_PAUSE_RECT, "Pause", icon="pause")
        self.arrival_retry = Button((476, 72, 120, 32), "Retry arrival")
        self.clock_button = Button(GAME_CLOCK_RECT, "Next day", icon="clock")
        self.day_popup = None  # None, "bed" or "clock"
        self.bed_advance_button = Button(BED_ADVANCE_RECT, "")
        self.bed_cancel_button = Button(BED_CANCEL_RECT, "Cancel")
        self.save_message = ""
        self.save_message_remaining = 0
        self.send_button = Button((1136, 644, 120, 44), "Send")
        self.done_button = Button((1136, 644, 120, 44), "Done")
        self.next_button = Button((1176, 552, 80, 32), "Next")
        self.back_button = Button((1092, 552, 80, 32), "Back")
        self.subjects_button = Button((976, 552, 112, 32), "Subjects")
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
        self.mouse_follow = False
        self.request_status = RequestStatus(max_requests)
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
        interaction = self.app.simulation.player_interaction if self.app and self.app.simulation else None
        record = (self.app.conversations.record if self.app and self.app.conversations and interaction
                  and interaction.id == self.dialogue.interaction_id else None)
        return record or self.dialogue.record

    def reset(self):
        self.apartment_scrollbar.cancel()
        self.day_popup = None
        self.confirm_menu = False
        self.menu_restore_focus = False
        self.inventory.close()
        self.apartment_open = False
        self.apartment_scroll = 0
        self.close_dialogue()
        self.prompt_input.set_value("")
        self.camera_offset[:] = (0.0, 0.0)
        self.camera_zoom = 1.0
        self.pan_start = None
        self.pan_last = None
        self.panning = False
        self.mouse_follow = False
        self.save_message = ""
        self.save_message_remaining = 0
        self.request_status = RequestStatus(self.request_status.max_requests)

    def show_save_message(self, message):
        self.save_message = message
        self.save_message_remaining = 8

    def close_dialogue(self, *, terminate=True, notify=True):
        self.dialogue_scrollbar.cancel()
        self.dialogue.scroll_limit = 0
        if notify and self.app and self.app.conversations:
            self.app.sync_conversation_view()
            self.app.conversations.view_closed(terminate=terminate)
            self.app.present_conversation_events()
        self.dialogue.reset()
        self.actions.menu = None
        self.prompt_input.blur()

    def set_reply(self, text):
        normalized = " ".join(text.split())
        self.dialogue.status = "".join(char for char in normalized if char.isprintable())
        self.dialogue.scroll = 0
        self.prompt_input.blur()
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
        from simulation.maps import active_player
        self.clock_layout(app)
        world = app.simulation
        if (event.type == pygame.WINDOWFOCUSLOST
                or event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP) and event.button == 1):
            self.mouse_follow = False
        if self.dialogue.open and self.dialogue.conversation is None and self.dialogue.actor is not None:
            self.update_player_pages(app)
        if self.day_popup:
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
        if self.dialogue.picker is not None:
            self.actions.handle_picker_event(app, event)
            return
        if self.actions.menu is not None:
            self.actions.handle_event(app, event)
            return
        if (self.dialogue.open and active_player(world)
                and self.actions_button.clicked(event)):
            self.actions.menu = "actions"
            self.prompt_input.blur()
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
        if active_player(world) and self.inventory_button.clicked(event):
            self.pan_start = self.pan_last = None
            self.prompt_input.blur()
            self.inventory.open(app.simulation.player)
            return
        if self.pending_offer and self.dialogue.open and self.player_response_ready:
            if self.accept_offer_button.clicked(event, enabled=not app.simulation.paused):
                app.sync_conversation_view()
                app.conversations.accept_gift_offer()
                app.present_conversation_events()
                return
            if self.decline_offer_button.clicked(event, enabled=not app.simulation.paused):
                app.sync_conversation_view()
                app.conversations.decline_gift_offer()
                app.present_conversation_events()
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
        if self.request_status.handle_event(app, event):
            return
        world_buttons = (self.menu_button, self.stats_button, self.save_button, self.pause_button,
                         self.inventory_button if active_player(world) else self.clock_button)
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
                if active_player(world):
                    self.select_at(app, event.pos)
                    self.mouse_follow = bool(world.player.journey and world.player.journey.purpose == "walk")
                    self.pan_start = self.pan_last = None
                    return
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
                return
            if self.back_button.clicked(event, enabled=self.dialogue.index > 0):
                self.dialogue.move(-1)
                self.prompt_input.blur()
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
                app.sync_conversation_view()
                app.conversations.submit(app.game.dialogue.actor, app.game.prompt_input.text, active=app.screen == "game")
                app.present_conversation_events()
                self.focus_player_reply(app)
            elif self.prompt_input.handle_event(event):
                app.sync_conversation_view()
                app.conversations.submit(app.game.dialogue.actor, app.game.prompt_input.text, active=app.screen == "game")
                app.present_conversation_events()

    @property
    def player_response_ready(self):
        return self.dialogue.conversation is None and self.dialogue.final_page

    def update_player_pages(self, app):
        changed = self.dialogue.refresh(self.player_record, app.font, self.reply_rect.width - 12,
            original=app.original_dialogue, condensed=getattr(app, "condensed_dialogue", False), player_reply=True)
        if changed:
            self.prompt_input.blur()
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
        from simulation.maps import active_player
        if self.pause_button.clicked(event):
            app.simulation.paused = not app.simulation.paused
            self.pan_start = self.pan_last = None
            self.panning = False
            return True
        if not active_player(app.simulation) and self.clock_button.clicked(event, enabled=not app.simulation.paused):
            self.day_popup = "clock"
            self.pan_start = self.pan_last = None
            self.panning = False
            return True
        return False

    def can_use_bed(self, world):
        from simulation.maps import active_player
        return (active_player(world) and self.location == world.player.location == PLAYER_ROOM_LOCATION
                and not world.player_interaction and not world.player.journey
                and not (world.visit and world.visit.stage == "travel"))

    def day_popup_valid(self, world):
        from simulation.maps import active_player
        return self.can_use_bed(world) if self.day_popup == "bed" else not active_player(world)

    def can_advance_from_bed(self, world):
        return self.day_popup_valid(world) and not world.paused

    def handle_bed_event(self, app, event):
        if self.bed_cancel_button.clicked(event) or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.day_popup = None
        elif self.bed_advance_button.clicked(event, enabled=self.can_advance_from_bed(app.simulation)):
            self.day_popup = None
            app.conversations.next_day()

    def draw_bed_popup(self, app):
        if not self.day_popup:
            return
        world = app.simulation
        title = "Rest in your bed" if self.day_popup == "bed" else "Skip to the next day?"
        draw_modal(app.window, POPUP_RECT, title, app.heading_font, POPUP_TITLE_Y)
        message = ("Resume the game before passing time." if world.paused else
                   "Advance to the next day? All conversations end immediately. A new resident arrives if there is room.")
        draw_wrapped_text(app.window, message, app.font, INK, BED_POPUP_MESSAGE_RECT)
        self.bed_advance_button.label = f"Advance to {world.day + 1}"
        self.bed_advance_button.draw(app.window, app.small_font, enabled=self.can_advance_from_bed(world))
        self.bed_cancel_button.draw(app.window, app.small_font)

    def open_menu_confirmation(self):
        self.confirm_menu = True
        self.menu_restore_focus = self.prompt_input.active
        self.prompt_input.blur()
        self.pan_start = self.pan_last = None
        self.panning = False

    def draw_menu_confirmation(self, app):
        if not self.confirm_menu:
            return
        draw_modal(app.window, POPUP_RECT, "Return to main menu?", app.heading_font, POPUP_TITLE_Y)
        draw_wrapped_text(app.window, "Progress since your last save will be lost.", app.font, INK,
                          GAME_MENU_CONFIRM_MESSAGE_RECT)
        self.keep_playing_button.draw(app.window, app.small_font)
        self.leave_game_button.draw(app.window, app.small_font)

    def clock_layout(self, app):
        day_rect = pygame.Rect((0, 0), app.small_font.size(f"Day {app.simulation.day}"))
        day_rect.topleft = GAME_DAY_POSITION
        rect = day_rect.inflate(*GAME_DAY_PADDING).unionall([
            pygame.Rect(GAME_CLOCK_RECT), pygame.Rect(GAME_PAUSE_RECT)])
        offset = (WINDOW_SIZE[0] // 2 - rect.centerx, 0)
        self.clock_button.rect = pygame.Rect(GAME_CLOCK_RECT).move(offset)
        self.inventory_button.rect = self.clock_button.rect.copy()
        self.pause_button.rect = pygame.Rect(GAME_PAUSE_RECT).move(offset)
        return (day_rect.move(offset),
                rect.move(offset).inflate(2 * GAME_TOOLBAR_PADDING, 2 * GAME_TOOLBAR_PADDING))

    def draw_toolbar_background(self, surface, rect):
        pygame.draw.rect(surface, PAPER, rect, border_radius=BORDER_RADIUS)
        pygame.draw.rect(surface, GAME_TOOLBAR_BORDER_COLOR, rect,
                         width=GAME_TOOLBAR_BORDER_WIDTH, border_radius=BORDER_RADIUS)

    def toolbar_rect(self):
        rect = self.menu_button.rect.unionall([self.save_button.rect, self.stats_button.rect])
        return rect.inflate(2 * GAME_TOOLBAR_PADDING, 2 * GAME_TOOLBAR_PADDING)

    def draw_clock(self, app):
        from simulation.maps import active_player
        day_rect, toolbar_rect = self.clock_layout(app)
        self.draw_toolbar_background(app.window, toolbar_rect)
        paused = app.simulation.paused
        self.pause_button.label = "Resume" if paused else "Pause"
        self.pause_button.icon = "play" if paused else "pause"
        self.pause_button.draw(app.window, app.small_font)
        if active_player(app.simulation):
            self.inventory_button.draw(app.window, app.small_font)
        else:
            self.clock_button.draw(app.window, app.small_font, enabled=not paused)
        day_label = app.small_font.render(f"Day {app.simulation.day}", True, GAME_DAY_TEXT_COLOR)
        pygame.draw.rect(app.window, GAME_DAY_BACKGROUND, day_rect.inflate(*GAME_DAY_PADDING),
                         border_radius=BORDER_RADIUS)
        app.window.blit(day_label, day_rect)

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
        from simulation.maps import active_player
        from simulation.player import can_claim, can_interrupt_approach, start_approach
        from simulation.maps import clamp_point, targets
        from simulation.locations import start_journey
        world = app.simulation
        if active_player(world) and (world.player_interaction
                                     or world.player.journey and (world.paused or world.player.journey.purpose != "walk")
                                     or world.visit and world.visit.stage == "travel"):
            return
        if active_player(world) and world.player.journey:
            world.player.journey = None
        point = self.screen_to_world(screen_pos)
        if self.can_use_bed(world) and pygame.Rect(ROOM_BED_RECT).collidepoint(point):
            self.close_dialogue()
            self.day_popup = "bed"
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
                    self.close_dialogue()
                    start_journey(world.player, self.location, target["point"], purpose="door", fast=True)
                    world.player_interaction = PlayerInteraction(id="door", kind="door", stage="travel", target=target,
                        location=self.location, npc_id=None, record_id=None)
                else:
                    app.use_door(target)
                return
        if active_player(world) and self.location == "garden" and hit_actor is None and abs(point[0] - world.tree.x) < 70 and abs(point[1] - world.tree.y) < 100:
            self.close_dialogue()
            start_journey(world.player, "garden", world.tree.interaction_point, purpose="tree", fast=True)
            return
        npc = hit_actor
        if npc is None:
            self.close_dialogue()
            if active_player(world) and not world.paused and can_claim(world):
                start_journey(world.player, self.location, clamp_point(self.location, point), purpose="walk")
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
        from simulation.player import can_claim
        return (not world.paused and self.location == "garden" and can_claim(world)
                and not self.dialogue.open and math.dist((world.player.x, world.player.y),
                                                        world.tree.interaction_point) <= HARVEST_DISTANCE)

    def can_give_apple(self, app):
        from simulation.transfers import validate_transfer
        if (not self.dialogue.open or not self.player_response_ready or self.pending_offer
                or app.conversations.input_unavailable(self.dialogue.actor, active=app.screen == "game")):
            return False
        try:
            validate_transfer(app.simulation, app.simulation.player, self.dialogue.actor, "fruit", 1)
        except ValueError:
            return False
        return True

    def draw(self, app):
        from simulation.maps import active_player
        draw_world(app.window, app.npcs, app.tree, app.small_font, app.character.renderer,
                   self.view_offset, zoom=self.camera_zoom,
                   viewport=self.world_viewport, snake=app.simulation.snake, world=app.simulation, location=self.location,
                   speaking=generation_bubbles(app.conversation_requests, app.requests.statuses()))
        if self.can_harvest(app.simulation):
            self.harvest_button.draw(app.window, app.small_font)
            self.harvest_amount.draw(app.window, app.small_font)
        self.draw_toolbar_background(app.window, self.toolbar_rect())
        self.menu_button.draw(app.window, app.small_font)
        self.stats_button.draw(app.window, app.small_font)
        self.save_button.draw(app.window, app.small_font)
        self.draw_clock(app)
        self.request_status.draw(app)
        pending = app.simulation.pending_arrival_days
        error = app.simulation.arrival_error
        if pending or error:
            color = INK
            label = f"Arrivals waiting: {', '.join(str(d) for d in pending)}" if pending else "Next arrival"
            if error:
                label += f" / {error}"
                self.arrival_retry.draw(app.window, app.small_font, enabled=not app.simulation.paused)
            else:
                label += " / arriving"
            draw_wrapped_text(app.window, label, app.small_font, color, (476, 110, 550, 64))
        if self.save_message_remaining:
            color = PAPER if self.location != "garden" else INK
            draw_wrapped_text(app.window, self.save_message, app.small_font, color, (24, 480, 1232, 44))

        if self.dialogue.open and app.simulation.player_interaction:
            self.end_button.draw(app.window, app.small_font)
        if not self.dialogue.open:
            if self.apartment_open:
                self.draw_apartments(app)
            from constants import LOCATION_STATUS_POSITION
            app.window.blit(app.small_font.render(f"{app.simulation.mode.title()} / {self.location}", True, PAPER if self.location != "garden" else INK), LOCATION_STATUS_POSITION)
            return
        if self.dialogue.conversation is not None:
            self.draw_conversation(app)
            self.draw_dialogue_actions(app)
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
            status = "Reply failed."
        title = app.font.render(turn.speaker if turn else "Request status" if page else self.dialogue.actor.name, True, INK)
        title_rect = title.get_rect(topleft=(self.reply_rect.x, 557))
        app.window.blit(title, title_rect)
        status_right = self.back_button.rect.left
        draw_wrapped_text(app.window, status, app.small_font, MUTED,
                          (title_rect.right + 16, 558, max(0, status_right - title_rect.right - 24), 24))
        if len(self.dialogue.pages) > 1:
            self.back_button.draw(app.window, app.small_font, enabled=self.dialogue.index > 0)
        closing = self.dialogue.actor.player_reply.closing
        self.next_button.label = "Next"
        if len(self.dialogue.pages) > 1:
            self.next_button.draw(app.window, app.small_font, enabled=not responding)
        rect = self.reply_rect if responding else pygame.Rect(self.reply_rect.x, DIALOGUE_READING_Y,
                                                             self.reply_rect.width, DIALOGUE_READING_HEIGHT)
        self.draw_dialogue_lines(app.window, wrap_text(text, app.font, rect.width - 12), app.font, rect,
                                 color=INK if turn else MUTED)
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
                              FAILED_CONVERSATION_SELECTED_COLOR, (304, 696, self.reply_rect.width - 210, 24))
        elif responding and not closing and self.prompt_input.error:
            app.window.blit(app.small_font.render(self.prompt_input.error, True, MUTED), (304, 696))
        if responding and not closing:
            count = app.small_font.render(f"{len(self.prompt_input.text)}/300", True, MUTED)
            app.window.blit(count, count.get_rect(topright=(1256, 696)))
        elif not responding:
            progress = app.small_font.render(f"{self.dialogue.index + 1}/{len(self.dialogue.pages)}", True, MUTED)
            app.window.blit(progress, progress.get_rect(topright=(1256, 696)))
        self.draw_dialogue_actions(app)
        if self.dialogue.picker is not None:
            self.actions.draw_picker(app)

    def draw_dialogue_actions(self, app):
        if active_player(app.simulation):
            self.actions_button.draw(app.window, app.small_font)
            if self.actions.menu is not None:
                self.actions.draw(app)

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
        rect = pygame.Rect(APARTMENT_INSPECTOR_RECT)
        draw_modal(app.window, rect)
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
        title_right = self.subjects_button.rect.left if conversation.record.intended_subjects else self.back_button.rect.left
        draw_wrapped_text(app.window, title, app.font, INK, (self.reply_rect.x, 557, title_right - self.reply_rect.x - 16, 30))
        if conversation.record.intended_subjects:
            self.subjects_button.draw(app.window, app.small_font)
        self.back_button.draw(app.window, app.small_font, enabled=self.dialogue.index > 0)
        last_page = bool(self.dialogue.pages) and self.dialogue.index == len(self.dialogue.pages) - 1
        self.next_button.label = "Done" if last_page else "Next"
        can_finish = not (last_page and conversation.active and app.simulation.paused)
        self.next_button.draw(app.window, app.small_font, enabled=bool(self.dialogue.pages) and can_finish)
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
            status_left = self.actions_button.rect.right + 8 if active_player(app.simulation) else rect.x
            draw_wrapped_text(app.window, status, app.small_font, status_color,
                              (status_left, 696, rect.right - status_left - 80, 24))
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
        from simulation.apartments import pending_invitation
        return pending_invitation(world)

    def observe_location(self, location):
        from simulation.maps import map_size
        self.close_dialogue()
        self.location = location
        width, height = map_size(location)
        self.camera_offset[:] = ((WORLD_WIDTH - width * self.camera_zoom) / 2,
                                 (WORLD_HEIGHT - height * self.camera_zoom) / 2)

    def update_physical(self, app, dt):
        from simulation.maps import active_player
        from simulation.player import move_player
        from simulation.maps import clamp_point, map_size
        from simulation.locations import start_journey
        from constants import GARDEN_LOCATION
        world = app.simulation
        self.app = app
        world.player_running = False
        if not active_player(world) and self.inventory.visible and self.inventory.owner is world.player:
            self.inventory.close()
        if self.day_popup and not self.day_popup_valid(world):
            self.day_popup = None
        world.player_menu_blocked = bool(self.inventory.visible or self.apartment_open
            or self.confirm_menu or self.day_popup or app.screen != "game")
        if (world.paused or world.player_menu_blocked or not active_player(world)
                or not pygame.display.get_init() or not pygame.mouse.get_pressed()[0]):
            self.mouse_follow = False
        if self.confirm_menu:
            return
        if active_player(world):
            self.location = world.player.location
            interaction = world.player_interaction
            if interaction and interaction.kind in ("resident", "snake"):
                if (self.dialogue.open and self.dialogue.conversation is None
                        and self.dialogue.interaction_id != interaction.id):
                    self.close_dialogue()
                if interaction.stage == "conversation" and not self.dialogue.open:
                    app.sync_conversation_view()
                    app.conversations.open_player_dialogue(world.actor_by_id(interaction.npc_id))
                    app.present_conversation_events()
            elif interaction and interaction.kind == "door" and not world.player.journey:
                target = interaction.target
                world.player_interaction = None
                app.use_door(target)
            blocking = (world.player_menu_blocked or self.dialogue.open or self.prompt_input.active
                        or pygame.display.get_init() and pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_ALT))
            if blocking or world.player_interaction:
                self.mouse_follow = False
            if not blocking and app.screen == "game" and dt > 0 and pygame.display.get_init():
                keys = pygame.key.get_pressed()
                world.player_running = bool(keys[pygame.K_SPACE])
                dx, dy = keys[pygame.K_d] - keys[pygame.K_a], keys[pygame.K_s] - keys[pygame.K_w]
                if dx or dy:
                    self.mouse_follow = False
                if self.mouse_follow:
                    pos = pygame.mouse.get_pos()
                    journey = world.player.journey
                    if journey and journey.purpose != "walk":
                        if journey.purpose != "doorway":
                            self.mouse_follow = False
                    elif (self.world_viewport.collidepoint(pos) and not any(rect.collidepoint(pos) for rect in (
                            self.toolbar_rect(), self.clock_layout(app)[1], self.request_status.rect))):
                        point = clamp_point(self.location, self.screen_to_world(pos))
                        start_journey(world.player, self.location, point, purpose="walk")
                move_player(world, dx, dy, dt, running=world.player_running)
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
