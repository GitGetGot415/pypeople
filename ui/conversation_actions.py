import pygame
from constants import (
    INK, MUTED, PAPER, BORDER_RADIUS, PLAYER_ACTION_PICKER_RECT, PLAYER_ACTION_PICKER_ROWS,
    PLAYER_ACTION_PICKER_ROW_HEIGHT, PLAYER_ACTION_PICKER_INSET, PLAYER_ACTION_PICKER_TITLE_HEIGHT,
    PLAYER_ACTION_PICKER_FOOTER_HEIGHT, PLAYER_ACTION_PICKER_BUTTON_WIDTH,
    PLAYER_ACTION_PICKER_ROW_BUTTON_HEIGHT, PLAYER_ACTION_PICKER_FOOTER_BUTTON_HEIGHT,
    PLAYER_ACTION_PICKER_BORDER_WIDTH,
)
from ui.widgets.button import Button
from ui.text import draw_wrapped_text, wrap_text


class ConversationActions:
    def __init__(self, game):
        self.game = game
        self.menu = None

    def available(self, app):
        from simulation.transfers import validate_transfer
        actor = self.game.dialogue.actor
        kill = ("Kill", "kill", app.simulation.player_killing_unavailable(actor) or "")
        reason = app.conversations.input_unavailable(actor, active=app.screen == "game")
        if self.game.dialogue.conversation is not None:
            reason = "This person is busy in a conversation."
        elif not self.game.player_response_ready:
            reason = "Finish reading first"
        elif self.game.pending_offer:
            reason = "Respond to the gift first"
        request_reason = reason or ("Unavailable for this character" if actor is app.simulation.snake else "")
        apple_reason = reason
        if not apple_reason:
            try:
                validate_transfer(app.simulation, app.simulation.player, actor, "fruit", 1)
            except ValueError as exc:
                apple_reason = str(exc)
        actions = [("Request action", "request", request_reason),
                   ("Give apple", "apple", apple_reason)]
        if self.game.pending_invitation(app.simulation):
            actions.append(("Accept invitation", "invitation", reason))
        actions.append(kill)
        return actions

    def layout(self, app):
        if self.menu == "confirm_kill":
            rect = pygame.Rect(184, 544, 680, 150)
            buttons = [(Button((200, 620, 170, 32), "Confirm kill"), "confirm", "", 32),
                       (Button((380, 620, 170, 32), "Cancel"), "cancel", "", 32)]
            return rect, buttons
        actions = self.available(app)
        reason_width = 774
        heights = [max(30, len(wrap_text(reason, app.small_font, reason_width)) * app.small_font.get_linesize())
                   if reason else 30 for _, _, reason in actions]
        rect = pygame.Rect(184, 694 - (sum(heights) + 3 * (len(actions) - 1) + 20), 1068,
                           sum(heights) + 3 * (len(actions) - 1) + 20)
        y = rect.top + 10
        buttons = []
        for (label, value, reason), height in zip(actions, heights):
            buttons.append((Button((200, y, 250, 30), label), value, reason, height))
            y += height + 3
        return rect, buttons

    def handle_event(self, app, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.menu = None
            self.game.focus_player_reply(app)
            return
        menu_rect, buttons = self.layout(app)
        for button, value, reason, height in buttons:
            if not button.clicked(event, enabled=not reason):
                continue
            if value == "kill":
                self.menu = "confirm_kill"
            elif value == "cancel":
                self.menu = None
                self.game.focus_player_reply(app)
            elif value == "request":
                self.menu = None
                self.game.dialogue.picker = dict(command={}, page=0, focus=True)
            elif value == "confirm":
                self.menu = None
                if not app.simulation.player_killing_unavailable(self.game.dialogue.actor):
                    app.sync_conversation_view()
                    app.conversations.kill_resident(self.game.dialogue.actor)
                    app.present_conversation_events()
            elif value == "apple":
                self.menu = None
                if self.game.can_give_apple(app):
                    app.sync_conversation_view()
                    try:
                        app.conversations.give_player_apple(self.game.dialogue.actor)
                    except ValueError as exc:
                        self.game.show_save_message(str(exc))
                    app.present_conversation_events()
                self.game.focus_player_reply(app)
            elif value == "invitation":
                self.menu = None
                offer = self.game.pending_invitation(app.simulation)
                if offer:
                    app.sync_conversation_view()
                    app.conversations.accept_player_invitation(offer.id)
                    app.present_conversation_events()
                self.game.focus_player_reply(app)
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not menu_rect.collidepoint(event.pos):
                self.menu = None
                self.game.focus_player_reply(app)

    def draw(self, app):
        rect, buttons = self.layout(app)
        pygame.draw.rect(app.window, PAPER, rect)
        pygame.draw.rect(app.window, INK, rect, width=2)
        if self.menu == "confirm_kill":
            draw_wrapped_text(app.window, f"Kill {self.game.dialogue.actor.name}?", app.font, INK,
                              (200, 565, 630, 40))
        for button, value, reason, height in buttons:
            button.draw(app.window, app.small_font, enabled=not reason)
            if reason:
                draw_wrapped_text(app.window, reason, app.small_font, MUTED,
                                  (462, button.rect.y, 774, height))

    def picker_options(self, app):
        from simulation.player_commands import command_options
        command = self.game.dialogue.picker["command"]
        if command.get("item") == "fruit":
            quantity = command["quantity"]
            stock = self.game.dialogue.actor.inventory.get("fruit", 0)
            return "Choose apple quantity", [
                ("Fewer apples", "less", "" if quantity > 1 else "Minimum one"),
                ("More apples", "more", "" if quantity < stock else "All carried apples"),
                (f"Request {quantity} apple" + ("s" if quantity != 1 else ""), command.copy(),
                 "Not enough carried apples" if quantity > stock else "")]
        title = "Choose an action" if not command else "Choose a recipient" if not command.get("target") else "Choose an item"
        return title, command_options(app.simulation, self.game.dialogue.actor, command.get("action"), command.get("target"))

    def picker_footer(self):
        rect = pygame.Rect(PLAYER_ACTION_PICKER_RECT)
        inset, width = PLAYER_ACTION_PICKER_INSET, PLAYER_ACTION_PICKER_BUTTON_WIDTH
        return [Button((rect.x + inset + i * (width + inset), rect.bottom - PLAYER_ACTION_PICKER_FOOTER_HEIGHT,
                        width, PLAYER_ACTION_PICKER_FOOTER_BUTTON_HEIGHT), label)
                for i, label in enumerate(("Back", "Previous", "Next", "Cancel"))]

    def handle_picker_event(self, app, event):
        picker = self.game.dialogue.picker
        back, previous, following, cancel = self.picker_footer()
        if cancel.clicked(event) or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            focus = picker["focus"]
            self.game.dialogue.picker = None
            if focus:
                self.game.focus_player_reply(app)
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
        reason = app.conversations.input_unavailable(self.game.dialogue.actor, active=app.screen == "game")
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
                self.game.dialogue.picker = None
                app.sync_conversation_view()
                app.conversations.submit(self.game.dialogue.actor, "", command=value, active=app.screen == "game")
                app.present_conversation_events()
            else:
                picker["command"], picker["page"], picker["buttons"] = value, 0, []
            return

    def draw_picker(self, app):
        picker = self.game.dialogue.picker
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
        reason = app.conversations.input_unavailable(self.game.dialogue.actor, active=app.screen == "game")
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
