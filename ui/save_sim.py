import pygame

from .button import Button
from .text import draw_centered_text, draw_wrapped_text
from .text_input import TextInput
from .theme import INK, MUTED, PAPER


class SaveSimScreen:
    def __init__(self):
        self.name_input = TextInput((420, 315, 440, 48), max_length=80, placeholder="Simulation name")
        self.save_button = Button((655, 435, 205, 44), "Save")
        self.cancel_button = Button((420, 435, 205, 44), "Cancel")
        self.message = ""

    def open(self, name):
        self.message = ""
        self.name_input.set_value(name)
        self.name_input.focus()
        self.name_input.selected_all = True

    def handle_event(self, app, event):
        if self.cancel_button.clicked(event) or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            self.name_input.active = False
            app.screen = "game"
            return
        submitted = self.name_input.handle_event(event)
        if submitted or self.save_button.clicked(event):
            if app.save_current_simulation(name=self.name_input.text):
                self.name_input.active = False
                app.screen = "game"
            else:
                self.name_input.focus()

    def draw(self, app):
        app.game.draw(app)
        overlay = pygame.Surface(app.window.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        app.window.blit(overlay, (0, 0))
        pygame.draw.rect(app.window, PAPER, (390, 225, 500, 280))
        draw_centered_text(app.window, "Save simulation as", app.heading_font, 250)
        app.window.blit(app.small_font.render("Save name", True, INK), (420, 290))
        self.name_input.draw(app.window, app.font)
        hint = self.message or self.name_input.error or "Creates a new JSON file. Existing saves are kept."
        draw_wrapped_text(app.window, hint, app.small_font, MUTED, (420, 378, 440, 48))
        self.save_button.draw(app.window, app.font)
        self.cancel_button.draw(app.window, app.font)
