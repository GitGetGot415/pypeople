import pygame

from .button import Button
from .text import draw_centered_text, draw_wrapped_text
from .text_input import TextInput
from constants import ACCENT, GRASS, INK, MODAL_OVERLAY_COLOR, MUTED, PAPER


class LoadSimScreen:
    ROWS_PER_PAGE = 5

    def __init__(self):
        self.back_button = Button((260, 600, 160, 42), "Back")
        self.refresh_button = Button((860, 600, 160, 42), "Refresh")
        self.previous_button = Button((260, 545, 120, 34), "Previous")
        self.next_button = Button((620, 545, 120, 34), "Next")
        self.load_button = Button((780, 340, 240, 44), "Load simulation")
        self.rename_button = Button((780, 398, 240, 44), "Rename save")
        self.delete_button = Button((780, 456, 240, 44), "Delete save")
        self.confirm_button = Button((480, 395, 145, 44), "Delete")
        self.cancel_button = Button((655, 395, 145, 44), "Cancel")
        self.name_input = TextInput((780, 265, 240, 48), max_length=80, placeholder="Save name")
        self.entries = []
        self.selected_id = None
        self.page = 0
        self.message = ""
        self.confirm_delete = False

    @property
    def selected(self):
        return next((entry for entry in self.entries if entry.id == self.selected_id), None)

    @property
    def last_page(self):
        return max(0, (len(self.entries) - 1) // self.ROWS_PER_PAGE)

    def select(self, entry):
        self.selected_id = entry.id
        self.name_input.set_value(entry.name)
        self.name_input.active = False

    def open(self, store):
        self.message = ""
        self.confirm_delete = False
        self.name_input.active = False
        self.refresh(store)

    def refresh(self, store):
        try:
            self.entries = store.list_saves()
        except OSError:
            self.message = "Could not read saved simulations. Check folder permissions and try again."
            return
        self.page = min(self.page, self.last_page)
        entry = self.selected
        if entry is None and self.entries:
            entry = self.entries[self.page * self.ROWS_PER_PAGE]
        if entry is not None:
            self.select(entry)
        else:
            self.selected_id = None
            self.name_input.set_value("")

    def rename(self, app):
        if self.selected is None or self.selected.error:
            return
        try:
            self.selected_id = app.save_store.rename(self.selected_id, self.name_input.text)
        except FileExistsError:
            self.message = "A save with that name already exists. Choose another name."
            return
        except (OSError, ValueError):
            self.message = "Could not rename save. Use a valid filename and check folder permissions."
            return
        self.refresh(app.save_store)
        self.message = "Save renamed."

    def handle_event(self, app, event):
        escape = event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        if self.confirm_delete:
            if escape or self.cancel_button.clicked(event):
                self.confirm_delete = False
            elif self.confirm_button.clicked(event):
                try:
                    app.save_store.delete(self.selected_id)
                except (OSError, ValueError):
                    self.message = "Could not delete save. Check folder permissions and try again."
                else:
                    self.message = "Save deleted."
                self.confirm_delete = False
                self.refresh(app.save_store)
            return
        if escape or self.back_button.clicked(event):
            self.name_input.active = False
            app.screen = "menu"
            return
        entry = self.selected
        if self.load_button.clicked(event, enabled=entry is not None and not entry.error):
            self.name_input.active = False
            app.load_simulation(entry.id)
        elif self.rename_button.clicked(event, enabled=entry is not None and not entry.error):
            self.rename(app)
        elif self.delete_button.clicked(event, enabled=entry is not None):
            self.name_input.active = False
            self.confirm_delete = True
        elif self.refresh_button.clicked(event):
            self.message = ""
            self.refresh(app.save_store)
        elif self.previous_button.clicked(event, enabled=self.page > 0):
            self.page -= 1
        elif self.next_button.clicked(event, enabled=self.page < self.last_page):
            self.page += 1
        elif event.type == pygame.MOUSEWHEEL:
            self.page = max(0, min(self.last_page, self.page - event.y))
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for row, candidate in enumerate(self.entries[self.page * self.ROWS_PER_PAGE:][:self.ROWS_PER_PAGE]):
                if pygame.Rect(260, 185 + row * 68, 480, 60).collidepoint(event.pos):
                    self.select(candidate)
                    self.message = ""
                    return
            if entry is not None and not entry.error:
                self.name_input.handle_event(event)
        elif entry is not None and not entry.error and self.name_input.handle_event(event):
            self.rename(app)

    def draw(self, app):
        surface = app.window
        surface.fill(GRASS)
        pygame.draw.rect(surface, PAPER, (230, 75, 820, 510))
        draw_centered_text(surface, "Load simulation", app.heading_font, 100)
        surface.blit(app.small_font.render("Choose a simulation to continue, rename, or delete.", True, MUTED), (260, 148))
        entries = self.entries[self.page * self.ROWS_PER_PAGE:][:self.ROWS_PER_PAGE]
        if not entries:
            draw_wrapped_text(surface, "No saved simulations yet. Start a New Sim, then use Save to keep it.", app.font, MUTED, (260, 200, 480, 100))
        for row, entry in enumerate(entries):
            rect = pygame.Rect(260, 185 + row * 68, 480, 60)
            selected = entry.id == self.selected_id
            pygame.draw.rect(surface, ACCENT if selected else GRASS, rect)
            color = PAPER if selected else INK
            # Keep long names inside their row while retaining the full name in the editor.
            label = entry.name
            while app.font.size(label)[0] > rect.width - 24:
                label = label[:-4] + "..."
            surface.blit(app.font.render(label, True, color), (rect.x + 12, rect.y + 8))
            timestamp = entry.updated_at.astimezone().strftime("%Y-%m-%d %H:%M") if entry.updated_at else "Unreadable file"
            detail = f"{entry.npc_count} NPCs / {entry.mode.title()} / {timestamp}" if not entry.error else timestamp
            surface.blit(app.small_font.render(detail, True, color), (rect.x + 12, rect.y + 34))
        entry = self.selected
        surface.blit(app.font.render("Save name", True, INK), (780, 230))
        if entry is not None:
            self.name_input.draw(surface, app.font)
        else:
            surface.blit(app.small_font.render("Select a save first.", True, MUTED), (780, 275))
        enabled = entry is not None and not entry.error
        self.load_button.draw(surface, app.font, enabled=enabled)
        self.rename_button.draw(surface, app.font, enabled=enabled)
        self.delete_button.draw(surface, app.font, enabled=entry is not None)
        self.previous_button.draw(surface, app.small_font, enabled=self.page > 0)
        self.next_button.draw(surface, app.small_font, enabled=self.page < self.last_page)
        surface.blit(app.small_font.render(f"Page {self.page + 1}/{self.last_page + 1}", True, MUTED), (438, 554))
        self.back_button.draw(surface, app.font)
        self.refresh_button.draw(surface, app.font)
        hint = self.message or (entry.error if entry else "") or "Edit the name above, then choose Rename save."
        draw_wrapped_text(surface, hint, app.small_font, INK, (260, 650, 760, 28))
        if self.confirm_delete:
            overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            overlay.fill(MODAL_OVERLAY_COLOR)
            surface.blit(overlay, (0, 0))
            pygame.draw.rect(surface, PAPER, (395, 260, 490, 205))
            draw_centered_text(surface, "Delete this simulation?", app.heading_font, 280)
            draw_wrapped_text(surface, entry.name, app.font, INK, (425, 320, 430, 48))
            draw_centered_text(surface, "This permanently removes the save.", app.small_font, 370, MUTED)
            self.confirm_button.draw(surface, app.font)
            self.cancel_button.draw(surface, app.font)
