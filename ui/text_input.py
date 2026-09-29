import pygame
import pyperclip

from .theme import ACCENT, BORDER_RADIUS, INK, MUTED


class TextInput:
    """Single-line entry. Ctrl+A selects everything; text scrolls to its end."""

    def __init__(
        self, rect, *, max_length=300, secret=False, strip_paste=False, placeholder=""
    ):
        self.rect = pygame.Rect(rect)
        self.max_length = max_length
        self.secret = secret
        self.strip_paste = strip_paste
        self.placeholder = placeholder
        self.text = ""
        self.active = False
        self.selected_all = False
        self.error = ""
        self.backspace_held = False
        self.backspace_repeat_in = 0.0

    def set_value(self, value):
        self.text = value[: self.max_length]
        self.selected_all = False
        self.error = ""
        self.backspace_held = False
        self.backspace_repeat_in = 0.0

    def focus(self):
        self.active = True
        self.backspace_held = False
        pygame.key.set_text_input_rect(self.rect)

    def _insert(self, value):
        # Ignore newlines, control characters, and clipboard non-text values.
        value = "".join(char for char in value if char.isprintable())
        prefix = "" if self.selected_all else self.text
        self.text = (prefix + value)[: self.max_length]
        self.selected_all = False
        self.error = ""

    def _delete_backward(self):
        self.text = "" if self.selected_all else self.text[:-1]
        self.selected_all = False
        self.error = ""

    def update(self, dt):
        """Repeat Backspace after a short hold delay, then delete steadily."""
        if not self.active or not self.backspace_held:
            return
        self.backspace_repeat_in -= max(0.0, dt)
        while self.backspace_repeat_in <= 0.0:
            self._delete_backward()
            self.backspace_repeat_in += 0.04

    def handle_event(self, event):
        """Return True for Enter while focused; the caller decides to submit."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.active = self.rect.collidepoint(event.pos)
            self.selected_all = False
            if not self.active:
                self.backspace_held = False
            if self.active:
                self.focus()
        if event.type == pygame.KEYUP and event.key == pygame.K_BACKSPACE:
            self.backspace_held = False
        if not self.active:
            return False
        if event.type == pygame.TEXTINPUT:
            self._insert(event.text)
        elif event.type == pygame.KEYDOWN:
            control = bool(event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META))
            if control and event.key == pygame.K_a:
                self.selected_all = True
            elif control and event.key == pygame.K_v:
                try:
                    value = pyperclip.paste()
                    if not isinstance(value, str):
                        raise ValueError("Clipboard does not contain text")
                    self._insert(value.strip() if self.strip_paste else value)
                except Exception:
                    # Clipboard errors can contain sensitive text: never render them.
                    self.error = "Clipboard unavailable. You can still type manually."
            elif event.key == pygame.K_BACKSPACE:
                if not self.backspace_held:
                    self._delete_backward()
                    self.backspace_repeat_in = 0.4
                self.backspace_held = True
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                return True
        return False

    def draw(self, surface, font):
        pygame.draw.rect(surface, (255, 254, 248), self.rect, border_radius=BORDER_RADIUS)
        border = ACCENT if self.active else (181, 191, 174)
        pygame.draw.rect(surface, border, self.rect, width=2, border_radius=BORDER_RADIUS)
        inner = self.rect.inflate(-24, -12)
        display = "*" * len(self.text) if self.secret else self.text
        rendered = font.render(display or self.placeholder, True, INK if display else MUTED)
        x = inner.x + min(0, inner.width - rendered.get_width() - 3)
        y = inner.centery - rendered.get_height() // 2
        previous_clip = surface.get_clip()
        surface.set_clip(inner.clip(previous_clip))
        if self.active and self.selected_all and display:
            pygame.draw.rect(surface, (194, 222, 192), (x, y, rendered.get_width(), rendered.get_height()))
        surface.blit(rendered, (x, y))
        if self.active and pygame.time.get_ticks() % 1000 < 500:
            caret_x = x + (rendered.get_width() if display else 0) + 1
            pygame.draw.line(surface, INK, (caret_x, inner.y), (caret_x, inner.bottom), 2)
        surface.set_clip(previous_clip)
