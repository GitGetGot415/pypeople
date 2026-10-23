import pygame
import pyperclip

from constants import ACCENT, BORDER_RADIUS, INK, MUTED
from .text import scrollbar_rects, ScrollbarDrag


class TextInput:
    """Editable text with a caret, optional wrapping, and scrolling."""

    def __init__(
        self, rect, *, max_length=300, secret=False, strip_paste=False,
        placeholder="", multiline=False
    ):
        self.rect = pygame.Rect(rect)
        self.max_length = max_length
        self.secret = secret
        self.strip_paste = strip_paste
        self.placeholder = placeholder
        self.multiline = multiline
        self.text = ""
        self.cursor = 0
        self.scroll_y = 0
        self.scroll_x = 0
        self.scrollbar = ScrollbarDrag()
        self._font = None
        self._show_cursor = True
        self.active = False
        self.selected_all = False
        self.error = ""
        self.backspace_held = False
        self.backspace_repeat_in = 0.0
        self._navigation_key = None
        self._navigation_repeat_in = 0.0

    def set_value(self, value):
        self.scrollbar.cancel()
        self.text = value[: self.max_length]
        self.cursor = len(self.text)
        self.scroll_y = 0
        self.scroll_x = 0
        self._show_cursor = True
        self.selected_all = False
        self.error = ""
        self.backspace_held = False
        self.backspace_repeat_in = 0.0
        self._navigation_key = None

    def focus(self):
        self.active = True
        self.backspace_held = False
        self._navigation_key = None
        self._show_cursor = True
        pygame.key.set_text_input_rect(self.rect)

    def _insert(self, value):
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        value = "".join(char for char in value if char.isprintable() or (self.multiline and char == "\n"))
        if self.selected_all:
            self.text = ""
            self.cursor = 0
        if self.max_length is not None:
            value = value[: max(0, self.max_length - len(self.text))]
        self.text = self.text[:self.cursor] + value + self.text[self.cursor:]
        self.cursor += len(value)
        self.selected_all = False
        self.error = ""
        self._show_cursor = True

    def _delete_backward(self):
        if self.selected_all:
            self.text = ""
            self.cursor = 0
        elif self.cursor > 0:
            self.text = self.text[:self.cursor - 1] + self.text[self.cursor:]
            self.cursor -= 1
        self.selected_all = False
        self.error = ""
        self._show_cursor = True

    def _delete_forward(self):
        if self.selected_all:
            self._delete_backward()
        else:
            self.text = self.text[:self.cursor] + self.text[self.cursor + 1:]
            self.error = ""
            self._show_cursor = True

    def _move_cursor(self, position):
        self.cursor = max(0, min(len(self.text), position))
        self.selected_all = False
        self.backspace_held = False
        self._show_cursor = True

    @staticmethod
    def _wrapped_lines(text, font, width):
        """Return source ranges, preserving spaces, newlines, and empty lines."""
        lines = []
        start = 0
        last_space = None
        for index, char in enumerate(text):
            if char == "\n":
                lines.append((start, index))
                start = index + 1
                last_space = None
                continue
            while index > start and font.size(text[start:index + 1])[0] > width:
                end = last_space + 1 if last_space is not None else index
                lines.append((start, end))
                start = end
                space = text.rfind(" ", start, index)
                last_space = space if space >= start else None
            if char == " ":
                last_space = index
        lines.append((start, len(text)))
        return lines

    def _cursor_line(self, lines):
        # At a soft wrap, the caret belongs to the start of the next line.
        return next(index for index in range(len(lines) - 1, -1, -1) if lines[index][0] <= self.cursor)

    def _index_at_x(self, start, end, x):
        return min(range(start, end + 1), key=lambda index: abs(self._font.size(self.text[start:index])[0] - x))

    def _place_cursor(self, pos):
        if self._font is None:
            return
        inner = self.rect.inflate(-24, -12)
        if self.multiline:
            lines = self._wrapped_lines(self.text, self._font, inner.width)
            row = max(0, min(len(lines) - 1, (pos[1] - inner.y + self.scroll_y) // self._font.get_linesize()))
            start, end = lines[row]
            index = self._index_at_x(start, end, pos[0] - inner.x)
        else:
            display = "*" * len(self.text) if self.secret else self.text
            x = pos[0] - inner.x + self.scroll_x
            index = min(range(len(self.text) + 1), key=lambda index: abs(self._font.size(display[:index])[0] - x))
        self._move_cursor(index)

    def _move_vertical(self, step):
        if self._font is None:
            return
        lines = self._wrapped_lines(self.text, self._font, self.rect.width - 24)
        row = self._cursor_line(lines)
        x = self._font.size(self.text[lines[row][0]:self.cursor])[0]
        start, end = lines[max(0, min(len(lines) - 1, row + step))]
        self._move_cursor(self._index_at_x(start, end, x))

    def _navigate(self, key):
        if key in (pygame.K_LEFT, pygame.K_RIGHT):
            if self.selected_all:
                position = 0 if key == pygame.K_LEFT else len(self.text)
            else:
                position = self.cursor + (-1 if key == pygame.K_LEFT else 1)
            self._move_cursor(position)
        else:
            self._move_vertical(-1 if key == pygame.K_UP else 1)

    def scroll(self, steps):
        """Scroll wrapped text; return whether it moved so a parent can scroll."""
        if not self.multiline or self._font is None:
            return False
        lines = self._wrapped_lines(self.text, self._font, self.rect.width - 24)
        limit = max(0, len(lines) * self._font.get_linesize() - (self.rect.height - 12))
        previous = self.scroll_y
        self.scroll_y = max(0, min(limit, self.scroll_y - steps * self._font.get_linesize() * 3))
        self._show_cursor = False
        return self.scroll_y != previous

    def update(self, dt):
        """Repeat held Backspace and arrow keys after a short delay."""
        if not self.active:
            self._navigation_key = None
            return
        if self.backspace_held:
            self.backspace_repeat_in -= max(0.0, dt)
            while self.backspace_repeat_in <= 0.0:
                self._delete_backward()
                self.backspace_repeat_in += 0.04
        if self._navigation_key is not None:
            self._navigation_repeat_in -= max(0.0, dt)
            while self._navigation_repeat_in <= 0.0:
                self._navigate(self._navigation_key)
                self._navigation_repeat_in += 0.04

    def handle_scroll_event(self, event):
        if (not self.multiline or self._font is None or event.type not in
                (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION, pygame.WINDOWFOCUSLOST)):
            return False
        inner = self.rect.inflate(-24, -12)
        lines = self._wrapped_lines(self.text, self._font, inner.width)
        limit = max(0, len(lines) * self._font.get_linesize() - inner.height)
        value = self.scrollbar.handle_event(event, self.rect.inflate(-4, -12), self.scroll_y, limit)
        if value is None:
            return False
        self.scroll_y = value
        self._show_cursor = False
        return True

    def handle_event(self, event):
        """Enter submits single-line fields or inserts a multiline newline."""
        if self.handle_scroll_event(event):
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.active = self.rect.collidepoint(event.pos)
            self.selected_all = False
            if not self.active:
                self.backspace_held = False
            if self.active:
                self.focus()
                self._place_cursor(event.pos)
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_BACKSPACE:
                self.backspace_held = False
            if event.key == self._navigation_key:
                self._navigation_key = None
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
            elif event.key == pygame.K_DELETE:
                self._delete_forward()
            elif event.key in (pygame.K_LEFT, pygame.K_RIGHT) or (self.multiline and event.key in (pygame.K_UP, pygame.K_DOWN)):
                self._navigate(event.key)
                self._navigation_key = event.key
                self._navigation_repeat_in = 0.4
            elif event.key in (pygame.K_HOME, pygame.K_END):
                start, end = 0, len(self.text)
                if self.multiline and not control and self._font is not None:
                    lines = self._wrapped_lines(self.text, self._font, self.rect.width - 24)
                    start, end = lines[self._cursor_line(lines)]
                self._move_cursor(start if event.key == pygame.K_HOME else end)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if self.multiline:
                    self._insert("\n")
                else:
                    return True
        return False

    def draw(self, surface, font):
        self._font = font
        pygame.draw.rect(surface, (255, 254, 248), self.rect, border_radius=BORDER_RADIUS)
        border = ACCENT if self.active else (181, 191, 174)
        pygame.draw.rect(surface, border, self.rect, width=2, border_radius=BORDER_RADIUS)
        inner = self.rect.inflate(-24, -12)
        display = "*" * len(self.text) if self.secret else self.text
        previous_clip = surface.get_clip()
        surface.set_clip(inner.clip(previous_clip))
        if self.multiline:
            lines = self._wrapped_lines(display, font, inner.width)
            row = self._cursor_line(lines)
            line_height = font.get_linesize()
            if self.active and self._show_cursor:
                self.scroll_y = min(self.scroll_y, row * line_height)
                self.scroll_y = max(self.scroll_y, (row + 1) * line_height - inner.height)
                self._show_cursor = False
            self.scroll_y = max(0, min(self.scroll_y, len(lines) * line_height - inner.height))
            visible_lines = lines if display else self._wrapped_lines(self.placeholder, font, inner.width)
            for index, (start, end) in enumerate(visible_lines):
                y = inner.y + index * line_height - self.scroll_y
                if y + line_height <= inner.y or y >= inner.bottom:
                    continue
                text = (display or self.placeholder)[start:end]
                rendered = font.render(text, True, INK if display else MUTED)
                if self.active and self.selected_all and display:
                    pygame.draw.rect(surface, (194, 222, 192), (inner.x, y, max(3, rendered.get_width()), line_height))
                surface.blit(rendered, (inner.x, y))
            caret_x = inner.x + font.size(display[lines[row][0]:self.cursor])[0]
            caret_y = inner.y + row * line_height - self.scroll_y
            caret_height = line_height
        else:
            rendered = font.render(display or self.placeholder, True, INK if display else MUTED)
            cursor_x = font.size(display[:self.cursor])[0]
            self.scroll_x = max(0, min(self.scroll_x, cursor_x))
            self.scroll_x = max(self.scroll_x, cursor_x - inner.width + 3)
            x = inner.x - self.scroll_x
            y = inner.centery - rendered.get_height() // 2
            if self.active and self.selected_all and display:
                pygame.draw.rect(surface, (194, 222, 192), (x, y, rendered.get_width(), rendered.get_height()))
            surface.blit(rendered, (x, y))
            caret_x, caret_y, caret_height = x + cursor_x, inner.y, inner.height
        if self.active and pygame.time.get_ticks() % 1000 < 500:
            pygame.draw.line(surface, INK, (caret_x + 1, caret_y), (caret_x + 1, caret_y + caret_height), 2)
        surface.set_clip(previous_clip)
        if self.multiline and len(lines) * line_height > inner.height:
            track, thumb = scrollbar_rects(self.rect.inflate(-4, -12), self.scroll_y,
                                           len(lines) * line_height - inner.height)
            pygame.draw.rect(surface, (181, 191, 174), track)
            pygame.draw.rect(surface, ACCENT, thumb)
