"""Observer gallery, crisp previews, and explicit transient music playback."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import pygame

from constants import (ACCENT, ART_PALETTE, ART_PREVIEW_SCALE, ART_SIZE, CREATIONS_FILTER_RECTS,
                       CREATIONS_CONTENT_PADDING, CREATIONS_PLAY_RECT, CREATIONS_PREVIEW_TOP, CREATIONS_SCORE_TOP,
                       CREATIONS_STOP_RECT, CREATIONS_TEXT_INSET, INK, MUSIC_CACHE_SIZE, MUSIC_CHANNELS,
                       MUSIC_SAMPLE_FORMAT, MUSIC_SAMPLE_RATE, MUSIC_SUPPORTED_CHANNELS, MUTED)
from creations import MusicPayload
from music import duration, synthesize
from .button import Button
from .text import wrap_text


class CreationsGallery:
    def __init__(self):
        self.selected_id = None
        self.filter = "All"
        self.filters = {label: Button(rect, label) for label, rect in zip(("All", "Art", "Music"), CREATIONS_FILTER_RECTS)}
        self.play_button = Button(CREATIONS_PLAY_RECT, "Play")
        self.stop_button = Button(CREATIONS_STOP_RECT, "Stop")
        self.executor = None
        self.preparing = None
        self.pending_play = None
        self.next_play = None
        self.channel = None
        self.cache = OrderedDict()
        self.status = ""
        self.audio_error = ""
        self.preview = None
        self.preview_id = None
        self.preview_top = CREATIONS_PREVIEW_TOP

    def entries(self, world):
        return [work for work in reversed(world.creations) if self.filter == "All" or work.kind == self.filter.lower()]

    def selected(self, world):
        return next((work for work in world.creations if work.id == self.selected_id), None)

    def painting_preview(self, work, scale=ART_PREVIEW_SCALE):
        key = (work.id, scale)
        if self.preview_id != key:
            original = pygame.Surface((ART_SIZE, ART_SIZE))
            for y, row in enumerate(work.payload["pixels"]):
                for column, index in enumerate(row):
                    original.set_at((column, y), ART_PALETTE[index])
            self.preview = pygame.transform.scale(original, (ART_SIZE * scale,) * 2)
            self.preview_id = key
        return self.preview

    def select(self, work):
        if work.id != self.selected_id:
            self.stop()
            self.status = ""
            self.selected_id = work.id

    def stop(self):
        self.pending_play = None
        self.next_play = None
        if self.channel is not None:
            self.channel.stop()
            self.channel = None
            self.status = "Stopped"

    def close(self):
        self.stop()
        if self.preparing:
            self.preparing[1].cancel()
        self.preparing = None
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
        self.cache.clear()
        self.preview = self.preview_id = None

    def play(self, work):
        self.stop()
        if self.preparing:
            self.status = "Preparing audio..."
            if self.preparing[0][0] == work.id:
                self.pending_play = work.id
            else:
                self.next_play = work
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=MUSIC_SAMPLE_RATE, size=MUSIC_SAMPLE_FORMAT, channels=MUSIC_CHANNELS)
            format = pygame.mixer.get_init()
            if format[1] != MUSIC_SAMPLE_FORMAT or format[2] not in MUSIC_SUPPORTED_CHANNELS:
                raise ValueError("The audio device needs a signed 16-bit mono or stereo mixer.")
            self.audio_error = ""
            key = (work.id, format)
            if key in self.cache:
                self.cache.move_to_end(key)
                self.channel = self.cache[key].play(loops=0)
                self.status = "Playing" if self.channel else "Audio device has no available playback channel."
                return
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="Creation-audio")
            self.preparing = (key, self.executor.submit(synthesize, work.payload, *format))
            self.pending_play = work.id
            self.status = "Preparing audio..."
        except (pygame.error, ValueError) as exc:
            self.audio_error = f"Playback unavailable: {exc}"
            self.status = self.audio_error

    def update(self):
        if self.channel is not None and not self.channel.get_busy():
            self.channel = None
            self.status = "Finished"
        if not self.preparing or not self.preparing[1].done():
            return
        key, future = self.preparing
        self.preparing = None
        try:
            pcm = future.result()
            if pygame.mixer.get_init() != key[1]:
                raise ValueError("The audio device changed; press Play to prepare again.")
            sound = pygame.mixer.Sound(buffer=pcm)
            self.cache[key] = sound
            while len(self.cache) > MUSIC_CACHE_SIZE:
                self.cache.popitem(last=False)
            if self.pending_play == self.selected_id == key[0]:
                self.channel = sound.play(loops=0)
                self.status = "Playing" if self.channel else "Audio device has no available playback channel."
            else:
                self.status = "Ready"
        except (pygame.error, ValueError, RuntimeError) as exc:
            self.status = f"Playback unavailable: {exc}"
        self.pending_play = None
        requested = self.next_play
        self.next_play = None
        if requested is not None and requested.id == self.selected_id:
            self.play(requested)

    def handle_event(self, app, event, scroll):
        for label, button in self.filters.items():
            if button.clicked(event):
                self.stop()
                self.filter = label
                return "filter"
        work = self.selected(app.simulation)
        if work and work.kind == "music":
            for button, action in ((self.play_button, "play"), (self.stop_button, "stop")):
                rect = button.rect.move(0, -scroll)
                if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                        and app.stats.detail_rect.collidepoint(event.pos) and rect.collidepoint(event.pos)):
                    if action == "stop":
                        self.stop()
                    elif not self.audio_error:
                        self.play(work)
                    return "playback"
        return None

    def layout(self, app):
        """Return text positions and a content height for the shared detail scroller."""
        work = self.selected(app.simulation)
        rows = []
        y = 0
        def add(text, font=app.font, color=INK):
            nonlocal y
            for line in wrap_text(text, font, app.stats.detail_rect.width - CREATIONS_TEXT_INSET):
                rows.append((line, font, color, y))
                y += font.get_linesize()
        if not work:
            add("Select a painting or song to explore it.", color=MUTED)
            return rows, y
        add(work.title, app.heading_font)
        creator = app.simulation.actor_by_id(work.creator_id)
        add(f"By {creator.name}" + (" (departed)" if creator.departure else ""), app.small_font)
        owner = app.simulation.participant_by_id(work.owner_id)
        add(f"Owned by {owner.name}" + (" (departed)" if getattr(owner, "departure", None) else ""), app.small_font)
        add(f"Day {work.day} / {work.phase.title()} / {work.kind.title()}", app.small_font, MUTED)
        if work.kind == "art":
            self.preview_top = max(y, CREATIONS_PREVIEW_TOP)
            return rows, self.preview_top + ART_SIZE * ART_PREVIEW_SCALE + CREATIONS_CONTENT_PADDING
        score = MusicPayload.model_validate(work.payload)
        add(f"{score.tempo} BPM / {duration(score):.1f} seconds / {len(score.tracks)} "
            + ("track" if len(score.tracks) == 1 else "tracks"), app.small_font)
        y = max(y, CREATIONS_SCORE_TOP)
        add(self.status or "Press Play to hear this song once.", app.small_font, MUTED)
        for number, track in enumerate(score.tracks, 1):
            add(f"Track {number}: {track.waveform}, {len(track.notes)} notes", app.font, ACCENT)
            notes = ", ".join(f"{note.pitch}@{note.start}+{note.duration}" for note in track.notes) or "Rest"
            add(notes, app.small_font)
        add("Score: MIDI pitch @ start step + duration in steps.", app.small_font, MUTED)
        return rows, y

    def draw(self, app, scroll):
        surface = app.window
        for label, button in self.filters.items():
            button.draw(surface, app.small_font)
            if self.filter == label:
                pygame.draw.line(surface, INK, button.rect.bottomleft, button.rect.bottomright, 2)
        previous_clip = surface.get_clip()
        surface.set_clip(app.stats.detail_rect.clip(previous_clip))
        rows, _ = self.layout(app)
        x, top = app.stats.detail_rect.topleft
        for text, font, color, y in rows:
            surface.blit(font.render(text, True, color), (x, top + y - scroll))
        work = self.selected(app.simulation)
        if work and work.kind == "art":
            surface.blit(self.painting_preview(work), (x + CREATIONS_CONTENT_PADDING, top + self.preview_top - scroll))
        elif work:
            for button in (self.play_button, self.stop_button):
                moved = Button(button.rect.move(0, -scroll), button.label)
                moved.draw(surface, app.small_font, enabled=button is self.stop_button or not self.audio_error)
        surface.set_clip(previous_clip)
