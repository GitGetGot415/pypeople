"""Bounded, standard-library chiptune synthesis. No audio device is needed here."""

from array import array
import math

from constants import (MUSIC_ATTACK_SECONDS, MUSIC_RELEASE_SECONDS, MUSIC_SAMPLE_FORMAT,
                       MUSIC_SAMPLE_RATE_RANGE, MUSIC_STEPS, MUSIC_STEPS_PER_BEAT, MUSIC_SUPPORTED_CHANNELS,
                       MUSIC_TRACK_GAIN)
from creations import MusicPayload


def duration(score):
    return MUSIC_STEPS * 60.0 / score.tempo / MUSIC_STEPS_PER_BEAT


def synthesize(payload, sample_rate, sample_format, channels):
    """Return native signed 16-bit PCM matching the mixer's actual channel count."""
    score = MusicPayload.model_validate(payload)
    if (sample_format != MUSIC_SAMPLE_FORMAT or channels not in MUSIC_SUPPORTED_CHANNELS
            or not MUSIC_SAMPLE_RATE_RANGE[0] <= sample_rate <= MUSIC_SAMPLE_RATE_RANGE[1]):
        raise ValueError("Playback requires a mono or stereo signed 16-bit mixer.")
    step = 60.0 / score.tempo / MUSIC_STEPS_PER_BEAT
    samples = [0.0] * math.ceil(duration(score) * sample_rate)
    for track in score.tracks:
        for note in track.notes:
            start = round(note.start * step * sample_rate)
            count = min(round(note.duration * step * sample_rate), len(samples) - start)
            frequency = 440.0 * 2 ** ((note.pitch - 69) / 12)
            attack = max(1, round(MUSIC_ATTACK_SECONDS * sample_rate))
            release = max(1, round(MUSIC_RELEASE_SECONDS * sample_rate))
            for i in range(count):
                phase = (i * frequency / sample_rate) % 1.0
                wave = (1.0 if phase < 0.5 else -1.0) if track.waveform == "square" else 1.0 - 4.0 * abs(phase - 0.5)
                envelope = min(1.0, i / attack, (count - 1 - i) / release)
                samples[start + i] += wave * envelope * MUSIC_TRACK_GAIN
    output = array("h")
    for sample in samples:
        value = round(max(-1.0, min(1.0, sample)) * 32767)
        for _ in range(channels):
            output.append(value)
    # Pygame's native mixer format uses native byte order, as does array('h').
    return output.tobytes()
