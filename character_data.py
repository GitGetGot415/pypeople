"""The character JSON format shared by the creator and the game."""

import json
from pathlib import Path


FIELDS = ("Name", "Age", "Weight", "Height", "Sex")
HAIR_STYLES = ("one", "two", "three", "four")
DEFAULT_COLORS = {
    "Shirt Color": "#000AFF", "Skin Color": "#FF0000",
    "Eye Color": "#05FF00", "Hair Color": "#FFA800",
}
DIRECTIONS = ("North", "North east", "East", "South east", "South", "South west", "West", "North west")


def parse_color(value):
    if not isinstance(value, str):
        raise ValueError("Colors must be hex text such as #FF0000.")
    hex_color = value.strip().removeprefix("#")
    if len(hex_color) != 6 or any(char not in "0123456789abcdefABCDEF" for char in hex_color):
        raise ValueError("Colors must be hex text such as #FF0000.")
    return tuple(int(hex_color[index:index + 2], 16) for index in (0, 2, 4))


def normalize_character(data):
    """Validate the whole character before callers apply any of its values."""
    if not isinstance(data, dict):
        raise ValueError("The JSON must contain a character object.")
    values = {}
    for name in FIELDS:
        value = data.get(name, "")
        if not isinstance(value, (str, int, float)):
            raise ValueError(f"{name} must be text.")
        values[name] = str(value)
    if "Hair" not in data or data["Hair"] not in HAIR_STYLES:
        raise ValueError("Hair must be one, two, three, or four.")
    values["Hair"] = data["Hair"]
    for name, default in DEFAULT_COLORS.items():
        try:
            rgb = parse_color(data.get(name, default))
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
        values[name] = "#" + "".join(f"{channel:02X}" for channel in rgb)
    return values


def load_character(path):
    with Path(path).open(encoding="utf-8") as character_file:
        return normalize_character(json.load(character_file))
