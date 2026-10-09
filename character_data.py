"""The character JSON format shared by the creator and the game."""

import json
from pathlib import Path
import random


FIELDS = ("Name", "Age", "Weight", "Height", "Sex", "Details")
HAIR_STYLES = ("one", "two", "three", "four")
DEFAULT_COLORS = {
    "Shirt Color": "#000AFF", "Skin Color": "#FF0000",
    "Eye Color": "#05FF00", "Hair Color": "#FFA800",
}
DIRECTIONS = ("North", "North east", "East", "South east", "South", "South west", "West", "North west")
RANDOM_CONFIG_PATH = Path(__file__).resolve().parent / "assets" / "character_randomization.json"


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
    if "Details" not in data or not isinstance(data["Details"], str):
        raise ValueError("Details must be text.")
    values = {}
    for name in FIELDS:
        value = data[name] if name == "Details" else data.get(name, "")
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
    values["Opinions"] = normalize_opinions(data.get("Opinions", {}))
    return values


def normalize_opinions(opinions):
    """Opinions belong to game state and are deliberately excluded from FIELDS."""
    if not isinstance(opinions, dict):
        raise ValueError("Opinions must map person names to an opinion and reason.")
    values = {}
    for person, entry in opinions.items():
        if not isinstance(person, str) or not person.strip() or not isinstance(entry, dict):
            raise ValueError("Each opinion needs a person name, numeric opinion, and reason.")
        score = entry.get("opinion")
        reason = entry.get("reason")
        if type(score) is not int or not -100 <= score <= 100:
            raise ValueError("Opinion scores must be integers from -100 to 100.")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Each opinion needs a nonempty reason.")
        values[person.strip()] = {"opinion": score, "reason": reason.strip()}
    return values


def load_character(path):
    with Path(path).open(encoding="utf-8") as character_file:
        return normalize_character(json.load(character_file))


def random_character(config_path=RANDOM_CONFIG_PATH):
    """Generate a complete character using the creator's random settings."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Random settings must be a JSON object.")
    for key in ("first_names", "last_names", "details"):
        choices = config.get(key)
        if not isinstance(choices, list) or not choices or any(
            not isinstance(value, str) or not value.strip() for value in choices
        ):
            raise ValueError(f"{key} must be a nonempty list of text.")
    limits = config.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("Random settings must include numeric limits.")
    values = {}
    for name in ("Age", "Weight", "Height"):
        bounds = limits.get(name)
        if not isinstance(bounds, dict):
            raise ValueError(f"{name} needs min and max limits.")
        low, high = bounds.get("min"), bounds.get("max")
        if type(low) is not int or type(high) is not int or low > high:
            raise ValueError(f"{name} needs integer limits with min <= max.")
        values[name] = str(random.randint(low, high))
    values["Name"] = f"{random.choice(config['first_names']).strip()} {random.choice(config['last_names']).strip()}"
    values["Sex"] = random.choice(("M", "F"))
    values["Details"] = random.choice(config["details"]).strip()
    values["Hair"] = random.choice(HAIR_STYLES)
    for name in DEFAULT_COLORS:
        values[name] = "#" + "".join(f"{random.randint(0, 255):02X}" for _ in range(3))
    return normalize_character(values)
