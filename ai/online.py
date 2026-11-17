"""Online connection choices and nonsecret preferences."""

from dataclasses import dataclass
from urllib.parse import urlsplit


# Providers with a native SDK connect to their own service and take no base URL.
NATIVE_PROVIDERS = ("gemini", "anthropic")
PROVIDERS = {
    "gemini": ("Google Gemini", "gemini-2.5-flash", "", "GEMINI_API_KEY"),
    "openai": ("OpenAI", "gpt-4.1-mini", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ("Anthropic Claude", "claude-opus-5-5", "", "ANTHROPIC_API_KEY"),
    "custom": ("Custom / OpenAI-compatible", "", "", ""),
}


@dataclass(frozen=True)
class OnlineConfig:
    provider: str = "gemini"
    model: str = PROVIDERS["gemini"][1]
    base_url: str = ""

    @property
    def label(self):
        return PROVIDERS[self.provider][0]

    def validate(self):
        if self.provider not in PROVIDERS:
            raise ValueError("Choose an online provider in Settings.")
        if not self.model.strip():
            raise ValueError("Enter a model name for the online provider.")
        if self.provider not in NATIVE_PROVIDERS:
            try:
                url = urlsplit(self.base_url)
                valid = (url.scheme == "https" and url.hostname and not url.username
                         and not url.password and not url.query and not url.fragment)
                url.port  # Reject invalid ports before starting a request.
            except ValueError:
                valid = False
            if not valid:
                raise ValueError("Enter an HTTPS base URL without credentials, a query, or a fragment.")


def load_profiles(value):
    """Read only known, nonsecret fields; incomplete custom profiles are editable."""
    value = value if isinstance(value, dict) else {}
    profiles = {}
    for provider, (_, model, base_url, _) in PROVIDERS.items():
        profile = value.get(provider, {})
        profile = profile if isinstance(profile, dict) else {}
        profiles[provider] = OnlineConfig(
            provider,
            profile["model"].strip() if isinstance(profile.get("model"), str) else model,
            (profile["base_url"].strip().rstrip("/")
             if provider == "custom" and isinstance(profile.get("base_url"), str) else base_url),
        )
    return profiles


def profile_settings(profiles):
    return {provider: {"model": profiles[provider].model, "base_url": profiles[provider].base_url}
            for provider in PROVIDERS}
