"""Provider I/O for prepared requests; no conversation snapshots or stage decisions."""

import json
import re
import unicodedata
import httpx
from contextlib import contextmanager
from dataclasses import dataclass
from time import monotonic
from google import genai
from google.genai import errors, types
from pydantic import ValidationError
from dialogue.errors import RequestError, InvalidResponseError, StreamRepetitionError
from constants import (
    MODEL_STREAM_REPEAT_CHECK_CHARACTERS, MODEL_STREAM_REPEAT_COUNT,
    MODEL_STREAM_REPEAT_MAX_BLOCK_CHARACTERS, MODEL_STREAM_REPEAT_MIN_CHARACTERS,
    MODEL_STREAM_REPEAT_PUNCTUATION_CHARACTERS, MODEL_STREAM_REPEAT_SYMBOL_CHARACTERS,
)
from ai.debug import check_cancelled, cancellable_stream, disconnect_response, record_output, streaming_output, watch_response
from ai.online import OnlineConfig


OLLAMA_URL = "http://localhost:11434/api/chat"


@dataclass(frozen=True)
class TransportSettings:
    timeout: float
    online_temperature: float | None
    ollama_options: dict


class _StreamRepetitionGuard:
    """Bounded, request-local checks, independent of provider chunk boundaries."""

    def __init__(self):
        self.tails = {}
        self.unchecked = {}

    def add(self, kind, text):
        window = max(MODEL_STREAM_REPEAT_MIN_CHARACTERS,
                     MODEL_STREAM_REPEAT_COUNT * MODEL_STREAM_REPEAT_MAX_BLOCK_CHARACTERS)
        tail = self.tails.get(kind, "")
        unchecked = self.unchecked.get(kind, 0)
        for offset in range(0, len(text), MODEL_STREAM_REPEAT_CHECK_CHARACTERS):
            piece = text[offset:offset + MODEL_STREAM_REPEAT_CHECK_CHARACTERS].casefold()
            tail = (tail + piece)[-window:]
            unchecked += len(piece)
            # A short punctuation loop can crash a constrained decoder before
            # the general repeated-prose threshold is reached.
            punctuation = tail[-MODEL_STREAM_REPEAT_PUNCTUATION_CHARACTERS:]
            if (len(punctuation) == MODEL_STREAM_REPEAT_PUNCTUATION_CHARACTERS
                    and re.fullmatch(r"([.!?,;:\u2026])(?:\s*\1)+\s*", punctuation.strip())):
                reason = ("Generation stopped: the stream kept repeating punctuation. "
                          "Start fresh, follow the requested Voice and return complete finite JSON strings.")
                record_output("validation", reason)
                raise StreamRepetitionError(reason)
            symbols = tail[-MODEL_STREAM_REPEAT_SYMBOL_CHARACTERS:]
            if len(symbols) == MODEL_STREAM_REPEAT_SYMBOL_CHARACTERS and not re.search(r"\w", symbols):
                for size in range(1, len(symbols) // MODEL_STREAM_REPEAT_COUNT + 1):
                    block = symbols[-size:]
                    if (any(unicodedata.category(char).startswith(("S", "M")) for char in block)
                            and symbols[:-size] == symbols[size:]):
                        reason = (f"Generation stopped: the {kind} stream kept repeating the same symbol sequence. "
                                  "Start fresh; use a finite sequence representing each original sentence, "
                                  "then close every JSON string. Do not loop on emoji or modifiers.")
                        record_output("validation", reason)
                        raise StreamRepetitionError(reason)
            if unchecked < MODEL_STREAM_REPEAT_CHECK_CHARACTERS:
                continue
            unchecked = 0
            if len(tail) < MODEL_STREAM_REPEAT_MIN_CHARACTERS:
                continue
            for size in range(1, min(MODEL_STREAM_REPEAT_MAX_BLOCK_CHARACTERS,
                                     len(tail) // MODEL_STREAM_REPEAT_COUNT) + 1):
                if tail[-size:] != tail[-2 * size:-size]:
                    continue
                length = max(MODEL_STREAM_REPEAT_MIN_CHARACTERS, MODEL_STREAM_REPEAT_COUNT * size)
                sample = tail[-length:]
                if sample[:-size] == sample[size:]:
                    reason = (f"Generation stopped: the {kind} stream kept repeating the same text. "
                              "Start fresh and return one finite JSON response for the current task. "
                              "Do not repeat phrases, paragraphs, or reasoning in a loop; finish the response.")
                    record_output("validation", reason)
                    raise StreamRepetitionError(reason)
        self.tails[kind], self.unchecked[kind] = tail, unchecked


def _api_error_message(exc, model):
    # Inspect provider text only to classify it; never display or log it.
    message = (exc.message or "").lower()
    if exc.code == 429:
        return "Gemini quota reached. Try again later or check your Google AI Studio limits."
    if "api key" in message or "api_key" in message or exc.code == 401:
        return "Gemini rejected your API key. Check it in Settings."
    if exc.code == 404 or (
        exc.code in (400, 403) and "model" in message
    ):
        return f"Model {model} is unavailable for this key. Check Gemini model access."
    if exc.code == 403:
        return "Gemini access denied. Check your API key and Google AI Studio access."
    if exc.code in (408, 504):
        return "Gemini timed out. Check your connection and try again."
    if exc.code >= 500:
        return "Gemini is temporarily unavailable. Try again later."
    return "Gemini could not process this request. Check model access or try another message."


def _transport_messages(request):
    """Serialize structured context only at the provider boundary."""
    messages = []
    for message in request.messages:
        content = message["content"]
        if not isinstance(content, str):
            # Retain the existing presentation of player assistant examples and
            # engine reminders; compact world and resident speech objects.
            compact = "world_state" in content or "parts" in content or "author" in content
            content = json.dumps(content, ensure_ascii=False,
                                 separators=(",", ":") if compact else None)
        messages.append({"role": message["role"], "content": content})
    return messages


def _gemini_contents(messages):
    if len(messages) == 1:
        return messages[0]["content"]
    contents = []
    for message in messages:
        role = "model" if message["role"] == "assistant" else "user"
        part = types.Part.from_text(text=message["content"])
        if contents and contents[-1].role == role:
            contents[-1].parts.append(part)
        else:
            contents.append(types.Content(role=role, parts=[part]))
    return contents


@contextmanager
def provider_errors(provider, model, online_config=OnlineConfig()):
    """Keep provider-specific error wording around I/O and shared validation."""
    if provider == "online" and online_config.provider == "gemini":
        try:
            yield
        except (RequestError, InvalidResponseError):
            raise
        except (ValidationError, ValueError):
            raise RequestError("Gemini returned an invalid action or reply. Try again.") from None
        except errors.APIError as exc:
            raise RequestError(_api_error_message(exc, online_config.model)) from None
        except (TimeoutError, httpx.TimeoutException):
            raise RequestError("Gemini timed out. Check your connection and try again.") from None
        except (OSError, httpx.HTTPError):
            raise RequestError("Cannot reach Gemini. Check your connection and try again.") from None
        except Exception:
            raise RequestError("Could not get a reply from Gemini. Check your connection or try again.") from None
    elif provider == "online":
        label = online_config.label
        try:
            yield
        except (RequestError, InvalidResponseError):
            raise
        except (TimeoutError, httpx.TimeoutException):
            raise RequestError(f"{label} timed out. Check your connection and try again.") from None
        except (OSError, httpx.HTTPError):
            raise RequestError(f"Cannot reach {label}. Check your connection and endpoint in Settings.") from None
        except (ValidationError, ValueError, KeyError, TypeError, AttributeError):
            raise RequestError(f"{label} returned an invalid action or reply. Try again.") from None
        except Exception:
            raise RequestError(f"Could not get a reply from {label}. Check Settings or try again.") from None
    else:
        try:
            yield
        except (RequestError, InvalidResponseError):
            raise
        except httpx.TimeoutException:
            raise RequestError(f"Ollama timed out. Try again after {model} finishes loading.") from None
        except httpx.ConnectError:
            raise RequestError("Cannot reach Ollama. Start Ollama and try again.") from None
        except httpx.HTTPError:
            raise RequestError("Cannot reach Ollama. Check that it is running and try again.") from None
        except (ValidationError, ValueError, KeyError, TypeError, AttributeError):
            raise RequestError("Ollama returned an invalid action or reply. Try again.") from None


def request_gemini(api_key, prepared, settings, model):
    contents = _gemini_contents(_transport_messages(prepared))
    with genai.Client(
        api_key=api_key,
        vertexai=False,
        http_options=types.HttpOptions(
            timeout=max(1, int(settings.timeout * 1000)),
            retry_options=types.HttpRetryOptions(attempts=1),
            client_args={"event_hooks": {"response": [watch_response]}},
        ),
    ) as client, cancellable_stream(client.close):
        config = types.GenerateContentConfig(
            system_instruction=prepared.system_instruction,
            temperature=settings.online_temperature,
            response_mime_type="application/json",
            response_schema=prepared.response_schema,
            candidate_count=1,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        live = streaming_output()
        if live:
            config.thinking_config = types.ThinkingConfig(include_thoughts=True)
            responses = client.models.generate_content_stream(model=model, contents=contents, config=config)
        else:
            responses = [client.models.generate_content(model=model, contents=contents, config=config)]
        chunks, reason = [], None
        repetition = _StreamRepetitionGuard()
        deadline = monotonic() + settings.timeout
        for response in responses:
            if monotonic() > deadline:
                raise RequestError("Gemini timed out. Check your connection and try again.")
            feedback = response.prompt_feedback
            if (feedback and feedback.block_reason
                    and feedback.block_reason != types.BlockedReason.BLOCKED_REASON_UNSPECIFIED):
                raise RequestError("Gemini blocked this request. Try another message.")
            if not response.candidates:
                continue
            candidate = response.candidates[0]
            if candidate.finish_reason is not None:
                reason = candidate.finish_reason
            if live:
                for part in (candidate.content.parts if candidate.content else ()) or ():
                    if part.text:
                        kind = "reasoning" if part.thought else "output"
                        record_output(kind, part.text)
                        repetition.add(kind, part.text)
                        if not part.thought:
                            chunks.append(part.text)
            elif response.text:
                chunks.append(response.text)
        if reason == types.FinishReason.MAX_TOKENS:
            raise RequestError("Gemini returned an incomplete reply. Try again.")
        if reason != types.FinishReason.STOP:
            raise RequestError("Gemini could not finish a reply. Try another message.")
        text = "".join(chunks)
        if not text or not text.strip():
            raise RequestError("Gemini returned an empty reply. Try again.")
        return text


def request_online(api_key, prepared, settings, config):
    try:
        config.validate()
    except ValueError as exc:
        raise RequestError(str(exc)) from None
    if not api_key.strip():
        raise RequestError(f"Add a {config.label} API key in Settings.")
    if config.provider == "gemini":
        return request_gemini(api_key, prepared, settings, config.model)
    return request_openai_compatible(api_key, prepared, settings, config)


def _online_http_error(status, config):
    label = config.label
    if status == 401:
        return f"{label} rejected your API key. Check it in Settings."
    if status == 403:
        return f"{label} denied access. Check your key and model permissions."
    if status == 404:
        return f"{label} could not find the endpoint or model. Check Settings."
    if status == 429:
        return f"{label} rate or quota limit reached. Try later or check your account limits."
    if status in (408, 504):
        return f"{label} timed out. Try again."
    if status >= 500:
        return f"{label} is temporarily unavailable. Try again later."
    return (f"{label} rejected the request. Check the model and base URL; "
            "the model must support Chat Completions with JSON output.")


def _sse_data(response, deadline, label):
    """SSE events can span multiple data lines; comments are keepalives."""
    data = []
    for line in response.iter_lines():
        check_cancelled()
        if monotonic() > deadline:
            raise RequestError(f"{label} timed out. Try again.")
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


def request_openai_compatible(api_key, prepared, settings, config):
    live = streaming_output()
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": prepared.system_instruction
             + "\nReturn only JSON matching this schema: "
             + json.dumps(prepared.response_schema, ensure_ascii=False, separators=(",", ":"))},
            *_transport_messages(prepared),
        ],
        "response_format": {"type": "json_object"},
        "stream": live,
    }
    # Leave sampling defaults to the service: some reasoning models reject temperature.
    deadline = monotonic() + settings.timeout
    chunks, reason = [], None
    repetition = _StreamRepetitionGuard()

    def consume(data, streaming):
        nonlocal reason
        if data.get("error"):
            raise RequestError(f"{config.label} could not complete the reply. Check model access or try again.")
        choices = data.get("choices", [])
        if not choices:
            return
        choice = choices[0]
        if choice.get("finish_reason") is not None:
            reason = choice["finish_reason"]
        message = choice.get("delta" if streaming else "message", {})
        if message.get("refusal"):
            raise RequestError(f"{config.label} declined this request. Try another message.")
        text = message.get("content") or ""
        if not isinstance(text, str):
            raise RequestError(f"{config.label} returned an unsupported reply format.")
        if live:
            record_output("output", text)
            repetition.add("output", text)
            thinking = message.get("reasoning_content")
            if isinstance(thinking, str):
                record_output("reasoning", thinking)
                repetition.add("reasoning", thinking)
        chunks.append(text)

    with httpx.stream(
        "POST", config.base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key.strip()}"}, json=payload,
        timeout=settings.timeout,
    ) as response, cancellable_stream(lambda: disconnect_response(response)):
        if response.is_error or response.is_redirect:
            # Never expose provider error bodies, headers, or URLs containing secrets.
            raise RequestError(_online_http_error(response.status_code, config))
        if live:
            for event in _sse_data(response, deadline, config.label):
                if event.strip() == "[DONE]":
                    break
                consume(json.loads(event), True)
        else:
            response.read()
            consume(response.json(), False)
    if monotonic() > deadline:
        raise RequestError(f"{config.label} timed out. Try again.")
    if reason == "length":
        raise RequestError(f"{config.label} returned an incomplete reply. Try again.")
    if reason != "stop":
        raise RequestError(f"{config.label} could not finish a reply. Try another message.")
    text = "".join(chunks)
    if not text.strip():
        raise RequestError(f"{config.label} returned an empty reply. Try again.")
    return text


def check_online_connection(api_key, config):
    """Generate a small structured dialogue reply without touching simulation state."""
    from pydantic import BaseModel, Field
    from ai.contracts import PreparedRequest, structured_schema
    from constants import ONLINE_CONNECTION_TIMEOUT_SECONDS

    class ConnectionReply(BaseModel):
        reply: str = Field(min_length=1)

    prepared = PreparedRequest(
        "Write a single short, original greeting as a friendly fictional character. Return JSON with a reply field.",
        [{"role": "user", "content": "Greet someone you have just met."}],
        structured_schema(ConnectionReply), ConnectionReply,
    )
    with provider_errors("online", config.model, config):
        raw = request_online(api_key, prepared, TransportSettings(ONLINE_CONNECTION_TIMEOUT_SECONDS, None, {}), config)
        reply = ConnectionReply.model_validate_json(raw).reply
        if not reply.strip():
            raise RequestError("The connection returned an empty dialogue reply.")
    return "Connected. Received a JSON dialogue reply."


def request_ollama(prepared, settings, model):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": prepared.system_instruction + "\nReturn only JSON matching this schema: "
                + json.dumps(prepared.response_schema, ensure_ascii=False, separators=(",", ":")),
            },
            *_transport_messages(prepared),
        ],
        "format": prepared.response_schema,
        "stream": streaming_output(),
        "keep_alive": "15m",
        # The profiles, rules, schema, and dialogue exceed Ollama's
        # default context. Truncation can remove the speaker rules.
        "options": settings.ollama_options,
    }

    def check_response(response):
        if response.status_code == 404:
            raise RequestError(f"Ollama could not find {model}. Run: ollama pull {model}")
        if response.is_error:
            raise RequestError(f"Ollama could not process this request. Check that {model} is available.")

    # Each call (including rewrites and repairs) may wait in Ollama again.
    # The chat API confirms generation only once it streams actual tokens.
    record_output("status", "waiting_for_ollama")
    if streaming_output():
        deadline = monotonic() + settings.timeout
        chunks, data = [], {}
        repetition = _StreamRepetitionGuard()
        receiving = False
        with httpx.stream("POST", OLLAMA_URL, json=payload, timeout=settings.timeout) as response, cancellable_stream(lambda: disconnect_response(response)):
            check_response(response)
            for line in response.iter_lines():
                if monotonic() > deadline:
                    raise RequestError(f"Ollama timed out. Try again after {model} finishes loading.")
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise RequestError(f"Ollama could not process this request. Check that {model} is available.")
                part = data.get("message", {})
                if not receiving and (part.get("thinking") or part.get("content")):
                    receiving = True
                    record_output("status", "running")
                record_output("reasoning", part.get("thinking", ""))
                repetition.add("reasoning", part.get("thinking", ""))
                text = part.get("content", "")
                record_output("output", text)
                repetition.add("output", text)
                chunks.append(text)
                if data.get("done"):
                    break
        content = "".join(chunks)
    else:
        response = httpx.post(OLLAMA_URL, json=payload, timeout=settings.timeout)
        check_response(response)
        data = response.json()
        content = data["message"]["content"]
        record_output("status", "running")
    if not data.get("done", False) or data.get("done_reason") not in (None, "stop"):
        raise RequestError("Ollama returned an incomplete reply. Try again.")
    return content
