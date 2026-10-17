import queue
import threading

from pydantic import ValidationError

from llm import InvalidResponseError, RequestError, request_response, validate_response
from ollama_manager import DEFAULT_OLLAMA_MODEL


def _request_worker(session_id, request_id, provider, api_key, message, snapshot, results, ollama_model):
    """Only communicate through the queue; never touch Pygame or live state."""
    try:
        result = request_response(provider, api_key, message, snapshot, ollama_model=ollama_model)
        results.put((session_id, request_id, result.model_dump_json(), None, snapshot))
    except RequestError as exc:
        results.put((session_id, request_id, None, str(exc), snapshot))
    except Exception:
        results.put((session_id, request_id, None, "Could not get a reply from the model. Please try again.", snapshot))


class RequestManager:
    def __init__(self, max_requests=2):
        self.max_requests = max_requests
        self.workers = {}
        self.results = queue.Queue()

    @property
    def busy(self):
        return self.available_slots == 0

    @property
    def available_slots(self):
        # Keep slots reserved until replies are polled and their workers exit.
        return max(0, self.max_requests - len(self.workers))

    def submit(self, session_id, provider, api_key, message, snapshot, *, request_id=None, ollama_model=DEFAULT_OLLAMA_MODEL):
        key = (session_id, request_id)
        if self.busy or key in self.workers:
            return False
        worker = threading.Thread(
            target=_request_worker,
            args=(session_id, request_id, provider, api_key, message, snapshot, self.results, ollama_model),
            daemon=True,
            name="NPC-model-request",
        )
        self.workers[key] = worker
        try:
            worker.start()
        except RuntimeError:
            self.workers.pop(key)
            return False
        return True

    def poll(self, active_session):
        """Match replies to their session and request; discard old sessions."""
        replies = []
        while True:
            try:
                session_id, request_id, raw, error, snapshot = self.results.get_nowait()
            except queue.Empty:
                break
            if session_id != active_session:
                continue
            if error:
                replies.append((request_id, None, error))
                continue
            try:
                # Revalidate at the main-thread boundary before touching the world.
                result = validate_response(raw, snapshot)
            except InvalidResponseError as exc:
                replies.append((request_id, None, f"The model's reply failed validation: {exc}"))
                continue
            except (ValidationError, ValueError, TypeError):
                replies.append((request_id, None, "The model returned an invalid action or reply. Try again."))
                continue
            replies.append((request_id, result, None))
        # A result may arrive after draining the queue; poll it next frame
        # before freeing the slot that belongs to that result.
        if self.results.empty():
            self.workers = {key: worker for key, worker in self.workers.items() if worker.is_alive()}
        return replies
