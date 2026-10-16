import queue
import threading

from pydantic import ValidationError

from llm import RequestError, request_response, validate_response
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
    def __init__(self):
        self.worker = None
        self.pending_session = None
        self.results = queue.Queue()

    @property
    def busy(self):
        # Keep the slot reserved until the queue is polled AND the thread exits.
        return self.worker is not None

    def submit(self, session_id, provider, api_key, message, snapshot, *, request_id=None, ollama_model=DEFAULT_OLLAMA_MODEL):
        if self.busy:
            return False
        self.pending_session = session_id
        self.worker = threading.Thread(
            target=_request_worker,
            args=(session_id, request_id, provider, api_key, message, snapshot, self.results, ollama_model),
            daemon=True,
            name="NPC-model-request",
        )
        try:
            self.worker.start()
        except RuntimeError:
            self.worker = None
            self.pending_session = None
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
            except (ValidationError, ValueError, TypeError):
                replies.append((request_id, None, "The model returned an invalid action or reply. Try again."))
                continue
            replies.append((request_id, result, None))
        if self.worker is not None and not self.worker.is_alive():
            # A result may arrive after draining the queue. Leave it for the
            # next frame before freeing the request slot.
            if self.results.empty():
                self.worker = None
                self.pending_session = None
        return replies
