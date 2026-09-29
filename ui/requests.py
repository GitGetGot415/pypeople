import queue
import threading

from pydantic import ValidationError

from llm import ActionResponse, RequestError, request_action


def _request_worker(session_id, api_key, message, snapshot, results):
    """Only communicate through the queue; never touch Pygame or live state."""
    try:
        result = request_action(api_key, message, snapshot)
        results.put((session_id, result.model_dump_json(), None))
    except RequestError as exc:
        results.put((session_id, None, str(exc)))
    except Exception:
        results.put((session_id, None, "Could not get a reply from Gemini. Please try again."))


class RequestManager:
    def __init__(self):
        self.worker = None
        self.pending_session = None
        self.results = queue.Queue()

    @property
    def busy(self):
        # Keep the slot reserved until the queue is polled AND the thread exits.
        return self.worker is not None

    def submit(self, session_id, api_key, message, snapshot):
        if self.busy:
            return False
        self.pending_session = session_id
        self.worker = threading.Thread(
            target=_request_worker,
            args=(session_id, api_key, message, snapshot, self.results),
            daemon=True,
            name="Judy-Gemini-request",
        )
        try:
            self.worker.start()
        except RuntimeError:
            self.worker = None
            self.pending_session = None
            return False
        return True

    def poll(self, active_session):
        """Return (response, error) pairs for the active game; discard old replies."""
        replies = []
        while True:
            try:
                session_id, raw, error = self.results.get_nowait()
            except queue.Empty:
                break
            if session_id != active_session:
                continue
            if error:
                replies.append((None, error))
                continue
            try:
                # Revalidate at the main-thread boundary before touching the world.
                result = ActionResponse.model_validate_json(raw)
                if not result.reply.strip():
                    raise ValueError("Empty reply")
            except (ValidationError, ValueError, TypeError):
                replies.append((None, "Gemini returned an invalid action or reply. Try again."))
                continue
            replies.append((result, None))
        if self.worker is not None and not self.worker.is_alive():
            # A result may arrive after draining the queue. Leave it for the
            # next frame before freeing the request slot.
            if self.results.empty():
                self.worker = None
                self.pending_session = None
        return replies
