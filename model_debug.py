"""Per-request diagnostics, isolated from simulation state and provider credentials."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
from pathlib import Path
from socket import SHUT_RDWR
from threading import Condition, Event, Lock, Thread
from time import monotonic
from uuid import uuid4

from constants import MODEL_LOG_FLUSH_SECONDS, DAY_SKIP_CANCEL_BEFORE_TURN, DAY_SKIP_WRAP_UP_TURNS


LOG_DIRECTORY = Path(__file__).resolve().parent / "logs" / "model_requests"
_current_log = ContextVar("model_request_log", default=None)
_current_control = ContextVar("model_request_control", default=None)


class RequestCancelled(BaseException):
    """Leave provider/recovery loops immediately without retrying cancellation."""


class ProviderAdmission:
    """Bound remote calls, allowing logical exchanges to yield between calls."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.condition = Condition()
        self.waiting = []
        self.active = 0
        self.sequence = 0
        self.players = set()

    def register(self, control):
        if control.priority == 0:
            with self.condition:
                self.players.add(control)
                self.condition.notify_all()

    def unregister(self, control):
        with self.condition:
            self.players.discard(control)
            self.condition.notify_all()

    def resize(self, capacity):
        with self.condition:
            self.capacity = capacity
            self.condition.notify_all()

    def wake(self):
        with self.condition:
            self.condition.notify_all()

    @contextmanager
    def call(self, control):
        queued_at = monotonic()
        control.waiting_since = queued_at
        with self.condition:
            ticket = (control.priority, self.sequence, control)
            self.sequence += 1
            self.waiting.append(ticket)
            self.condition.notify_all()
            control.stage = "Waiting for provider admission"
            record_output("stage", control.stage)
            record_output("status", "queued")
            try:
                while (self.active >= self.capacity or ticket != min(self.waiting, key=lambda t: t[:2])
                       or control.priority > 0 and any(not p.cancelled.is_set() for p in self.players)):
                    control.check()
                    self.condition.wait()
                control.check()
                self.active += 1
                control.wait_seconds += monotonic() - queued_at
                control.waiting_since = None
                control.provider_active = True
                control.stage = "Generating"
                record_output("stage", control.stage)
                record_output("status", "running")
            finally:
                if control.waiting_since is not None:
                    control.wait_seconds += monotonic() - queued_at
                    control.waiting_since = None
                self.waiting.remove(ticket)
                self.condition.notify_all()
        try:
            yield
        finally:
            with self.condition:
                self.active -= 1
                control.provider_active = False
                control.stage = "Between provider calls"
                self.condition.notify_all()


def admitted_provider(function):
    """Provider boundary; prompts and repair functions inherit request context."""
    from functools import wraps
    @wraps(function)
    def admitted(*args, **kwargs):
        control = _current_control.get()
        if control is None or control.admission is None:
            return function(*args, **kwargs)
        with control.admission.call(control):
            return function(*args, **kwargs)
    return admitted


class RequestControl:
    def __init__(self, admission=None, priority=1):
        self.cancelled = Event()
        self._lock = Lock()
        self._close = None
        self.admission = admission
        self.priority = priority
        self.provider_active = False
        self.stage = "Starting request"
        self.wait_seconds = 0.0
        self.waiting_since = None

        self.conversation_turn = 0
        self.conversation_turns_done = False
        self.conversation_finish_turn = None

    def wrap_up_conversation(self):
        """Freeze a closing deadline, or cancel a chat that has barely started."""
        with self._lock:
            if self.conversation_turns_done:
                return True
            if self.conversation_turn < DAY_SKIP_CANCEL_BEFORE_TURN:
                # Stop the worker before it can start another turn. The request
                # manager then disconnects its stream using ordinary cancellation.
                self.cancelled.set()
                return False
            if self.conversation_finish_turn is None:
                # Even with a zero cutoff, let both speakers have a turn.
                self.conversation_finish_turn = max(2, self.conversation_turn + max(1, DAY_SKIP_WRAP_UP_TURNS))
            return True

    def start_conversation_turn(self, turn):
        with self._lock:
            self.check()
            self.conversation_turn = turn
            return self.conversation_finish_turn is not None and turn >= self.conversation_finish_turn

    def finish_conversation_turns(self):
        with self._lock:
            self.check()
            self.conversation_turns_done = True

    @property
    def queued_seconds(self):
        return self.wait_seconds + (monotonic() - self.waiting_since if self.waiting_since is not None else 0.0)

    def check(self):
        if self.cancelled.is_set():
            raise RequestCancelled()

    def cancel(self):
        self.cancelled.set()
        if self.admission is not None:
            self.admission.wake()
        with self._lock:
            close = self._close
        if close is not None:
            # Closing a blocked socket must never block pygame's event thread.
            def disconnect():
                try:
                    close()
                except Exception:
                    pass
            Thread(target=disconnect, daemon=True, name="Cancel-model-stream").start()


def disconnect_response(response):
    """Interrupt a blocked streaming read before releasing its HTTP connection."""
    stream = response.extensions.get("network_stream")
    connection = stream.get_extra_info("socket") if stream is not None else None
    if connection is not None:
        try:
            connection.shutdown(SHUT_RDWR)
        except OSError:
            pass
    response.close()


def watch_response(response):
    """SDK response hook: make its active HTTP stream cancellable too."""
    control = _current_control.get()
    if control is not None:
        with control._lock:
            control._close = lambda: disconnect_response(response)
        control.check()


@contextmanager
def cancellable_stream(close):
    control = _current_control.get()
    if control is None:
        yield
        return
    with control._lock:
        control._close = close
    try:
        control.check()
        yield
        control.check()
    finally:
        with control._lock:
            control._close = None


def streaming_output():
    return _current_log.get() is not None


def check_cancelled():
    control = _current_control.get()
    if control is not None:
        control.check()


def start_conversation_turn(turn):
    control = _current_control.get()
    return control.start_conversation_turn(turn) if control is not None else False


def finish_conversation_turns():
    control = _current_control.get()
    if control is not None:
        control.finish_conversation_turns()


def provider_wait_seconds():
    control = _current_control.get()
    return control.wait_seconds if control is not None else 0.0


def record_output(kind, text):
    control = _current_control.get()
    if control is not None and kind in ("call", "output", "reasoning"):
        control.check()
    log = _current_log.get()
    if log is not None and text:
        log.record(kind, str(text))


@contextmanager
def capture_output(log, control=None):
    token = _current_log.set(log)
    control_token = _current_control.set(control)
    if control is not None and control.admission is not None:
        control.admission.register(control)
    try:
        yield
    finally:
        if control is not None and control.admission is not None:
            control.admission.unregister(control)
        _current_log.reset(token)
        _current_control.reset(control_token)
        if log is not None:
            log.flush()


class RequestLog:
    def __init__(self, directory, **metadata):
        self.metadata = {"started": datetime.now(timezone.utc).isoformat(), **metadata}
        self.path = Path(directory) / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex}.jsonl"
        self.started_at = monotonic()
        self._lock = Lock()
        self._events = []
        self._status = "queued"
        self._pending = [{"metadata": self.metadata}]
        self._next_flush = 0
        self.revision = 0
        self.save_error = ""

    def _append(self, event):
        if event["kind"] == "status" and event["text"] in ("queued", "waiting_for_ollama", "running", "completed", "failed"):
            # Cancellation or game rejection wins over a worker finishing late.
            if self._status != "failed":
                self._status = event["text"]
        elif event["kind"] == "slot" and event["text"].isdigit():
            self.metadata["slot"] = int(event["text"])
        # Coalesce tokens for display while retaining their original disk events.
        if self._events and event["kind"] in ("output", "reasoning") and self._events[-1]["kind"] == event["kind"]:
            self._events[-1] = {**self._events[-1], "text": self._events[-1]["text"] + event["text"]}
        else:
            self._events.append(event)
        self.revision += 1

    def record(self, kind, text):
        now = monotonic()
        event = {"elapsed": round(max(0, now - self.started_at), 3), "kind": kind, "text": text}
        with self._lock:
            self._append(event)
            if not self.save_error:
                self._pending.append(event)
            if now >= self._next_flush or kind in ("error", "finished", "disposition", "status", "slot"):
                self._flush()
                self._next_flush = now + MODEL_LOG_FLUSH_SECONDS

    def _flush(self):
        if not self._pending or self.save_error:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as output:
                for event in self._pending:
                    output.write(json.dumps(event) + "\n")
        except OSError:
            self.save_error = "Could not save this log; check the logs folder permissions."
        self._pending.clear()

    def flush(self):
        with self._lock:
            self._flush()

    def snapshot(self):
        with self._lock:
            return self.revision, tuple(self._events), self.save_error

    @property
    def status(self):
        with self._lock:
            return self._status

    @property
    def status_label(self):
        status = self.status
        return "Waiting for Ollama" if status == "waiting_for_ollama" else status.title()

    @classmethod
    def load(cls, path):
        with path.open(encoding="utf-8") as source:
            metadata = json.loads(next(source))["metadata"]
            for name in ("started", "participants", "provider", "model", "topic"):
                if not isinstance(metadata.get(name), str):
                    raise ValueError("Invalid log metadata.")
            for name in ("slot", "session_id", "request_id"):
                if name not in metadata:
                    raise ValueError("Missing log metadata.")
            log = cls(path.parent, **metadata)
            log.path = path
            log._pending.clear()
            for line in source:
                try:
                    event = json.loads(line)
                    if (isinstance(event.get("text"), str) and isinstance(event.get("kind"), str)
                            and isinstance(event.get("elapsed"), (int, float))):
                        log._append(event)
                except (ValueError, AttributeError):
                    continue  # A crash can leave a partial last line.
            if not any(event["kind"] == "status" for event in log._events):
                # Older logs lack explicit states. Recover final outcomes without
                # treating a failed rewrite with usable original dialogue as fatal.
                finished = [event["text"] for event in log._events if event["kind"] == "finished"]
                rejected = any(event["kind"] == "disposition" and event["text"].startswith(
                    ("Rejected", "Discarded", "Cancelled")) for event in log._events)
                log._status = "completed" if finished and finished[-1] == "Reply ready" and not rejected else "failed"
            elif log._status in ("queued", "waiting_for_ollama", "running"):
                # A saved request cannot still be running after a restart.
                log._status = "failed"
        return log


class DebugLogs:
    def __init__(self, directory=LOG_DIRECTORY):
        self.directory = Path(directory)
        self.logs = []
        self.read_error = ""
        self._loaded = False

    def create(self, **metadata):
        log = RequestLog(self.directory, **metadata)
        self.logs.append(log)
        return log

    def history(self):
        if not self._loaded:
            self._loaded = True
            known = {log.path for log in self.logs}
            try:
                for path in self.directory.glob("*.jsonl"):
                    if path not in known:
                        try:
                            self.logs.append(RequestLog.load(path))
                        except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration):
                            self.read_error = "Some saved logs could not be read."
            except OSError:
                self.read_error = "Could not read the model logs folder."
        return sorted(self.logs, key=lambda log: log.metadata["started"], reverse=True)
