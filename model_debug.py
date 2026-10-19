"""Per-request diagnostics, isolated from simulation state and provider credentials."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
from pathlib import Path
from socket import SHUT_RDWR
from threading import Event, Lock, Thread
from time import monotonic
from uuid import uuid4

from constants import MODEL_LOG_FLUSH_SECONDS


LOG_DIRECTORY = Path(__file__).resolve().parent / "logs" / "model_requests"
_current_log = ContextVar("model_request_log", default=None)
_current_control = ContextVar("model_request_control", default=None)


class RequestCancelled(BaseException):
    """Leave provider/recovery loops immediately without retrying cancellation."""


class RequestControl:
    def __init__(self):
        self.cancelled = Event()
        self._lock = Lock()
        self._close = None

    def check(self):
        if self.cancelled.is_set():
            raise RequestCancelled()

    def cancel(self):
        self.cancelled.set()
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
    try:
        yield
    finally:
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
        self._pending = [{"metadata": self.metadata}]
        self._next_flush = 0
        self.revision = 0
        self.save_error = ""

    def _append(self, event):
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
            if now >= self._next_flush or kind in ("error", "finished", "disposition"):
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
