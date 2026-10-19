import queue
import threading
from dataclasses import dataclass
from time import monotonic

from pydantic import ValidationError

from constants import DIALOGUE_TIMEOUT_SECONDS, MAX_LLM_REQUESTS
from llm import (MODEL, ConversationTurn, InvalidResponseError, RequestError, VoiceRewriteResponse,
                 VoiceRewriteValidationError, request_response,
                 request_voice_rewrite, validate_response, validate_voice_rewrite, voice_rewrite_snapshot)
from ollama_manager import DEFAULT_OLLAMA_MODEL
from model_debug import DebugLogs, RequestLog, RequestCancelled, RequestControl, capture_output, record_output


@dataclass
class RequestWorker:
    thread: threading.Thread
    session_id: int
    slot: int
    participants: str
    topic: str
    started_at: float
    stage: str = "Starting request"
    finished_at: float | None = None
    log: RequestLog | None = None
    request_id: str | None = None
    control: RequestControl | None = None


def _request_description(snapshot):
    kind = snapshot.get("request_kind")
    if kind == "conversation":
        initiator, recipient = snapshot["initiator"], snapshot["recipient"]
        participants = f"{initiator['npc_name']} & {recipient['npc_name']}"
        purpose = snapshot.get("purpose", "chat").replace("_", " ").title()
        topic = initiator.get("private_context", {}).get("immediate_objective") or snapshot.get("topic", "")
        return participants, f"{purpose}: {topic}" if topic else purpose
    if kind == "snake_encounter":
        return (f"{snapshot['snake']['name']} & {snapshot['resident']['npc_name']}",
                snapshot["purpose"].replace("_", " ").title())
    name = snapshot["snake"]["name"] if kind == "snake_player" else snapshot.get("npc_name", "NPC")
    return f"You & {name}", "Player conversation"


def _request_worker(session_id, request_id, provider, api_key, message, snapshot, results, progress, ollama_model, log=None, control=None):
    """Queue simulation results; diagnostics have their own synchronized log."""
    deadline = monotonic() + DIALOGUE_TIMEOUT_SECONDS
    control = control or RequestControl()

    def report(stage):
        nonlocal deadline
        control.check()
        # Match the interaction's progress-based inactivity deadline across calls.
        deadline = monotonic() + DIALOGUE_TIMEOUT_SECONDS
        progress.put((session_id, request_id, stage, None, control))
        record_output("stage", stage)

    stage = "Request failed"
    with capture_output(log, control):
        try:
            record_output("status", "running")
            report("Starting request")
            result = request_response(provider, api_key, message, snapshot, ollama_model=ollama_model, progress=report)
            control.check()
            rewritten = []
            if snapshot.get("_voice_rewrites", True):
                rewritten = request_voice_rewrite(provider, api_key, result, snapshot, deadline=deadline,
                                                  ollama_model=ollama_model, progress=report)
            else:
                record_output("stage", "Voice rewrites disabled; using original dialogue.")
            control.check()
            raw = result.model_dump_json()
            record_output("status", "completed")
            results.put((session_id, request_id, raw, None, snapshot, [], rewritten, control))
            stage = "Reply ready"
        except RequestCancelled:
            stage = "Cancelled by player"
        except RequestError as exc:
            if control.cancelled.is_set():
                stage = "Cancelled by player"
            else:
                record_output("error", str(exc))
                results.put((session_id, request_id, None, str(exc), snapshot,
                             [turn.model_dump() for turn in exc.failed_turns], [], control))
        except Exception:
            error = "Could not get a reply from the model. Please try again."
            record_output("error", error)
            if not control.cancelled.is_set():
                results.put((session_id, request_id, None, error, snapshot, [], [], control))
            else:
                stage = "Cancelled by player"
        finally:
            if stage != "Reply ready":
                record_output("status", "failed")
            record_output("finished", stage)
            progress.put((session_id, request_id, stage, monotonic(), control))


class RequestManager:
    def __init__(self, max_requests=MAX_LLM_REQUESTS):
        self.max_requests = max_requests
        self.workers = {}
        self.results = queue.Queue()
        self.progress = queue.Queue()
        self.debug_logs = DebugLogs()
        self.logs = {}

    def prepare(self, session_id, provider, snapshot, *, request_id=None, ollama_model=DEFAULT_OLLAMA_MODEL):
        key = (session_id, request_id)
        if key not in self.logs:
            participants, topic = _request_description(snapshot)
            log = self.debug_logs.create(session_id=session_id, request_id=request_id, slot=None,
                                         provider=provider, model=ollama_model if provider == "ollama" else MODEL,
                                         participants=participants, topic=topic)
            self.logs[key] = log
            log.record("status", "queued")
        return self.logs[key]

    def finish(self, session_id, request_id, error=None):
        log = self.logs.get((session_id, request_id))
        if log is not None:
            log.record("disposition", f"Rejected by the game: {error}" if error else "Accepted by the game.")
            log.record("status", "failed" if error else "completed")

    def expire_queued(self, session_id, request_ids):
        for key, log in self.logs.items():
            if log.status == "queued" and key not in self.workers and (key[0] != session_id or key[1] not in request_ids):
                self.finish(*key, error="The conversation ended before a worker became available.")

    @property
    def busy(self):
        return self.available_slots == 0

    @property
    def available_slots(self):
        # Keep slots reserved until replies are polled and their workers exit.
        return max(0, self.max_requests - len(self.workers))

    def statuses(self):
        # Read progress even while paused, without applying any completed replies.
        while True:
            try:
                session_id, request_id, stage, finished_at, control = self.progress.get_nowait()
            except queue.Empty:
                break
            worker = self.workers.get((session_id, request_id))
            if worker is not None and worker.control is control:
                worker.stage = stage
                worker.finished_at = finished_at
        return tuple(self.workers.values())

    def submit(self, session_id, provider, api_key, message, snapshot, *, request_id=None,
               ollama_model=DEFAULT_OLLAMA_MODEL, voice_rewrites=True, dialogue_validation=True):
        key = (session_id, request_id)
        if self.busy or key in self.workers:
            return False
        snapshot = snapshot.copy()
        snapshot["_voice_rewrites"] = voice_rewrites
        snapshot["_dialogue_validation"] = dialogue_validation
        participants, topic = _request_description(snapshot)
        occupied = {worker.slot for worker in self.workers.values()}
        slot = next(slot for slot in range(1, self.max_requests + 1) if slot not in occupied)
        log = self.prepare(session_id, provider, snapshot, request_id=request_id, ollama_model=ollama_model)
        log.record("slot", str(slot))
        control = RequestControl()
        worker = threading.Thread(
            target=_request_worker,
            args=(session_id, request_id, provider, api_key, message, snapshot, self.results, self.progress, ollama_model, log, control),
            daemon=True,
            name=f"NPC-model-request-{slot}",
        )
        self.workers[key] = RequestWorker(worker, session_id, slot, participants, topic, monotonic(),
                                          log=log, request_id=request_id, control=control)
        try:
            worker.start()
        except RuntimeError:
            self.workers.pop(key)
            log.record("error", "Could not start the model worker.")
            log.record("status", "failed")
            return False
        return True

    def cancel(self, session_id, request_id):
        worker = self.workers.pop((session_id, request_id), None)
        if worker is None:
            log = self.logs.get((session_id, request_id))
            if log is None or log.status != "queued":
                return False
            self.finish(session_id, request_id, "Cancelled before the request started.")
            return True
        worker.control.cancel()
        worker.log.record("disposition", "Cancelled by player; discard this request and its effects.")
        worker.log.record("status", "failed")
        return True

    def cancel_all(self):
        """Disconnect outstanding requests and detach their old session queues."""
        for session_id, request_id in tuple(self.workers):
            self.cancel(session_id, request_id)
        for key, log in self.logs.items():
            if log.status == "queued":
                self.cancel(*key)
        self.logs.clear()
        self.results = queue.Queue()
        self.progress = queue.Queue()

    def poll(self, active_session):
        """Match replies to their session and request; discard old sessions."""
        self.statuses()
        replies = []
        while True:
            try:
                session_id, request_id, raw, error, snapshot, failed_turns, rewritten, control = self.results.get_nowait()
            except queue.Empty:
                break
            if control is not None and control.cancelled.is_set():
                continue
            worker = self.workers.get((session_id, request_id))
            log = worker.log if worker is not None else None

            def disposition(text):
                if log is not None:
                    log.record("disposition", text)

            if session_id != active_session:
                disposition("Discarded: request belongs to a previous session.")
                self.finish(session_id, request_id, "Request belongs to a previous session.")
                continue
            if error:
                self.finish(session_id, request_id, error)
                replies.append((request_id, None, error, [ConversationTurn.model_validate(turn) for turn in failed_turns], []))
                continue
            try:
                # Revalidate at the main-thread boundary before touching the world.
                result = validate_response(raw, snapshot)
            except InvalidResponseError as exc:
                disposition(f"Rejected at main-thread validation: {exc}")
                self.finish(session_id, request_id, str(exc))
                replies.append((request_id, None, f"The model's reply failed validation: {exc}", exc.failed_turns, []))
                continue
            except (ValidationError, ValueError, TypeError):
                disposition("Rejected at main-thread validation: invalid action or reply.")
                self.finish(session_id, request_id, "Invalid action or reply.")
                replies.append((request_id, None, "The model returned an invalid action or reply. Try again.", [], []))
                continue
            for reason in result._dialogue_errors:
                disposition(f"Base dialogue failed validation; retaining dialogue and effects: {reason}")
            if rewritten:
                try:
                    state = voice_rewrite_snapshot(result, snapshot)
                    rewritten = validate_voice_rewrite(VoiceRewriteResponse(turns=rewritten).model_dump_json(),
                                                       state["turns"], speakers=state["speakers"]).turns
                except VoiceRewriteValidationError as exc:
                    disposition(f"Voice rewrite failed validation; retaining generated text: {exc}")
                    rewritten = exc.response.turns
                except (ValidationError, ValueError, TypeError, AttributeError) as exc:
                    disposition(f"Voice rewrite rejected; using original dialogue: {exc}")
                    rewritten = []
            replies.append((request_id, result, None, [], rewritten))
            disposition("Validated and returned to the simulation for live-state checks." if snapshot.get("_dialogue_validation", True)
                        else "Base dialogue coherence checks disabled; parsed reply returned for live-state checks.")
        # A result may arrive after draining the queue; poll it next frame
        # before freeing the slot that belongs to that result.
        if self.results.empty():
            self.workers = {key: worker for key, worker in self.workers.items() if worker.thread.is_alive()}
        return replies
